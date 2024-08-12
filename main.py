import os
import time
import json
import boto3
import requests
import noisereduce as nr
import numpy as np
import spacy
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from result import router as result_router
from pydub import AudioSegment
from pydub.effects import normalize
from pydub.silence import split_on_silence

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

app.include_router(result_router)

# Load spaCy model for NER
nlp = spacy.load('en_core_web_sm')

def upload_to_s3(local_file_path, bucket_name, s3_file_path):
    s3 = boto3.client('s3')
    s3.upload_file(local_file_path, bucket_name, s3_file_path)
    return f"s3://{bucket_name}/{s3_file_path}"

def reduce_noise(audio_segment):
    audio_data = np.array(audio_segment.get_array_of_samples())
    reduced_noise_audio_data = nr.reduce_noise(y=audio_data, sr=audio_segment.frame_rate)
    return AudioSegment(reduced_noise_audio_data.tobytes(), frame_rate=audio_segment.frame_rate, sample_width=audio_segment.sample_width, channels=audio_segment.channels)

def normalize_audio(audio_segment):
    return normalize(audio_segment)

def segment_audio(audio_segment, min_silence_len=1000, silence_thresh=-16):
    return split_on_silence(audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

def process_audio(file_path):
    audio = AudioSegment.from_file(file_path)

    # Reduce noise
    audio = reduce_noise(audio)

    # Normalize audio
    audio = normalize_audio(audio)

    # Segment audio
    segments = segment_audio(audio)

    # Save segments to temporary files
    segment_files = []
    for i, segment in enumerate(segments):
        segment_file = f"temp_segment_{i}.mp3"
        segment.export(segment_file, format="mp3")
        segment_files.append(segment_file)

    return segment_files

def transcribe_audio_with_custom_vocab(local_file_path, bucket_name, vocabulary_name):
    transcribe = boto3.client('transcribe')
    
    job_name = f"{os.path.basename(local_file_path).split('.')[0]}-{int(time.time())}"
    s3_file_path = job_name + os.path.splitext(local_file_path)[1]
    job_uri = upload_to_s3(local_file_path, bucket_name, s3_file_path)

    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': job_uri},
        MediaFormat='mp3',
        LanguageCode='en-US',
        Settings={
            'ShowSpeakerLabels': True, 
            'MaxSpeakerLabels': 2,
            'VocabularyName': vocabulary_name
        }
    )

    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
        time.sleep(10)
    
    if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
        response = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        transcript_url = response['TranscriptionJob']['Transcript']['TranscriptFileUri']
        transcript = json.loads(requests.get(transcript_url).text)
        results = transcript['results']
        return results
    
    return None

def extract_names_from_transcript(transcript_results):
    transcript_text = ' '.join(item['alternatives'][0]['content'] for item in transcript_results['items'] if item['type'] == 'pronunciation')
    doc = nlp(transcript_text)
    names = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
    return names

def chunk_text(text, max_bytes):
    chunks = []
    current_chunk = ""
    current_chunk_bytes = 0
    
    words = text.split()
    for word in words:
        word_bytes = len(word.encode('utf-8')) + 1
        
        if current_chunk_bytes + word_bytes > max_bytes:
            chunks.append(current_chunk.strip())
            current_chunk = ""
            current_chunk_bytes = 0
        
        current_chunk += word + " "
        current_chunk_bytes += word_bytes
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def analyze_transcript_chunks_detailed(chunks):
    comprehend = boto3.client('comprehend')
    detailed_sentiments = []
    
    for chunk in chunks:
        response = comprehend.detect_sentiment(
            Text=chunk,
            LanguageCode='en'
        )
        detailed_sentiments.append(response['Sentiment'])
    
    return detailed_sentiments

def analyze_sentiment_per_speaker(transcript_results):
    comprehend = boto3.client('comprehend')
    speaker_texts = {}
    
    for item in transcript_results['items']:
        if item['type'] == 'pronunciation':
            speaker_label = item['speaker_label']
            word = item['alternatives'][0]['content']

            if speaker_label not in speaker_texts:
                speaker_texts[speaker_label] = ""
            
            speaker_texts[speaker_label] += word + " "
    
    speaker_sentiments = {}
    
    for speaker, text in speaker_texts.items():
        chunks = chunk_text(text, max_bytes=4500)
        sentiment_scores = []
        for chunk in chunks:
            response = comprehend.detect_sentiment(
                Text=chunk,
                LanguageCode='en'
            )
            sentiment_scores.append(response['SentimentScore'])

        aggregated_scores = {
            'Positive': sum(score['Positive'] for score in sentiment_scores) / len(sentiment_scores),
            'Negative': sum(score['Negative'] for score in sentiment_scores) / len(sentiment_scores),
            'Neutral': sum(score['Neutral'] for score in sentiment_scores) / len(sentiment_scores),
            'Mixed': sum(score['Mixed'] for score in sentiment_scores) / len(sentiment_scores)
        }
        speaker_sentiments[speaker] = [aggregated_scores]
    
    return speaker_sentiments

def rate_conversation(sentiments):
    sentiment_counts = {'POSITIVE': 0, 'MIXED': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
    
    for sentiment in sentiments:
        sentiment_counts[sentiment] += 1
    
    max_sentiment = max(sentiment_counts, key=sentiment_counts.get)
    
    if max_sentiment == 'POSITIVE':
        rating = 5
    elif max_sentiment == 'MIXED':
        rating = 3
    elif max_sentiment == 'NEGATIVE':
        rating = 1
    else:
        rating = 2  # NEUTRAL
    
    return rating

def format_transcript(transcript_results):
    speaker_map = {}
    formatted_transcript = ""
    current_speaker = None
    current_paragraph = []

    for item in transcript_results['items']:
        if item['type'] == 'pronunciation':
            speaker_label = item['speaker_label']
            word = item['alternatives'][0]['content']

            if speaker_label not in speaker_map:
                speaker_map[speaker_label] = speaker_label
            
            if speaker_label != current_speaker:
                if current_speaker is not None:
                    formatted_transcript += f"{speaker_map[current_speaker]}: {' '.join(current_paragraph)}\n\n"
                current_speaker = speaker_label
                current_paragraph = [word]
            else:
                current_paragraph.append(word)
    
    if current_paragraph:
        formatted_transcript += f"{speaker_map[current_speaker]}: {' '.join(current_paragraph)}\n\n"

    return formatted_transcript

def save_transcript(transcript_text, filename):
    output_dir = "transcripts"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(transcript_text)
    
    return file_path

@app.get("/", response_class=HTMLResponse)
def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/transcribe_and_analyze")
async def transcribe_and_analyze(request: Request, file: UploadFile = File(...)):
    local_file_path = os.path.join("uploads", file.filename)
    bucket_name = "inoday-bedrock-chat-pdf"
    vocabulary_name = "MyVocab01"

    os.makedirs("uploads", exist_ok=True)

    with open(local_file_path, "wb") as f:
        f.write(await file.read())

    transcript_results = transcribe_audio_with_custom_vocab(local_file_path, bucket_name, vocabulary_name)
    
    if transcript_results:
        formatted_transcript = format_transcript(transcript_results)
        transcript_filename = f"{os.path.splitext(file.filename)[0]}.txt"
        transcript_file_path = save_transcript(formatted_transcript, transcript_filename)

        transcript_text = ' '.join(item['alternatives'][0]['content'] for item in transcript_results['items'] if item['type'] == 'pronunciation')
        chunks = chunk_text(transcript_text, max_bytes=4500)
        sentiments = analyze_transcript_chunks_detailed(chunks)
        sentiment_per_speaker = analyze_sentiment_per_speaker(transcript_results)
        conversation_rating = rate_conversation(sentiments)
        detected_names = extract_names_from_transcript(transcript_results)
        
        return JSONResponse(content={
            "transcript_file_path": transcript_file_path,
            "sentiment_per_speaker": sentiment_per_speaker,
            "conversation_rating": conversation_rating,
            "detected_names": detected_names
        })
    
    return JSONResponse(content={"error": "Transcription failed"}, status_code=500)
