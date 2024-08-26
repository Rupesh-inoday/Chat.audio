from fastapi import APIRouter, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import boto3
import os
import json
from botocore.exceptions import ClientError
import logging
import vertexai
from google.oauth2 import service_account
from vertexai.generative_models import GenerativeModel, SafetySetting

logging.basicConfig(level=logging.DEBUG)

router = APIRouter()

# Load AWS credentials from environment variables (Jamba)
aws_access_key_id = os.getenv("")
aws_secret_access_key = os.getenv("/bLONwDcKRTzRxFTQEHY+chwtuX0IxnCtQoTSOpD")
aws_region = "us-east-1"

# Initialize Vertex AI with credentials (Gemini)
key_path = 'inoday-retail-6d16d16543aa.json'
credentials = service_account.Credentials.from_service_account_file(
    key_path, scopes=['https://www.googleapis.com/auth/cloud-platform'])

PROJECT_ID = 'inoday-retail'
REGION = 'us-central1'
vertexai.init(project=PROJECT_ID, location=REGION, credentials=credentials)

templates = Jinja2Templates(directory="templates")

def converse_with_jamba(user_message: str, text_prompt: str, temperature: float = 0.2) -> str: 
    client = boto3.client(
        "bedrock-runtime",
        region_name=aws_region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    model_id = "ai21.jamba-instruct-v1:0"

    conversation = [
        {
            "role": "user",
            "content": f"Transcript: {user_message}",
        },
        {
            "role": "assistant",
            "content": f"Prompt: {text_prompt}",
        }
    ]

    try:
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps({
                "messages": conversation,
                "temperature": temperature  # Adding the temperature parameter
            })
        )

        response_body = response["body"].read().decode()
        model_response = json.loads(response_body)

        if "choices" not in model_response or len(model_response["choices"]) == 0:
            raise ValueError(f"Unexpected response format: {response_body}")

        response_text = model_response["choices"][0]["message"]["content"]
        return response_text

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        return f"ERROR: Can't invoke '{model_id}'. Reason: {e}"

def converse_with_gemini(user_message: str, text_prompt: str) -> str:
    # Define the prompt for Gemini
    prompt = f"""
    Analyze the conversation transcript and respond based on the following prompt.

    Transcript:
    {user_message}

    Prompt: {text_prompt}
    """

    # Initialize the Generative Model
    model = GenerativeModel("gemini-1.5-pro-001")

    # Define generation configuration
    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 1,
        "top_p": 0.95,
    }

    # Define safety settings
    safety_settings = [
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        ),
    ]

    # Generate the content
    responses = model.generate_content(
        [prompt],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )

    # Collect the responses
    generated_response = ""
    for response in responses:
        generated_response += response.text

    return generated_response

@router.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/generate", response_class=HTMLResponse)
async def generate(request: Request, file: str = Form(...), text_prompt: str = Form(...), model_choice: str = Form(...)):
    try:
        # Read the transcript from the file
        with open(f"transcripts/{file}", "r", encoding="utf-8") as f:
            user_message = f.read()

        # Generate response based on the chosen model
        if model_choice == "Jamba2":
            summary = converse_with_jamba(user_message, text_prompt)
        elif model_choice == "Gemini":
            summary = converse_with_gemini(user_message, text_prompt)
        else:
            raise ValueError("Invalid model choice selected.")

        print(f"Generated Summary: {summary}")  # Debug line to print the summary
        
        return templates.TemplateResponse("result.html", {"request": request, "evaluation_summary": summary})
    except Exception as e:
        print(f"Error processing file: {e}")  # Debug line to print the error
        return templates.TemplateResponse("index.html", {"request": request, "error": f"Error processing file: {e}"})

@router.post("/generate_qna", response_class=HTMLResponse)
async def generate_qna(request: Request, file: str = Form(...), qna_prompt: str = Form(...), model_choice: str = Form(...)):
    try:
        # Read the transcript from the file
        with open(f"transcripts/{file}", "r", encoding="utf-8") as f:
            user_message = f.read()

        # Generate Q&A based on the chosen model
        if model_choice == "Jamba2":
            qna_answers = converse_with_jamba(user_message, qna_prompt)
        elif model_choice == "Gemini":
            qna_answers = converse_with_gemini(user_message, qna_prompt)
        else:
            raise ValueError("Invalid model choice selected.")
        
        logging.debug(f"Q&A Answers: {qna_answers}")
        
        return templates.TemplateResponse("qna.html", {"request": request, "qna_answers": qna_answers})
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": f"Error processing file: {e}"})

@router.get("/chatbot", response_class=HTMLResponse)
async def chatbot_page(request: Request):
    transcript_file = request.query_params.get("transcript_file", None)
    return templates.TemplateResponse("chatbot.html", {"request": request, "transcript_file": transcript_file})

@router.post("/chat_with_transcript", response_class=HTMLResponse)
async def chat_with_transcript(request: Request, chat_prompt: str = Form(...), transcript_file: str = Form(...), model_choice: str = Form(...)):
    if not transcript_file:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Transcript file not found"})

    try:
        file_path = f"transcripts/{transcript_file}"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Transcript file '{file_path}' not found.")
        
        with open(file_path, "r", encoding="utf-8") as f:
            user_message = f.read()

        # Generate chat response based on the chosen model
        if model_choice == "Jamba2":
            chat_response = converse_with_jamba(user_message, chat_prompt)
        elif model_choice == "Gemini":
            chat_response = converse_with_gemini(user_message, chat_prompt)
        else:
            raise ValueError("Invalid model choice selected.")
        
        return templates.TemplateResponse("chatbot.html", {"request": request, "chat_response": chat_response})
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": f"Error processing file: {e}"})
