from fastapi import APIRouter, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import boto3
import os
import json
from botocore.exceptions import ClientError
import logging

logging.basicConfig(level=logging.DEBUG)

router = APIRouter()

# Load AWS credentials from environment variables (optional)
aws_access_key_id = os.getenv("")
aws_secret_access_key = os.getenv("")
aws_region = "us-east-1"

templates = Jinja2Templates(directory="templates")

def converse_with_model(user_message: str, text_prompt: str, temperature: float = 0.2) -> str: 
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

@router.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/generate", response_class=HTMLResponse)
async def generate(request: Request, file: str = Form(...), text_prompt: str = Form(...)):
    try:
        # Read the transcript from the file
        with open(f"transcripts/{file}", "r", encoding="utf-8") as f:
            user_message = f.read()
        
        # Generate the points and questions
        summary = converse_with_model(user_message, text_prompt)
        print(f"Generated Summary: {summary}")  # Debug line to print the summary
        
        return templates.TemplateResponse("result.html", {"request": request, "evaluation_summary": summary})
    except Exception as e:
        print(f"Error processing file: {e}")  # Debug line to print the error
        return templates.TemplateResponse("index.html", {"request": request, "error": f"Error processing file: {e}"})

@router.post("/generate_qna", response_class=HTMLResponse)
async def generate_qna(request: Request, file: str = Form(...), qna_prompt: str = Form(...)):
    try:
        # Read the transcript from the file
        with open(f"transcripts/{file}", "r", encoding="utf-8") as f:
            user_message = f.read()
        
        # Generate the Q&A
        qna_answers = converse_with_model(user_message, qna_prompt)
        
        # Log the value of qna_answers
        logging.debug(f"Q&A Answers: {qna_answers}")
        
        # Return the Q&A answers to the template
        return templates.TemplateResponse("qna.html", {"request": request, "qna_answers": qna_answers})
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": f"Error processing file: {e}"})
@router.get("/chatbot", response_class=HTMLResponse)
async def chatbot_page(request: Request):
    transcript_file = request.query_params.get("transcript_file", None)
    return templates.TemplateResponse("chatbot.html", {"request": request, "transcript_file": transcript_file})

@router.post("/chat_with_transcript", response_class=HTMLResponse)
async def chat_with_transcript(request: Request, chat_prompt: str = Form(...), transcript_file: str = Form(...)):
    if not transcript_file:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Transcript file not found"})

    try:
        file_path = f"transcripts/{transcript_file}"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Transcript file '{file_path}' not found.")
        
        with open(file_path, "r", encoding="utf-8") as f:
            user_message = f.read()
        
        chat_response = converse_with_model(user_message, chat_prompt)
        
        return templates.TemplateResponse("chatbot.html", {"request": request, "chat_response": chat_response})
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": f"Error processing file: {e}"})

