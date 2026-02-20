import os
import re
import json
import logging
import requests

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fastapi_app")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# Gemini API 設定（中身はいじらない）
# --------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY is not set")

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1/models/"
    "gemini-2.5-flash:generateContent?key=" + str(GEMINI_API_KEY)
)

# --------------------
# Models
# --------------------
class UnityRequest(BaseModel):
    text: str
    basePrompt: str

class Emotion(BaseModel):
    joy: float
    anger: float
    sadness: float
    fun: float

class UnityResponse(BaseModel):
    message: str
    emotion: Emotion

# --------------------
# Chat
# --------------------
@app.post("/chat", response_model=UnityResponse)
def chat(data: UnityRequest):

    if not GEMINI_API_KEY:
        return UnityResponse(
            message="APIキー未設定",
            emotion=Emotion(joy=0.0, anger=0.0, sadness=0.0, fun=0.0),
        )

    full_prompt = f"""
{data.basePrompt}

ユーザー入力:
{data.text}

必ず以下の JSON フォーマットのみで応答してください。
数値は 0.0 ～ 1.0 の範囲で指定してください。

{{
  "message": "string",
  "emotion": {{
    "joy": 0.0,
    "anger": 0.0,
    "sadness": 0.0,
    "fun": 0.0
  }}
}}
"""

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": full_prompt}
                ]
            }
        ]
    }

    try:
        resp = requests.post(
            GEMINI_URL,
            json=payload,
            timeout=20  # Free環境対策（少し短め）
        )
        resp.raise_for_status()

        gemini_json = resp.json()
        raw_text = gemini_json["candidates"][0]["content"]["parts"][0].get("text", "")

        if not raw_text:
            raise ValueError("Empty text")

        m = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not m:
            raise ValueError("JSON not found")

        parsed = json.loads(m.group())
        emotion = parsed.get("emotion", {})

        return UnityResponse(
            message=str(parsed.get("message", "")),
            emotion=Emotion(
                joy=float(emotion.get("joy", 0.0)),
                anger=float(emotion.get("anger", 0.0)),
                sadness=float(emotion.get("sadness", 0.0)),
                fun=float(emotion.get("fun", 0.0)),
            ),
        )

    except Exception:
        logger.exception("Error during /chat")
        return UnityResponse(
            message="エラーが発生しました",
            emotion=Emotion(joy=0.0, anger=0.0, sadness=0.0, fun=0.0),
        )

# --------------------
# UptimeRobot用ヘルスチェック
# --------------------
@app.get("/ping")
def ping():
    return JSONResponse(content={"status": "ok"}, status_code=200)