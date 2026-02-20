# main.py
import os
import re
import json
import logging
import requests

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --------------------
# logging
# --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fastapi_app")

# --------------------
# FastAPI app
# --------------------
app = FastAPI()

# --------------------
# CORS (Unity / local dev 用)
# --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# Gemini API 設定
# --------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY is not set")
    raise RuntimeError("GEMINI_API_KEY is not set")

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1/models/"
    "gemini-2.5-flash:generateContent?key=" + GEMINI_API_KEY
)

# --------------------
# request / response models
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
# main API
# --------------------
@app.post("/chat", response_model=UnityResponse)
def chat(data: UnityRequest):
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
            timeout=30
        )
        resp.raise_for_status()

        gemini_json = resp.json()

        if "candidates" not in gemini_json or not gemini_json["candidates"]:
            raise ValueError("No candidates in Gemini response")

        raw_text = gemini_json["candidates"][0]["content"]["parts"][0].get("text", "")
        if not raw_text:
            raise ValueError("Empty text from Gemini")

        # JSON 抽出（ネスト対応）
        m = re.search(
            r"\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}",
            raw_text,
            re.DOTALL
        )
        if not m:
            m = re.search(r"\{.*\}", raw_text, re.DOTALL)

        if not m:
            logger.error("JSON not found in Gemini response:\n%s", raw_text)
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

    except Exception as e:
        logger.exception("Error during /chat: %s", e)
        return UnityResponse(
            message="エラーが発生しました",
            emotion=Emotion(
                joy=0.0,
                anger=0.0,
                sadness=0.0,
                fun=0.0,
            ),
        )

# --------------------
# health check (UptimeRobot 用)
# --------------------
@app.api_route("/ping", methods=["GET", "HEAD"])
def ping(request: Request):
    return {"status": "ok"}
