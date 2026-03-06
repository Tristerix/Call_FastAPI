import os
import re
import json
import logging
import random
import requests

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import Response

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
# Gemini API keys
# --------------------
GEMINI_KEYS = [
    os.getenv(f"GEMINI_API_KEY_{i}") for i in range(1, 11)
]

GEMINI_KEYS = [k for k in GEMINI_KEYS if k]

if not GEMINI_KEYS:
    logger.error("Gemini APIキーが設定されていません")


def get_gemini_url():
    key = random.choice(GEMINI_KEYS)
    return f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={key}"


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

    if not GEMINI_KEYS:
        logger.error("APIキー未設定状態でリクエスト")
        return UnityResponse(
            message="サーバー設定エラー",
            emotion=Emotion(joy=0, anger=0, sadness=0, fun=0),
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

        resp = None

        for i in range(len(GEMINI_KEYS)):

            url = get_gemini_url()

            logger.info(f"Gemini request attempt {i+1}")

            resp = requests.post(
                url,
                json=payload,
                timeout=20
            )

            if resp.status_code == 429:
                logger.warning("Gemini 429 Too Many Requests")
                continue

            break

        if resp is None or resp.status_code == 429:

            logger.error("すべてのAPIキーが429で失敗")

            return UnityResponse(
                message="リクエストが集中しています。しばらく時間をおいてからお試しください。",
                emotion=Emotion(joy=0, anger=0, sadness=0, fun=0),
            )

        resp.raise_for_status()

        gemini_json = resp.json()

        raw_text = gemini_json["candidates"][0]["content"]["parts"][0].get("text", "")

        if not raw_text:
            raise ValueError("Geminiレスポンスが空")

        m = re.search(r"\{.*\}", raw_text, re.DOTALL)

        if not m:
            raise ValueError("JSON抽出失敗")

        parsed = json.loads(m.group())

        emotion = parsed.get("emotion", {})

        return UnityResponse(
            message=str(parsed.get("message", "")),
            emotion=Emotion(
                joy=float(emotion.get("joy", 0)),
                anger=float(emotion.get("anger", 0)),
                sadness=float(emotion.get("sadness", 0)),
                fun=float(emotion.get("fun", 0)),
            ),
        )

    except Exception as e:

        logger.exception("API処理エラー")

        return UnityResponse(
            message="サーバーエラーが発生しました",
            emotion=Emotion(joy=0, anger=0, sadness=0, fun=0),
        )


# --------------------
# Ping（Renderスリープ対策）
# --------------------
@app.head("/ping")
def ping_head():
    return Response(status_code=200)
