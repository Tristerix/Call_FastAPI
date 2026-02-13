from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ResponseModel(BaseModel):
    message: str
    score: int

@app.get("/review")
def review():
    return ResponseModel(
        message="Hello Unity",
        score=100
    )
