from fastapi import FastAPI
from pydantic import BaseModel

from ml.model import load_model

model = None
app = FastAPI()

# формат предсказания модели
class SentimentResponse(BaseModel):
    text: str # тело запроса
    sentiment_label: str # класс предсказания
    sentiment_score: float # значение предсказания



@app.get("/")
def index():
    return {"text": "Sentiment Analysis"}


@app.on_event("startup")
def startup_event():
    global model
    model = load_model()


# GET-запрос для получения предсказания по заданному тексту
@app.get("/predict")
def predict_sentiment(text: str):
    sentiment = model(text)

    response = SentimentResponse(
        text=text,
        sentiment_label=sentiment.label,
        sentiment_score=sentiment.score,
    )

    return response
