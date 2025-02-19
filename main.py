from fastapi import FastAPI, HTTPException
from http.client import HTTPConnection
from hf_pipeline import get_sentiments_with_pipeline

app = FastAPI()

@app.get('/')
async def root():
    return {'message': 'Hello World!'}

@app.get('/predict_sentiment/')
async def predict_sentiment(text: str):
    if not text:
        raise HTTPException(status_code=400, detail='No text provided')
    model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"

    result = get_sentiments_with_pipeline(
        model_name='nlp_models/distilbert-base',
        tokenizer_name='nlp_models/distilbert-base',
        string_arr=[text]
    )

    return result[0]