import config
from transformers import pipeline
from fastapi import FastAPI
import uvicorn

app = FastAPI(debug=True)

@app.get("/")
def home():
    return {"Project Name" : "English to Hindi Translation"}

@app.get("/predict")
def predict(translation_sentence:str):
    translator = pipeline("text2text-generation", model="Vasanth/"+config.MODEL_OUT_NAME)
    return translator(translation_sentence)

if __name__ == "__main__":
    uvicorn.run(app)

