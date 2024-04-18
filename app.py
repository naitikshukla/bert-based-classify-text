from fastapi import FastAPI
from pydantic import BaseModel
from scripts.new_model_predict import ModelPredictor

app = FastAPI()

class TextInput(BaseModel):
    text: str

# Initialize the predictor for the model
predictor  = ModelPredictor()

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Hate Speech Detection API"}

@app.post("/predict")
async def predict(input: TextInput):
    probabilities = predictor.predict_probability(input.text)
    readable_output = predictor.convert_to_readable_output(probabilities)
    return readable_output


# run using docker command
# docker run -d -p 8000:8000 hate-speech-detection-api:latest