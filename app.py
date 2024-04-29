from fastapi import FastAPI
from pydantic import BaseModel
from scripts.saved_custom_model_predict import TextClassifier

model_file = './models/pytorch_distilbert_custom.bin'

app = FastAPI()

class TextInput(BaseModel):
    text: str

# Initialize the predictor for the model
predictor  = TextClassifier(model_file)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Hate Speech Detection API"}

@app.post("/predict")
async def predict(input: TextInput):
    output = predictor.predict(input.text)
    # readable_output = predictor.convert_to_readable_output(probabilities)
    return output


# run using docker command
# docker run -d -p 8000:8000 hate-speech-detection-api:latest