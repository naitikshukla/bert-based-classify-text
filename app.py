from fastapi import FastAPI
from pydantic import BaseModel
from scripts.model_predict import ModelPredictor, Config, RobertaTokenizer, RobertaForSequenceClassification, torch, device

app = FastAPI()

class TextInput(BaseModel):
    text: str

class ModelPredictorAPI(ModelPredictor):
    def __init__(self):
        model = RobertaForSequenceClassification.from_pretrained(Config.pretrained_model_name)
        model.load_state_dict(torch.load(Config.model_path, map_location=device))
        tokenizer = RobertaTokenizer.from_pretrained(Config.pretrained_model_name)
        model.to(device)
        super().__init__(model, tokenizer)

predictor = ModelPredictorAPI()

@app.post("/predict")
async def predict(input: TextInput):
    probabilities = predictor.predict_probability(input.text)
    readable_output = predictor.convert_to_readable_output(probabilities)
    return readable_output