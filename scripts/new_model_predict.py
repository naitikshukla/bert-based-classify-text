import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from scripts.config import params
import logging
from typing import Any

from scripts.new_data_load import load_data, TextDataset, clean_text
 
logging.basicConfig(level=logging.INFO)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelPredictor:
    def __init__(self):
        # Load model and tokenizer
        model_path = "models/fold3_model.pt"
        self.model = RobertaForSequenceClassification.from_pretrained(params['PRETRAINED_MODEL_NAME'])
        self.model.load_state_dict(torch.load(model_path, map_location=device)) #load model weights new trained
        self.model.to(device)
        self.tokenizer = RobertaTokenizer.from_pretrained(params['PRETRAINED_MODEL_NAME'])

    def predict_probability(self, text: str) -> Any:
        text = clean_text(text)
        encoded_input = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=params['MAX_LENGTH'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded_input['input_ids'].to(device)
        attention_mask = encoded_input['attention_mask'].to(device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        return probabilities[0].tolist()
    
    @staticmethod
    def convert_to_readable_output(prob: Any) -> dict:
        return {'noHate': prob[0], 'hate': prob[1]}
    