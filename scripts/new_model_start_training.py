import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split

from scripts.new_data_load import TextDataset, load_data
from scripts.new_model_train import train_and_evaluate

import logging
logging.basicConfig(level=logging.INFO)
class ModelTrainer:
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.dataset = None
        self.model.to(self.device)

    def load_data(self):
        df = load_data()
        self.dataset = TextDataset(df, self.tokenizer)

    def train_and_evaluate_model(self):
        train_and_evaluate(self.model, self.dataset)

    def start_training(self):
        self.load_data()
        logging.info("Loaded data successfully!")
        self.train_and_evaluate_model()
        logging.info("Training completed!")
        
if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.start_training()