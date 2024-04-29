import torch
from transformers import DistilBertTokenizerFast
from scripts.new_data_load import clean_text

        

class TextClassifier:
    def __init__(self, model_file, vocab_file=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.map_predict_class = {0: "Not Hate Speech", 1: "Hate Speech"}
        self.model = torch.load(model_file)
        if vocab_file is None:
            self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        else:
            self.tokenizer = DistilBertTokenizerFast(vocab_file)
        self.model.to(self.device)
        # self.tokenizer.to(self.device)

    def predict(self, sentence):
        self.model.eval()
        inputs = self.tokenizer(clean_text(sentence), return_tensors="pt")
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        output = self.model(input_ids, attention_mask)
        _, prediction = torch.max(output, dim=1)
        return self.map_predict_class.get(prediction.item())


if __name__ == "__main__":
    model_file = './models/pytorch_distilbert_custom.bin'
    vocab_file = './models/vocab_distilbert_custom.bin/vocab.txt'
    classifier = TextClassifier(model_file, vocab_file=None)

    sentence = "I love data science"
    prediction = classifier.predict(sentence)
    print(prediction)
