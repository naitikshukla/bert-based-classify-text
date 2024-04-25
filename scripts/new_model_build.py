import torch
from transformers import DistilBertModel, DistilBertConfig
from torch import nn

class DistilBERTClass(nn.Module):
    def __init__(self,num_classes=2):
        super(DistilBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

def loss_fn(outputs, targets):
    return nn.CrossEntropyLoss()(outputs, targets)


if __name__ == "__main__":
    model = DistilBERTClass()
    model.to(device)