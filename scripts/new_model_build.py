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

# Creating the loss function and optimizer
def create_loss_fn():
    return torch.nn.CrossEntropyLoss()

def create_optimizer(model, learning_rate):
    return torch.optim.Adam(params =  model.parameters(), lr=learning_rate)


if __name__ == "__main__":

    learning_rate = 3e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DistilBERTClass()
    model.to(device)

    loss = create_loss_fn()
    optimizer = create_optimizer(model, learning_rate)

    from transformers import DistilBertTokenizerFast
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    from scripts.new_data_load import load_data
    df_train = load_data()
    df_test = load_data(test=True)
    print(df_train.head())
    print(df_test.head())

    from scripts.new_data_load import TextDataset
    train_dataloader = TextDataset(df_train,tokenizer)
    test_dataloader = TextDataset(df_test,tokenizer)

    train_params = {'batch_size': 32,
                'shuffle': True,
                'num_workers': 0
                }

    test_params = {'batch_size': 32,
                    'shuffle': True,
                    'num_workers': 0
                    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

