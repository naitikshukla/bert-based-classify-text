import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split

from new_data_load import load_data
from new_model_train import train_and_evaluate

# Load data
df = load_data()
# train_df = pd.read_csv('train.csv')
# test_df = pd.read_csv('test.csv')

# Split data into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Load tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Train and evaluate model using StratifiedKFold
train_and_evaluate(model, train_df, val_df)