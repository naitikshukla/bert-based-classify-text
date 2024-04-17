from torch.utils.data import Dataset
import pandas as pd
import os
import logging
from scripts.utils.train_utils import batch_encode

# Define the maximum number of words to tokenize (DistilBERT can tokenize up to 512)
MAX_LENGTH = 256

# Set parameters:
params = {'MAX_LENGTH': 128,
          'EPOCHS': 6,
          'LEARNING_RATE': 5e-5,
          'FT_EPOCHS': 2,
          'OPTIMIZER': 'adam',
          'FL_GAMMA': 2.0,
          'FL_ALPHA': 0.2,
          'BATCH_SIZE': 64,
        #   'NUM_STEPS': len(X_train.index) // 64,
          'DISTILBERT_DROPOUT': 0.2,
          'DISTILBERT_ATT_DROPOUT': 0.2,
          'LAYER_DROPOUT': 0.2,
          'KERNEL_INITIALIZER': 'GlorotNormal',
          'BIAS_INITIALIZER': 'zeros',
          'POS_PROBA_THRESHOLD': 0.5,          
          'ADDED_LAYERS': 'Dense 256, Dense 32, Dropout 0.2',
          'LR_SCHEDULE': '5e-5 for 6 epochs, Fine-tune w/ adam for 2 epochs @2e-5',
          'FREEZING': 'All DistilBERT layers frozen for 6 epochs, then unfrozen for 2',
          'CALLBACKS': '[early_stopping w/ patience=0]',
          'RANDOM_STATE':42,
          'TRAIN_RATIO': 0.7,
            'VALID_RATIO': 0.15
          }

class CustomDataset(Dataset):
    def __init__(self, data_dir, annotations_file, tokenizer):
        self.data_dir = data_dir
        self.annotations_file = annotations_file
        self.tokenizer = tokenizer
        self.data = self.load_data()
        self.X_train, self.X_valid, self.X_test, self.y_train, self.y_valid, self.y_test = self.split_data(params['TRAIN_RATIO'], params['VALID_RATIO'])

    def load_data(self) -> pd.DataFrame:
        annotations = pd.read_csv(self.annotations_file)
        annotations.set_index('file_id', inplace=True)
        data = {}
        for file_name in os.listdir(self.data_dir):
            file_id = os.path.splitext(file_name)[0]
            file_path = os.path.join(self.data_dir, file_name)
            if file_id in annotations.index:
                label = annotations.loc[file_id, 'label']
                label = 1 if label == 'hate' else 0
                try:
                    with open(file_path, 'r') as file:
                        text = file.read()
                except Exception as e:
                    logging.error(f"Error reading file {file_path}: {e}")
                    continue
                #create a dictionary with the text and label and then create pandas df
                data[file_id] = {'text': text, 'label': label}
        # set variable NUM_STEPS in params after file loaded
        params['NUM_STEPS'] = len(data) // params['BATCH_SIZE']

        return pd.DataFrame.from_dict(data, orient='index')
    
    def split_data(self, train_ratio, valid_ratio) -> pd.DataFrame:
        # Split the data into training, validation, and test set
        X = self.data['text']
        y = self.data['label']
        train_size = int(train_ratio * len(X))
        valid_size = int(valid_ratio * len(X))
        X_train, X_valid, X_test = X[:train_size], X[train_size:train_size+valid_size], X[train_size+valid_size:]
        y_train, y_valid, y_test = y[:train_size], y[train_size:train_size+valid_size], y[train_size+valid_size:]
        return X_train, X_valid, X_test, y_train, y_valid, y_test
    

    def tokenize_data(self, text: pd.Series):
        # Instantiate DistilBERT tokenizer...we use the Fast version to optimize runtime
        tokenizer = self.tokenizer

        # Encode X_train
        return batch_encode(tokenizer, text.tolist())

        # # Encode X_valid
        # X_valid_ids, X_valid_attention = batch_encode(tokenizer, self.X_valid.tolist())

        # # Encode X_test
        # X_test_ids, X_test_attention = batch_encode(tokenizer, self.X_test.tolist())
