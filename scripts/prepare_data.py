from torch.utils.data import Dataset
import pandas as pd
import os
import logging

from scripts.utils.train_utils import batch_encode
from scripts.config import params

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

if __name__ == "__main__":
    from transformers import DistilBertTokenizerFast

    tokenizer_name = 'distilbert-base-uncased'
    data_dir = 'data'
    model_dir = './models'
    annotations_file_path = os.path.join(data_dir, 'annotations_metadata.csv')
    training_data_dir = os.path.join(data_dir, 'sampled_train')
    
    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_name)
    
    app = CustomDataset(data_dir=training_data_dir, annotations_file=annotations_file_path, tokenizer=tokenizer)
    print("Loaded data successfully!")
    print(app.data)

    app.tokenize_data(tokenizer,pd.Series(app.X_train))
    app.X_valid.tolist()

    # convert O dtype to pd.series
    # app.X_train = pd.Series(app.X_train)
    # temp = app.X_train.tolist()

    # len(temp)

    # batch_encode(tokenizer, temp)

    # texts = temp
