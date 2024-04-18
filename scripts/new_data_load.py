from torch.utils.data import Dataset
import torch
import pandas as pd
import os
import logging
import re
import contractions

from scripts.config import params

class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len = params['MAX_LENGTH']):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.df.iloc[index]['text']
        label = self.df.iloc[index]['label']
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }



### Helper functions to read data from file

def clean_text(text: str) -> str:
    text = contractions.fix(text) # expand contractions like don't to do not
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # remove any URl
    text = re.sub(r'[^\w\s\?\!\.,\'\"…]', ' ', text) # remove special characters
    text = re.sub(r'[^\x00-\x7F]+', '', text) # remove non-ASCII characters
    text = re.sub(r'\s+', ' ', text).strip() # remove extra whitespaces
    return text

def load_data(test=False) -> pd.DataFrame:
    try:
        #load params required for this operation
        if test:
            data_dir = params['testing_data_dir']
        else:
            data_dir = params['training_data_dir']
        annotation_file = params['annotations_file_path']
    except Exception as e:
        logging.error(f"Error reading params file: {e}")
        return pd.DataFrame()
    
    try:
        annotations = pd.read_csv(annotation_file)
        annotations.set_index('file_id', inplace=True)
    except Exception as e:
        logging.error(f"Error reading annotations file: {e}")
        return pd.DataFrame()
    data = {}
    for file_name in os.listdir(data_dir):
        file_id = os.path.splitext(file_name)[0]
        file_path = os.path.join(data_dir, file_name)
        if file_id in annotations.index:
            label = annotations.loc[file_id, 'label']
            label = 1 if label == 'hate' else 0
            try:
                with open(file_path, 'r') as file:
                    text = file.read()
            except Exception as e:
                logging.error(f"Error reading file {file_path}: {e}")
                continue
            #clean raw text
            text = clean_text(text)

            #create a dictionary with the text and label and then create pandas df
            data[file_id] = {'text': text, 'label': label}

    return pd.DataFrame.from_dict(data, orient='index')

if __name__ == '__main__':
    from transformers import DistilBertTokenizerFast

    # load dataframe
    df = load_data()
    print(df.head())

    #load default tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Initialize dataset
    dataloader = TextDataset(df,tokenizer)
    print(dataloader[0])