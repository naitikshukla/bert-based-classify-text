import tensorflow as tf
import os
import numpy as np
import pandas as pd
import scikitplot as skplt
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
# from transformers import RobertaTokenizer, RobertaForSequenceClassification
import random
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, ConstantLR
from torch.optim import AdamW

import matplotlib.pyplot as plt
from typing import Tuple, Any
from torch.utils.data import Dataset, DataLoader
import torch
import logging

from scripts.model_build import ModelBuilder
from scripts.prepare_data import CustomDataset
from scripts.config import params

logging.basicConfig(level=logging.INFO)

SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Device set to {device}")

class ModelTrainer:
    def __init__(self):
        self.data_dir = 'data'
        self.model_dir = './models'
        self.annotations_file_path = os.path.join(self.data_dir, 'annotations_metadata.csv')
        self.training_data_dir = os.path.join(self.data_dir, 'sampled_train')
        self.tokenizer_name = 'distilbert-base-uncased'
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.tokenizer_name)
        self.dataloader = None
        self.X_train_ids = None
        self.X_train_attention = None
        self.X_valid_ids = None
        self.X_valid_attention = None
        self.y_train = None
        self.y_valid = None
        self.model = None
        self.base_model_flag = True
        self.folds = 4

    def prepare_data(self) -> Any:
        dataset = CustomDataset(data_dir=self.training_data_dir, annotations_file=self.annotations_file_path, tokenizer=self.tokenizer)
        self.dataloader = DataLoader(dataset, batch_size=params['BATCH_SIZE'], shuffle=True)

        # self.X_train_ids, self.X_train_attention = self.data.tokenize_data(self.data.X_train)
        # self.X_valid_ids, self.X_valid_attention = self.data.tokenize_data(self.data.X_valid)
        # self.y_train = self.data.y_train
        # self.y_valid = self.data.y_valid
        params['NUM_STEPS'] = len(dataset) // params['BATCH_SIZE']

    def build_model(self):
        if self.base_model_flag:
            self.model = DistilBertForSequenceClassification.from_pretrained(params['PRETRAINED_MODEL_NAME']).to(device)
            logging.info(f"Base model loaded successfully: {params['PRETRAINED_MODEL_NAME']}")
        else:
            model_builder = ModelBuilder(freeze=True)
            self.model = model_builder.model

    def train_model(self):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          mode='min',
                                                          min_delta=0,
                                                          patience=0,
                                                          restore_best_weights=True)

        train_history1 = self.model.fit(
            x=[self.X_train_ids, self.X_train_attention],
            y=self.y_train.to_numpy(),
            epochs=params['EPOCHS'],
            batch_size=params['BATCH_SIZE'],
            steps_per_epoch=params['NUM_STEPS'],
            validation_data=([self.X_valid_ids, self.X_valid_attention], self.y_valid.to_numpy()),
            callbacks=[early_stopping],
            verbose=2)

        return train_history1
    
    def train_model_with_folds(self):
        skf = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=SEED)
        dataset = self.dataloader.dataset
        labels = [sample['labels'] for sample in dataset.data]
        best_epochs = []

        # Start training for each fold
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset)), labels), 1):
            logging.info(f"\nFold {fold}/{self.folds}:\n")
            # Subset the dataloader
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

            train_dataloader = DataLoader(dataset, batch_size=params['BATCH_SIZE'], sampler=train_subsampler)
            val_dataloader = DataLoader(dataset, batch_size=params['BATCH_SIZE'], sampler=val_subsampler)

            # Train the model for the current fold        
            fold_model, best_fold_epoch = self.train_model_with_validation(params['EPOCHS'], self.model, train_dataloader, val_dataloader, device)
            logging.info(f"Best epoch for fold {fold}: {best_fold_epoch}")

            # Save the best model for the current fold
            save_model(fold_model, f'fold_{fold}.pt') 
            best_epochs.append(best_fold_epoch)

        return round(np.mean(best_epochs))

    def train_model_with_validation(self,num_epochs: int, model: Any, train_dataloader: Any, val_dataloader: Any, device: Any) -> Tuple[Any, int]:
        best_val_loss = float('inf')
        best_print_str = ""
        best_epoch = 0
        no_improve_epoch = 0
        
        optimizer, scheduler = self.create_optimizer_and_scheduler(model, len(train_dataloader))

        for epoch in range(num_epochs):
            # Training and validation steps
            train_loss, train_precision, train_recall, train_f1, train_auc = train_epoch(model, train_dataloader, optimizer, scheduler, device)
            val_loss, val_precision, val_recall, val_f1, val_auc = validate_epoch(model, val_dataloader, device)
            
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}, Validation Loss: {val_loss}")
            logging.info(f"Training Precision: {train_precision}, Training Recall: {train_recall}, Training F1: {train_f1}, Training AUC: {train_auc}")
            logging.info(f"Validation Precision: {val_precision}, Validation Recall: {val_recall}, Validation F1: {val_f1}, Validation AUC: {val_auc}")
            
            # Check for improvement in validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_print_str = f"""Epoch {epoch+1} | Train Loss: {train_loss}, Val Loss: {val_loss} \n
                TRAIN: Precision: {train_precision}, Recall: {train_recall}, F1: {train_f1}, AUC: {train_auc}" \n
                VALIDATION: Precision: {val_precision}, Recall: {val_recall}, F1: {val_f1}, AUC: {val_auc} \n"""
                no_improve_epoch = 0  # Reset counter
            else:
                no_improve_epoch += 1  # Increment counter if no improvement in validation loss

            # Early stopping check
            if no_improve_epoch >= Config.patience:
                logging.info("Early stopping triggered.")
                break
        
        logging.info(best_print_str)
        return model.state_dict(), best_epoch

    def create_optimizer_and_scheduler(self,model: Any, dataloader_len: int) -> Tuple[Any, Any]:
        optimizer = AdamW(model.parameters(), lr=params['LEARNING_RATE'], weight_decay=params['WEIGHT_DECAY'])

        total_steps = params['NUM_EPOCHS'] * dataloader_len # Total number of training steps
        warmup_steps = int(total_steps * 0.1) # 10% of the total as the warmup

        # Scheduler for the warmup phase
        scheduler_warmup = ConstantLR(optimizer, factor=1.0, total_iters=warmup_steps)

        # Scheduler for the cosine annealing phase
        scheduler_cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)

        # Combine both schedulers
        scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_steps])
        return optimizer, scheduler
    
    def save_model(self):
        tf.saved_model.save(self.model, os.path.join(self.model_dir, params['LOCAL_MODEL_NAME']))
        # self.model.save_pretrained(os.path.join(self.model_dir, params['LOCAL_MODEL_NAME']))
        print(f"Model saved successfully at {os.path.join(self.model_dir, params['LOCAL_MODEL_NAME'])}")

    def load_model(self):
        self.model = tf.saved_model.load(os.path.join(self.model_dir, params['LOCAL_MODEL_NAME']))


    def evaluate_model(self):
        X_test_ids, X_test_attention = self.data.tokenize_data(self.data.X_test)
        y_test = self.data.y_test

        y_pred = self.model.predict([X_test_ids, X_test_attention])
        y_pred_thresh = np.where(y_pred >= params['POS_PROBA_THRESHOLD'], 1, 0)

        accuracy = accuracy_score(y_test, y_pred_thresh)
        auc_roc = roc_auc_score(y_test, y_pred)

        return accuracy, auc_roc

    def plot_loss(self, train_history):
        history_df = pd.DataFrame(train_history.history)
        history_df.loc[:, ['loss', 'val_loss']].plot()
        plt.title(label='Training + Validation Loss Over Time', fontsize=17, pad=19)
        plt.xlabel('Epoch', labelpad=14, fontsize=14)
        plt.ylabel('Focal Loss', labelpad=16, fontsize=14)
        print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))
        #save plot image in material4gh folder
        plt.savefig('material4gh/loss_plot.png')
        plt.show()


    def plot_confusion_matrix(self):
        X_test_ids, X_test_attention = self.data.tokenize_data(self.data.X_test)
        y_test = self.data.y_test

        y_pred = self.model.predict([X_test_ids, X_test_attention])
        y_pred_thresh = np.where(y_pred >= params['POS_PROBA_THRESHOLD'], 1, 0)

        skplt.metrics.plot_confusion_matrix(y_test.to_list(),
                                            y_pred_thresh.tolist(),
                                            figsize=(6, 6),
                                            text_fontsize=14)
        plt.title(label='Test Confusion Matrix', fontsize=20, pad=17)
        plt.xlabel('Predicted Label', labelpad=14)
        plt.ylabel('True Label', labelpad=14)
        plt.savefig('material4gh/confusion.png')
        plt.show()

def train_and_save_model_end2end(plot=True):
    trainer = ModelTrainer()
    trainer.prepare_data() # Load data
    trainer.build_model() # Build model
    train_history = trainer.train_model() # Train model
    trainer.save_model() # Save model
    accuracy, auc_roc = trainer.evaluate_model() # Evaluate model
    if plot:
        trainer.plot_loss(train_history)
        trainer.plot_confusion_matrix()
    print(f"Accuracy: {accuracy}")
    print(f"AUC-ROC: {auc_roc}")


if __name__ == '__main__':
    train_and_save_model_end2end(plot=True)


    model_dir = './models'
    local_model_path = os.path.join(model_dir, 'hate_speech_detection_model')

    # Loading the model
    loaded_model = tf.saved_model.load(local_model_path)

    X_test_ids, X_test_attention = trainer.data.tokenize_data(trainer.data.X_test)
    y_test = trainer.data.y_test

    y_pred = loaded_model.predict([X_test_ids, X_test_attention])
    y_pred_thresh = np.where(y_pred >= params['POS_PROBA_THRESHOLD'], 1, 0)

    accuracy = accuracy_score(y_test, y_pred_thresh)
    auc_roc = roc_auc_score(y_test, y_pred)


    # Convert your input data to tf.Tensor
    X_test_ids_tensor = tf.convert_to_tensor(X_test_ids)
    X_test_attention_tensor = tf.convert_to_tensor(X_test_attention)


    predictions = loaded_model([X_test_ids_tensor, X_test_attention_tensor])

    trainer.model.save_pretrained('./models/hate_speech_detection_model')


    # Use the loaded model for prediction
    predictions = loaded_model.signatures['serving_default']([X_test_ids_tensor, X_test_attention_tensor])


    import pickle
    from scripts.utils.train_utils import batch_encode


    # Assuming `tokenizer` is your tokenizer
    with open('models/tokenizer.pkl', 'wb') as f:
        pickle.dump(trainer.tokenizer, f)

    # Load the saved model
    loaded_model = tf.saved_model.load('models/hate_speech_detection_model')

    # Load the tokenizer
    with open('models/tokenizer.pkl', 'rb') as f:
        tokenizer1 = pickle.load(f)

    # Tokenize your input data using the loaded tokenizer
    # Assuming `input_data` is your input for prediction
    input_ids, attention_mask = batch_encode(tokenizer1, trainer.data.X_test.tolist())




    # Convert your input data to tf.Tensor
    input_ids_tensor = tf.convert_to_tensor(input_ids)
    attention_mask_tensor = tf.convert_to_tensor(attention_mask)

    # Use the loaded model for prediction
    predictions = loaded_model.signatures['serving_default'](input_ids_tensor,attention_mask_tensor)
        input_ids=input_ids_tensor, 
        attention_mask=attention_mask_tensor
    )

    trainer.data.X_train.tolist()