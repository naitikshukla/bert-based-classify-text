import tensorflow as tf
import os
import numpy as np
import pandas as pd
import scikitplot as skplt
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from transformers import DistilBertTokenizerFast
import matplotlib.pyplot as plt

from scripts.model_build import ModelBuilder
from scripts.prepare_data import CustomDataset
from scripts.config import params

class ModelTrainer:
    def __init__(self):
        self.data_dir = 'data'
        self.model_dir = './models'
        self.annotations_file_path = os.path.join(self.data_dir, 'annotations_metadata.csv')
        self.training_data_dir = os.path.join(self.data_dir, 'sampled_train')
        self.tokenizer_name = 'distilbert-base-uncased'
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.tokenizer_name)
        self.data = None
        self.X_train_ids = None
        self.X_train_attention = None
        self.X_valid_ids = None
        self.X_valid_attention = None
        self.y_train = None
        self.y_valid = None
        self.model_builder = None
        self.model = None

    def load_data(self):
        self.data = CustomDataset(data_dir=self.training_data_dir, annotations_file=self.annotations_file_path, tokenizer=self.tokenizer)
        self.X_train_ids, self.X_train_attention = self.data.tokenize_data(self.data.X_train)
        self.X_valid_ids, self.X_valid_attention = self.data.tokenize_data(self.data.X_valid)
        self.y_train = self.data.y_train
        self.y_valid = self.data.y_valid
        params['NUM_STEPS'] = len(self.data.X_train) // params['BATCH_SIZE']

    def build_model(self):
        # self.model_builder = ModelBuilder(freeze=True)
        # self.model = self.model_builder.model

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

    def save_model(self):
        tf.saved_model.save(self.model, os.path.join(self.model_dir, 'hate_speech_detection_model'))
        print(f"Model saved successfully at {os.path.join(self.model_dir, 'hate_speech_detection_model')}")

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
        plt.show()

def train_and_save_model_end2end(plot=True):
    trainer = ModelTrainer()
    trainer.load_data() # Load data
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

    
