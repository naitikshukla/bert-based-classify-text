from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast
import torch
from scripts.new_model_build import DistilBERTClass, create_loss_fn, create_optimizer
from scripts.new_data_load import load_data, TextDataset
from scripts.config import params
import logging
from sklearn.metrics import confusion_matrix

logging.basicConfig(level=logging.INFO)

class TrainModel:
    def __init__(self):
        self.learning_rate = params['LEARNING_RATE']
        self.TRAIN_BATCH_SIZE = params['TRAIN_BATCH_SIZE']
        self.VALID_BATCH_SIZE = params['VALID_BATCH_SIZE']
        self.EPOCHS = params['EPOCHS']
        self.model = DistilBERTClass()
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_function = create_loss_fn()

    def train_and_evaluate_model(self):
        # Initialize model, tokenizer, loss function, and optimizer
        
        optimizer = create_optimizer(self.model, self.learning_rate)
        self.model.to(self.device)

        # Load data Train and test
        training_set = TextDataset(load_data(), self.tokenizer)

        train_params = {
            'batch_size': self.TRAIN_BATCH_SIZE,
            'shuffle': True,
            'num_workers': 0
        }
        training_loader = DataLoader(training_set, **train_params)

        train_epoch_accuracy = []
        train_epoch_loss = []

        val_epoch_accuracy = []
        val_epoch_loss = []

        # Training loop
        for epoch in range(self.EPOCHS):
            self.model.train()
            tr_loss = 0
            n_correct = 0
            nb_tr_steps = 0
            nb_tr_examples = 0

            for _, data in enumerate(training_loader, 0):
                ids = data['input_ids'].to(self.device, dtype=torch.long)
                mask = data['attention_mask'].to(self.device, dtype=torch.long)
                targets = data['label'].to(self.device, dtype=torch.long)

                optimizer.zero_grad()
                outputs = self.model(ids, mask)
                loss = self.loss_function(outputs, targets)
                loss.backward()
                optimizer.step()

                tr_loss += loss.item()
                big_val, big_idx = torch.max(outputs.data, dim=1)
                n_correct += self.calculate_accu(big_idx, targets)

                nb_tr_steps += 1
                nb_tr_examples += targets.size(0)

                if _ % 500 == 0:
                    loss_step = tr_loss / nb_tr_steps
                    accu_step = (n_correct * 100) / nb_tr_examples

                    logging.info(f"Training Loss per 500 steps: {loss_step}")
                    logging.info(f"Training Accuracy per 500 steps: {accu_step}")
            
            epoch_loss = tr_loss / nb_tr_steps
            epoch_accu = (n_correct * 100) / nb_tr_examples
            train_epoch_accuracy.append(epoch_accu)
            train_epoch_loss.append(epoch_loss)
            logging.info(f"Training Loss Epoch: {epoch_loss}")
            logging.info(f"Training Accuracy for Epoch {epoch}: {epoch_accu}")

            val_accu, val_loss = self.evaluate_model()
            val_epoch_accuracy.append(val_accu)
            val_epoch_loss.append(val_loss)
        return train_epoch_accuracy, train_epoch_loss,val_epoch_accuracy,val_epoch_loss

    def evaluate_model(self):
        self.model.eval()
        n_correct = 0
        tr_loss = 0
        nb_tr_steps = 0
        nb_tr_examples = 0

        testing_set = TextDataset(load_data(test=True), self.tokenizer)
        test_params = {
            'batch_size': self.VALID_BATCH_SIZE,
            'shuffle': True,
            'num_workers': 0
        }
        testing_loader = DataLoader(testing_set, **test_params)

        with torch.no_grad():
            for _, data in enumerate(testing_loader, 0):
                ids = data['input_ids'].to(self.device, dtype=torch.long)
                mask = data['attention_mask'].to(self.device, dtype=torch.long)
                targets = data['label'].to(self.device, dtype=torch.long)

                outputs = self.model(ids, mask).squeeze()
                loss = self.loss_function(outputs, targets)

                tr_loss += loss.item()
                _, big_idx = torch.max(outputs.data, dim=1)
                n_correct += self.calculate_accu(big_idx, targets)
                
                nb_tr_steps += 1
                nb_tr_examples += targets.size(0)

                if _ % 8 == 0:
                    loss_step = tr_loss / nb_tr_steps
                    accu_step = (n_correct * 100) / nb_tr_examples
                    logging.info(f"Validation Loss per 8 steps: {loss_step}")
                    logging.info(f"Validation Accuracy per 8 steps: {accu_step}")


        epoch_loss = tr_loss / nb_tr_steps
        epoch_accu = (n_correct * 100) / nb_tr_examples
        logging.info(f"\tValidation Loss Epoch: {epoch_loss}")
        logging.info(f"\tValidation Accuracy Epoch: {epoch_accu}")

        return epoch_accu, epoch_loss

    def create_confusion_matrix(self):
        self.model.eval()

        testing_set = TextDataset(load_data(test=True), self.tokenizer)
        test_params = {
            'batch_size': self.VALID_BATCH_SIZE,
            'shuffle': True,
            'num_workers': 0
        }
        testing_loader = DataLoader(testing_set, **test_params)
    
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in testing_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(data)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        cm = confusion_matrix(all_labels, all_preds)
        logging.info(f"Confusion Matrix: {cm}")
        return cm
               
    def save_model(self):
        output_model_file = './models/pytorch_distilbert_custom.bin'
        output_vocab_file = './models/vocab_distilbert_custom.bin'

        torch.save(self.model, output_model_file)
        self.tokenizer.save_vocabulary(output_vocab_file)

        print('All files saved')
        print('This tutorial is completed')

    def calculate_accu(self, big_idx, targets):
        n_correct = (big_idx == targets).sum().item()
        return n_correct

if __name__ == "__main__":
    # Create an instance of the API class
    model_train = TrainModel()

    # Call the necessary methods
    train_epoch_accuracy, train_epoch_loss,val_epoch_accuracy,val_epoch_loss = model_train.train_and_evaluate_model()
    model_train.create_confusion_matrix()
    model_train.save_model()
