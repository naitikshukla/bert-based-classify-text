from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve

import torch
from torch.utils.data import DataLoader
# from transformers import AdamW
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# import tqdm
import logging
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

from scripts.config import params
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, ConstantLR

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train_model(model, train_loader, val_loader, epochs, learning_rate):
    # create optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    #For Scheduler
    total_steps = params['EPOCHS'] * len(train_loader) # Total number of training steps
    warmup_steps = int(total_steps * 0.1) # 10% of the total as the warmup
    # Scheduler for the warmup phase
    scheduler_warmup = ConstantLR(optimizer, factor=1.0, total_iters=warmup_steps)
    # Scheduler for the cosine annealing phase
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    # Combine both schedulers
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_steps])

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        # progress = tqdm(total=len(train_loader), desc="Training")

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels) ##Forward pass
            loss = outputs.loss
            logits = outputs.logits
            train_loss += loss.item()
            train_acc += accuracy_score(labels.cpu().numpy(), torch.argmax(logits, dim=1).cpu().numpy())
            loss.backward()
            optimizer.step()
            scheduler.step()

        # do validation evaluation
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                val_loss += loss.item()
                val_acc += accuracy_score(labels.cpu().numpy(), torch.argmax(logits, dim=1).cpu().numpy())
            # progress.update(1) # update progress bar
        
        # progress.close()
        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)
        val_loss /= len(val_loader)
        val_acc /= len(val_loader.dataset)
        # print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        logging.info(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    return model,train_loss,train_acc,val_loss,val_acc


def train_and_evaluate(model, dataset, epochs=params['EPOCHS'],learning_rate= params['LEARNING_RATE']):    
    # Create StratifiedKFold object
    skf = StratifiedKFold(n_splits=params['n_splits'], shuffle=True, random_state=params['RANDOM_STATE'])
    # dataset = dataloader.dataset
    labels = [sample['label'] for sample in dataset]
    # Initialize metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    val_recall_list = []
    val_precision_list = []
    val_f1_list = []

    roc_auc_list = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset)), labels), 1):
        print(f"Fold: {fold}")
        
        # Subset the dataloader
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        train_dataloader = DataLoader(dataset, batch_size=params['BATCH_SIZE'], sampler=train_subsampler)
        val_dataloader = DataLoader(dataset, batch_size=params['BATCH_SIZE'], sampler=val_subsampler)

        fold_model,train_loss,train_acc,val_loss,val_acc = train_model(model, train_dataloader, val_dataloader, epochs, learning_rate)
        
        # Evaluate on validation set
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                outputs = fold_model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # val_accuracy = accuracy_score(val_labels, val_preds)
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='binary')
        
        #get confusion matrix
        cm = confusion_matrix(val_labels, val_preds)
        logging.info(f"Confusion Matrix: {cm}")

        #get ROC-AUC score
        roc_auc = roc_auc_score(val_labels, val_preds)
        logging.info(f"ROC-AUC Score: {roc_auc}")

        #plot roc curve and save
        fpr, tpr, _ = roc_curve(val_labels, val_preds)
        plt.figure(figsize=(10, 5))
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(f"{params['plot_dir']}/roc_curve_fold{fold}.png")
        # plt.show()
        # params['plot_dir'] = 'material4gh'
        # Save metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        #save precision, recall and f1 for validation
        val_precision_list.append(val_precision)
        val_recall_list.append(val_recall)
        val_f1_list.append(val_f1)
        roc_auc_list.append(roc_auc)
        
        # Save model for this fold
        torch.save(fold_model.state_dict(), f"{params['model_dir']}/fold{fold}_model.pt")
        
    # Compute average metrics across folds
    avg_train_loss = sum(train_losses) / len(train_losses)
    avg_train_accuracy = sum(train_accuracies) / len(train_accuracies)
    avg_val_loss = sum(val_losses) / len(val_losses)
    avg_val_accuracy = sum(val_accuracies) / len(val_accuracies)
    avg_val_precision = sum(val_precision_list) / len(val_precision_list)
    avg_val_recall = sum(val_recall_list) / len(val_recall_list)
    avg_val_f1 = sum(val_f1_list) / len(val_f1_list)
    
    logging.info(f"Average Metrics Across Folds:")
    logging.info(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}")
    logging.info(f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.4f}, Val Precision: {avg_val_precision:.4f}, Val Recall: {avg_val_recall:.4f}, Val F1: {avg_val_f1:.4f}")
    
    #identify best model parameters and get hyperparameters to train model on full dataet
    best_fold = val_f1_list.index(max(val_f1_list))
    logging.info(f"Best Fold: {best_fold}")
    best_model = f"{params['model_dir']}/fold{best_fold}_model.pt"
    logging.info(f"Best Model: {best_model}")
    
    #plot AUc-ROC curve and save AUC
    

    # Return metrics
    return train_losses, train_accuracies, val_losses, val_accuracies, val_precision_list, val_recall_list, val_f1_list, roc_auc_list


def plot_save_metrics(train_losses, train_accuracies, val_losses, val_accuracies, val_precision, val_recall, val_f1, roc_auc_list):
    # Plot metrics of train and validation accuracy in single plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Fold')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()

    plt.savefig(f"{params['plot_dir']}/loss_plot.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.legend()

    plt.savefig(f"{params['plot_dir']}/accuracy_plot.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(val_precision, label='Val Precision')
    plt.plot(val_recall, label='Val Recall')
    plt.plot(val_f1, label='Val F1')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title('Validation Precision, Recall, F1')
    plt.legend()
    plt.savefig(f"{params['plot_dir']}/precision_recall_f1_plot.png")
    plt.show()


    #plot roc_auc over folds
    plt.figure(figsize=(10, 5))
    plt.plot(roc_auc_list, label='ROC-AUC')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title('ROC-AUC Score')
    plt.legend()
    plt.savefig(f"{params['plot_dir']}/roc_auc_plot.png")
    plt.show()


if __name__ == "__main__":
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    from scripts.new_data_load import TextDataset, load_data

    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    df = load_data()
    dataset = TextDataset(df,tokenizer)

    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Train and evaluate model using StratifiedKFold
    train_losses, train_accuracies, val_losses, val_accuracies, val_precision, val_recall, val_f1,roc_auc_list = train_and_evaluate(model, dataset)

    plot_save_metrics(train_losses, train_accuracies, val_losses, val_accuracies, val_precision, val_recall, val_f1,roc_auc_list)
                      
