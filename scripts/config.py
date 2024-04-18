import os

class config:
    def __init__(self):
        self.data_dir = 'data'
        self.model_dir = './model'
        self.annotations_file_path = os.path.join(self.data_dir, 'annotations_metadata.csv')
        self.training_data_dir = os.path.join(self.data_dir, 'sampled_train')
        self.pretrained_model_name = 'distilbert-base-uncased'
        self.learning_rate = 3e-6 
        self.weight_decay = 0.01
        self.batch_size = 64
        self.num_epochs = 10
        self.num_folds = 10
        self.patience = 3
        self.MAX_LENGTH = 256
        self.EPOCHS = 10
        self.LEARNING_RATE = 5e-5


params = {'MAX_LENGTH': 128,
          'EPOCHS': 6,
          'LEARNING_RATE': 5e-5,
          'FT_EPOCHS': 2,
          'OPTIMIZER': 'adam',
          'FL_GAMMA': 2.0,
          'FL_ALPHA': 0.2,
          'BATCH_SIZE': 64,
        #'NUM_STEPS': len(X_train.index) // 64,
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
            'VALID_RATIO': 0.15,
            'PRETRAINED_MODEL_NAME':'distilbert-base-uncased',
            'LOCAL_MODEL_NAME':'hate_speech_detection_model',
          }
    