import os

class params:
    data_dir = 'data'
    model_dir = './model'
    annotations_file_path = os.path.join(data_dir, 'annotations_metadata.csv')
    training_data_dir = os.path.join(data_dir, 'sampled_train')
    pretrained_model_name = 'distilroberta-base'
    learning_rate = 3e-6 
    weight_decay = 0.01
    batch_size = 64
    num_epochs = 10
    num_folds = 10
    patience = 3

    MAX_LENGTH = 256
    EPOCHS=10
    LEARNING_RATE=5e-5
    OPTIMIZER


    params = {'MAX_LENGTH': 128,
          'EPOCHS': 6,
          'LEARNING_RATE': 5e-5,
          'FT_EPOCHS': 2,
          'OPTIMIZER': 'adam',
          'FL_GAMMA': 2.0,
          'FL_ALPHA': 0.2,
          'BATCH_SIZE': 64,
          'NUM_STEPS': len(X_train.index) // 64,
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
          'RANDOM_STATE':42
          }
    