from scripts.config import params

import tensorflow as tf
from scripts.utils.train_utils import focal_loss
from transformers import TFDistilBertModel, DistilBertConfig


class ModelBuilder:
    def __init__(self,freeze=True ,params=params):
        # Load DistilBERT model
        config = DistilBertConfig(dropout=params['DISTILBERT_DROPOUT'], 
                                  attention_dropout=params['DISTILBERT_ATT_DROPOUT'], 
                                  output_hidden_states=True)
        self.distilBERT = TFDistilBertModel.from_pretrained(params['PRETRAINED_MODEL_NAME'], config=config)
        
        if freeze:
            # Freeze DistilBERT layers to preserve pre-trained weights 
            for layer in self.distilBERT.layers:
                layer.trainable = False
        else:
            for layer in self.distilBERT.layers:
                layer.trainable = True
        self.model = self.build_model(self.distilBERT)

    def build_model(self,transformer, max_length=params['MAX_LENGTH'])->tf.keras.Model:
        # Define weight initializer with a random seed to ensure reproducibility
        weight_initializer = tf.keras.initializers.GlorotNormal(seed=params['RANDOM_STATE']) 
        
        # Define input layers
        input_ids_layer = tf.keras.layers.Input(shape=(max_length,), 
                                                name='input_ids', 
                                                dtype='int32')
        input_attention_layer = tf.keras.layers.Input(shape=(max_length,), 
                                                      name='input_attention', 
                                                      dtype='int32')
        
        # DistilBERT outputs a tuple where the first element at index 0
        # represents the hidden-state at the output of the model's last layer.
        # It is a tf.Tensor of shape (batch_size, sequence_length, hidden_size=768).
        last_hidden_state = self.transformer([input_ids_layer, input_attention_layer])[0]
        
        # We only care about DistilBERT's output for the [CLS] token, which is located
        # at index 0.  Splicing out the [CLS] tokens gives us 2D data.
        cls_token = last_hidden_state[:, 0, :]
        
        D1 = tf.keras.layers.Dropout(params['LAYER_DROPOUT'],
                                     seed=params['RANDOM_STATE']
                                    )(cls_token)
        
        X = tf.keras.layers.Dense(256,
                                  activation='relu',
                                  kernel_initializer=weight_initializer,
                                  bias_initializer='zeros'
                                  )(D1)
        
        D2 = tf.keras.layers.Dropout(params['LAYER_DROPOUT'],
                                     seed=params['RANDOM_STATE']
                                    )(X)
        
        X = tf.keras.layers.Dense(32,
                                  activation='relu',
                                  kernel_initializer=weight_initializer,
                                  bias_initializer='zeros'
                                  )(D2)
        
        D3 = tf.keras.layers.Dropout(params['LAYER_DROPOUT'],
                                     seed=params['RANDOM_STATE']
                                    )(X)
        
        # Define a single node that makes up the output layer (for binary classification)
        output = tf.keras.layers.Dense(1, 
                                       activation='sigmoid',
                                       kernel_initializer=weight_initializer,  # CONSIDER USING CONSTRAINT
                                       bias_initializer='zeros'
                                       )(D3)
        
        # Define the model
        model = tf.keras.Model([input_ids_layer, input_attention_layer], output)
        
        # Compile the model
        model.compile(tf.keras.optimizers.Adam(lr=params['LEARNING_RATE']), 
                      loss=focal_loss(),
                      metrics=['accuracy'])
        
        return model
