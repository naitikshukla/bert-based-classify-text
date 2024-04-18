from scripts.config import params
import tensorflow as tf
import tensorflow.keras.backend as K

def batch_encode(tokenizer, texts, batch_size=256, max_length=params['MAX_LENGTH'],type='pt'):
    """""""""
    A function that encodes a batch of texts and returns the texts'
    corresponding encodings and attention masks that are ready to be fed 
    into a pre-trained transformer model.
    
    Input:
        - tokenizer:   Tokenizer object from the PreTrainedTokenizer Class
        - texts:       List of strings where each string represents a text
        - batch_size:  Integer controlling number of texts in a batch
        - max_length:  Integer controlling max number of words to tokenize in a given text
        - type:        String specifying the type of tensor to be returned ('tf' for tensorflow) or ('pt' for PyTorch)
    Output:
        - input_ids:       sequence of texts encoded as a tf.Tensor object
        - attention_mask:  the texts' attention mask encoded as a tf.Tensor object
    """""""""
    
    input_ids = []
    attention_mask = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]  #0,256,512,768,1024,1280
        inputs = tokenizer.batch_encode_plus(batch,
                                             max_length=max_length,
                                             padding='max_length', #implements dynamic padding
                                             truncation=True,
                                             return_attention_mask=True,
                                             return_token_type_ids=False,
                                             return_tensors=type #'tf' for tensorflow
                                             )
        input_ids.extend(inputs['input_ids'])
        attention_mask.extend(inputs['attention_mask'])
    
    if type == 'pt':
        print("Returnin pytorch tokens")
        return input_ids, attention_mask
    
    return tf.convert_to_tensor(input_ids), tf.convert_to_tensor(attention_mask)



def focal_loss(gamma=params['FL_GAMMA'], alpha=params['FL_ALPHA']):
    """""""""
    Function that computes the focal loss.
    
    Code adapted from https://gist.github.com/mkocabas/62dcd2f14ad21f3b25eac2d39ec2cc95
    """""""""
    
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed