# NER_keras_tensorflow

Notebook in progress. NER model using Keras on tensorflow

# Background

Recreating a NER model in keras, inspired by the architecture of Guillaume Genthial's NER model in [tensorflow] (https://github.com/guillaumegenthial/sequence_tagging/). 

# Data

CONLL dataset
- downloaded and saved in notebook as train, dev and test.
- Used Glove embeddings
- Vocab words (vocab_words) from data/words.txt composed of 23 unique words.
- Vocab characters (vocab_chars) under data/chars.txt composed of 28 unique characters.
- Vocab tags (vocab_tags) under data/tags.txt composed of 9 unique tags associated with numbers 0-8. See below:
  'B-LOC': 7,
   'B-MISC': 3,
   'B-ORG': 5,
   'B-PER': 1,
   'I-LOC': 8,
   'I-MISC': 4,
   'I-ORG': 6,
   'I-PER': 2,
   'O': 0

# Network Architecture and Hyperparameters

Network takes in word embeddings and character embeddings, which have gone through forward and backward LSTMs, and then concatenates both and puts them through forward and backward LSTMs and then goes through a softmax activation function. The model is compiled with an Adam optimizer and categorical crossentropy for loss.

Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_2 (InputLayer)            (None, None, None)   0                                            
__________________________________________________________________________________________________
lambda_1 (Lambda)               (None, None)         0           input_2[0][0]                    
__________________________________________________________________________________________________
masking_2 (Masking)             (None, None)         0           lambda_1[0][0]                   
__________________________________________________________________________________________________
embedding_2 (Embedding)         (None, None, 100)    2800        masking_2[0][0]                  
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, None, 100)    0           embedding_2[0][0]                
__________________________________________________________________________________________________
lstm_1 (LSTM)                   (None, 100)          80400       dropout_1[0][0]                  
__________________________________________________________________________________________________
lstm_2 (LSTM)                   (None, 100)          80400       dropout_1[0][0]                  
__________________________________________________________________________________________________
input_1 (InputLayer)            (None, None)         0                                            
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 200)          0           lstm_1[0][0]                     
                                                                 lstm_2[0][0]                     
__________________________________________________________________________________________________
masking_1 (Masking)             (None, None)         0           input_1[0][0]                    
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 200)          0           concatenate_1[0][0]              
__________________________________________________________________________________________________
embedding_1 (Embedding)         (None, None, 300)    6900        masking_1[0][0]                  
__________________________________________________________________________________________________
lambda_2 (Lambda)               (None, None, 200)    0           dropout_2[0][0]                  
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, None, 500)    0           embedding_1[0][0]                
                                                                 lambda_2[0][0]                   
__________________________________________________________________________________________________
dropout_3 (Dropout)             (None, None, 500)    0           concatenate_2[0][0]              
__________________________________________________________________________________________________
lstm_3 (LSTM)                   (None, None, 300)    961200      dropout_3[0][0]                  
__________________________________________________________________________________________________
lstm_4 (LSTM)                   (None, None, 300)    721200      lstm_3[0][0]                     
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, None, 600)    0           lstm_3[0][0]                     
                                                                 lstm_4[0][0]                     
__________________________________________________________________________________________________
dropout_4 (Dropout)             (None, None, 600)    0           concatenate_3[0][0]              
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, None, 10)     6010        dropout_4[0][0]                  
__________________________________________________________________________________________________
activation_1 (Activation)       (None, None, 10)     0           dense_1[0][0]                    
==================================================================================================
Total params: 1,858,910
Trainable params: 1,852,010
Non-trainable params: 6,900
__________________________________________________________________________________________________

The best results so far were:

- Training:
  {'precision': 0.8522053133866392}
  {'recall': 0.8436103663985701}
  {'total_correct': 23499.0}
  {'acc': 97.4815957096763, 'f1': 84.78860588952332}

- Validation:
  {'precision': 0.7701129750085587}
  {'recall': 0.7571524739145069} 
  {'total_correct': 5942.0}
  {'acc': 95.72641252287684, 'f1': 76.35777325186693}

- Test:
  {'precision': 0.7235902926481085}
  {'recall': 0.7179532577903682}
  {'total_correct': 5648.0}
  {'acc': 94.57090556692151, 'f1': 72.07607536437965}
  
This was achieved on 100 epochs, with a learning rate of 0.0105, a learning rate decay of 0.0005, a batch size of 10 and the dev data used as validation_data when calling model.fit. These weights were saved as "softmax_test_6_3.hdf5".

# Evaluate

Evaluation was done similar to the Stanford NER evaluation technique, which uses chunking. Currently this process falls under three functions 'extract_data()', 'predict_labels()', and 'compute_metrics()'. This process trims the data to its original, unpadded size and then evaluates where the "chunks" with the named entity recognition are and if the predicted labeled chunks are correct in relation to the labelled data set. F1 and accuracy are used for evaluation, though F1 is used primarily. 

# Experiments up to 6/18/18

All of my experiments from 3/18/18 to 6/18/18 are available in this github under "experiments".

# Next Steps

Areas where there is room to improve:

- Evaluate the architecture
  - Add layers to word embeddings
  - Add more Dense layers and activation functions throughout
  - Add batch normalization?
- Evaluate the embeddings
  - are they structured correctly?
  - try out FastText with or instead of Glove
- Continue to fine tune
  - experiement further with batch sizes
  - use triangular cyclical learning rate
- Play more with data before inputting into model
  - preprocessing
  - subsampling 
  - try dynamic context windows

