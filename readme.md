## Introduction
A bare-bones but working tensorflow implementation of the paper [Multiresolution Recurrent Neural Networks: An Application to Dialogue Response Generation](https://arxiv.org/abs/1606.00776).

## Description of the files
* mr_rnn.py -- builds the graph
* train.py -- to train the model
* data_aux.py -- help functions to generate data

## Comments
Built in Python 3.5.2 with Tensorflow 0.10.0.

To run the program you have to add Googles w2v (from https://code.google.com/archive/p/word2vec/) to data/ or use random_embedding or create UbuntuWord2Vec using train_word2vec instead as well as add the extracted Ubuntu dialogoue corpus (from www.iulianserban.com/Files/UbuntuDialogueCorpus.zip) to data/. 
