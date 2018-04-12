# tensorflow-image_caption

## Introduction
A model for image caption which reference https://arxiv.org/abs/1609.06647

## Using
For training run `python main.py --mode=='train'`
For testing run  `python main.py --mode=='test'`

Other parameters:

batch_size:Set batch size of mini_batch
learning_rate:Set learn rate
train_step:Number of batch to train
hidden_layers:Number of layers of LSTM
hidden_units:Number of units of each layer LSTM
