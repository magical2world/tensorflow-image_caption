# tensorflow-image_caption

## Introduction

A model for image caption which reference https://arxiv.org/abs/1609.06647

## Usage

To usage this network ,you need to download [VGG19 NPY](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs)

For training run `python main.py --mode=='train'`

For testing run  `python main.py --mode=='test'`

Other parameters:

batch_size:Set batch size of mini_batch

learning_rate:Set learn rate

train_step:Number of batch to train

hidden_layers:Number of layers of LSTM

hidden_units:Number of units of each layer LSTM
