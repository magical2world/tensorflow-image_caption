import argparse

def config():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",type=str,default='train',help="train or test")
    parser.add_argument("--batch_size",type=int,default=64,help="batch size for training")
    parser.add_argument("--learning_rate",type=float,default=0.0001,help="learning rate")
    parser.add_argument("--train_step",type=int,default=40000,help="number of step to train")
    parser.add_argument("--hidden_layers",type=int,default=3,help="rnn hidden layers")
    parser.add_argument("--hidden_units",type=int,default=512,help="rnn hidden units")

    return parser.parse_args()
