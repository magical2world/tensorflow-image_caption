import tensorflow as tf
import config
from model import network
#from InceptionV3 import network
def main(args,train=False):
    if train:
        image=tf.placeholder(tf.float32,[args.batch_size,224,224,3])
        sentence=tf.placeholder(tf.int32,[args.batch_size,None])
    else:
        args.batch_size=1
        image=tf.placeholder(tf.float32,[1,224,224,3])
        sentence=tf.placeholder(tf.int32,[1,None])
    model=network(image,sentence,args)
    if train:
        model.train_network()
    else:
        model.test_network()

if __name__=="__main__":
    args=config.config()
    main(args)
