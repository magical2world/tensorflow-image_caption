import tensorflow as tf
import vgg19
from utils import *

vgg=vgg19.Vgg19()

class network():
    def __init__(self,image,sentence,config):
        self.image=image
        self.sentence=sentence
        self.config=config
        self.sentences=load_sentence()
        self.dict,self.total_words=build_dict(self.sentences)

    def cnn_features(self):
        features=vgg.build(self.image)
        features=tf.layers.flatten(features)
        return features
    def rnn_model(self,input_sequence,len_words):
        def cell():
            return tf.nn.rnn_cell.LSTMCell(self.config.hidden_units)
        rnn_cell=tf.nn.rnn_cell.MultiRNNCell([cell() for _ in range(self.config.hidden_layers)])
        initial_state=rnn_cell.zero_state(self.config.batch_size,tf.float32)
        rnn_outputs,last_state=tf.nn.dynamic_rnn(rnn_cell,input_sequence,initial_state=initial_state,scope='LSTM')
        outputs=tf.reshape(rnn_outputs,[-1,self.config.hidden_units])
        logits=tf.layers.dense(outputs,len_words)
        probs=tf.nn.softmax(logits)
        probs=tf.reshape(probs,[1,-1,len_words])
        return logits,last_state,probs
    def get_input(self):
        cnn_features=self.cnn_features()
        fc1=tf.layers.dense(cnn_features, 1024, activation=tf.nn.relu)
        fc2=tf.layers.dense(fc1, self.config.hidden_units, activation=tf.nn.sigmoid)
        cnn_features = tf.expand_dims(fc2, 1)
        with tf.device("/cpu:0"):
            embedding=tf.get_variable("embedding", [self.total_words, self.config.hidden_units])
            sentence=tf.nn.embedding_lookup(embedding, self.sentence)
        input=tf.concat([cnn_features,sentence[:,:-1, :]], 1)
        return input
    def train_network(self):
        input=self.get_input()
        logits,last_state,_=self.rnn_model(input,self.total_words)
        targets = self.sentence
        logits = tf.reshape(logits, [self.config.batch_size, -1, self.total_words])
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits), axis=1)
        cost = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(cost)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver=tf.train.Saver(tf.all_variables())
            gen_next=next_batch(self.dict,self.config.batch_size)
            losses=[]
            for i in range(self.config.train_step):
                img,sen=gen_next.next()
                feed_dict = {self.image: img, self.sentence: sen}
                train_loss,_=sess.run([cost,optimizer],
                                      feed_dict=feed_dict)
                losses.append(train_loss)
                if i%10==0:
                    saver.save(sess,'variables/image-caption.module',global_step=i)
                    print("number %d loss is %f"%(i,train_loss))
            import matplotlib.pyplot as plt
            plt.plot(losses)
            plt.show()

    def test_network(self):
        num_to_word={}
        for word in self.dict.keys():
            num_to_word[self.dict[word]]=word
        input=self.get_input()
        _,_,probs=self.rnn_model(input,self.total_words)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver=tf.train.Saver()
            saver.restore(sess,'variables/image-caption.module-28680')

            import matplotlib.pyplot as plt
            while(1):
                image_id=raw_input('Please input image id:')
                if image_id=='stop':
                    break
                try:
                    image=test_image(image_id)
                except:
                    print('Please input the correct image name.')
                sentence_=np.array([self.dict['BOS']])
                sentence_=np.reshape(sentence_,[1,-1])
                probs_=sess.run(probs,feed_dict={self.image:image,self.sentence:sentence_})
                num=np.argmax(probs_[:,-1,:])
                word=num_to_word[num]
                gen_sentence=word
                while word!='EOS':
                    sentence_=np.hstack((sentence_,num.reshape(1,-1)))
                    sentence_=np.reshape(sentence_,[1,-1])
                    probs_=sess.run(probs,feed_dict={self.image:image,self.sentence:sentence_})
                    num=np.argmax(probs_[:,-1,:])
                    if num_to_word[num]==word:
                        continue
                    else:
                        word=num_to_word[num]
                    gen_sentence = gen_sentence + word + ' '
                print(gen_sentence[3:-4])
                plt.imshow(np.array(image).reshape([224,224,-1]))
                plt.show()
