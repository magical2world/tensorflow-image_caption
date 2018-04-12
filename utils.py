import numpy as np
import nltk
import sys
import os
import collections
import skimage
import skimage.io
import skimage.color
import skimage.transform
sys.path.append('/home/liushaolin/cocoapi/PythonAPI/pycocotools')
import coco

image_path='/home/liushaolin/cocoapi/train2014'
test_image_path='/home/liushaolin/cocoapi/test2014'
datasets=coco.COCO(annotation_file='/home/liushaolin/cocoapi/annotations/captions_train2014.json')
datasets.createIndex()
anns=datasets.anns

def load_image(filename):
    #filename=path+'/'+name
    image=skimage.io.imread(filename)
    if image.shape!=3:
        image=skimage.color.gray2rgb(image)
    image=image/255.0
    assert (0<=image).all() and (image<=1.0).all()
    short_edge=min(image.shape[:2])
    yy=int((image.shape[0]-short_edge)/2)
    xx=int((image.shape[1]-short_edge)/2)
    crop_img=image[yy:yy+short_edge,xx:xx+short_edge]
    resized_img=skimage.transform.resize(crop_img,(224,224))
    return resized_img

def load_sentence():
    sentences=[]
    for ann in anns:
        line=anns[ann]['caption']
        line=line.strip()
        sentences.append(['BOS']+nltk.word_tokenize(line)+['EOS'])
    return sentences

def build_dict(sentences,max_words=50000):
    word_count=collections.Counter()
    for sentence in sentences:
        for s in sentence:
            word_count[s]+=1
    ls=word_count.most_common(max_words)
    total_word=len(ls)+1
    word_dict={w[0]:index+1 for (index,w) in enumerate(ls)}
    word_dict["UNK"]=0
    return word_dict,total_word

def encode(sentences,dict):
    length=len(sentences)
    out_sentences=[]
    for i in range(length):
        sentences[i]='BOS '+sentences[i]+' EOS'
        en_seq=[dict[w] if w in dict else 0 for w in nltk.word_tokenize(sentences[i])]
        out_sentences.append(en_seq)
    return out_sentences

def prepare_data(seqs):
    B=len(seqs)
    lengths=[len(seq) for seq in seqs]
    max_len=np.max(lengths)
    x=np.zeros((B,max_len)).astype('int32')
    for idx,seq in enumerate(seqs):
        x[idx,:lengths[idx]]=seq
    return x

def next_batch(dict,batch_size):
    path=image_path
    i=0
    image=[]
    caption=[]
    while(True):
        for ann in anns:
            i+=1
            id=str(anns[ann]['image_id'])
            id=id.zfill(12)
            id='COCO_train2014_'+id+'.jpg'
            img=load_image(path+'/'+id)
            image.append(img)
            sentence=anns[ann]['caption']
            caption.append(sentence)
            if i%batch_size==0:
                encode_sentence=encode(caption, dict)
                seq=prepare_data(encode_sentence)
                yield image,seq
                image=[]
                caption=[]
                i=0

def test_image(image_id):
    path=test_image_path
    image_name=path+'/'+image_id
    img=load_image(image_name)
    img=np.reshape(img,[1,224,224,3])
    return img

