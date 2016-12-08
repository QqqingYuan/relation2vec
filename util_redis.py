__author__ = 'PC-LiNing'

import numpy as np
import redis

word_embedding_size = 200

# redis
r = redis.StrictRedis(host='10.2.1.35',port=6379,db=0)

# exception word
# sdp 674
# sentence 18188
exception_words = []

# random init a 200-dim vector
def getRandom_vec():
    vec=np.random.rand(word_embedding_size)
    norm=np.sum(vec**2)**0.5
    normalized = vec / norm
    return normalized

# random exception words vector
def init_exception_words(number):
    exp_matrix = np.zeros(shape=(number,word_embedding_size),dtype=np.float32)
    for i in range(number):
        exp_matrix[i] = getRandom_vec()
    return exp_matrix

# exception word embed
exception_embeddings = init_exception_words(674)


# word embedding 's dimension is 200
def  getSentence_matrix(sentence,Max_length):
     words=sentence.split()
     # sent_matrix=np.ndarray(shape=(Max_length,word_embedding_size),dtype=np.float32)
     sent_matrix=np.zeros(shape=(Max_length,word_embedding_size),dtype=np.float32)
     i=0
     for word in words:
         result = r.get(word)
         if result is not None:
             vec=np.fromstring(r.get(word),dtype=np.float32)
             sent_matrix[i]=vec
         else:
             # if word in exception_words
             if word not in exception_words:
                 exception_words.append(word)

             vec=exception_embeddings[exception_words.index(word)]
             sent_matrix[i]=vec
         i+=1
     return sent_matrix

