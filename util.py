__author__ = 'PC-LiNing'

from gensim.models.word2vec import Word2Vec
import numpy as np

# wikienvectors.bin - 200
# GoogleNews - 300
# text8.bin - 100
# wikiparta[a,b]vec.bin - 100
# model = word2vec.load('data/GoogleNews-vectors-negative300.bin')
model = Word2Vec.load_word2vec_format('wordvec/wikienpartabvec.bin',binary=True)

word_embedding_size = 100

# exception word
# 693
exception_words = []


# random init a 100-dim vector
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
exception_embeddings = init_exception_words(693)

# word embedding 's dimension is 100
def  getSentence_matrix(sentence,Max_length):
     words=sentence.split()
     # sent_matrix=np.ndarray(shape=(Max_length,word_embedding_size),dtype=np.float32)
     sent_matrix=np.zeros(shape=(Max_length,word_embedding_size),dtype=np.float32)
     i=0
     for word in words:
         try:
            vec=model[word]
            sent_matrix[i]=vec
         except KeyError as e:
             # if word in exception_words
             if word not in exception_words:
                 exception_words.append(word)

             vec=exception_embeddings[exception_words.index(word)]
             sent_matrix[i]=vec
         i+=1
     return sent_matrix

