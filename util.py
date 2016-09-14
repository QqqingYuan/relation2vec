__author__ = 'PC-LiNing'

from gensim.models.word2vec import Word2Vec
import numpy as np

# wikienvectors.bin - 200
# GoogleNews - 300
# text8.bin - 100
# wikiparta[a,b]vec.bin - 100
# model = word2vec.load('data/GoogleNews-vectors-negative300.bin')
model = Word2Vec.load_word2vec_format('data/wikienpartabvec.bin',binary=True)

word_embedding_size = 100


# random init a 300-dim vector
def getRandom_vec():
    vec=np.random.rand(word_embedding_size)
    norm=np.sum(vec**2)**0.5
    normalized = vec / norm
    return normalized

# word embedding 's dimension is 300
def  getSentence_matrix(sentence,Max_length):
     words=sentence.split()
     sent_matrix=np.ndarray(shape=(Max_length,word_embedding_size),dtype=np.float32)
     i=0
     for word in words:
         try:
            vec=model[word]
            sent_matrix[i]=vec
         except KeyError as e:
            vec=getRandom_vec()
            sent_matrix[i]=vec
         i+=1
     return sent_matrix

