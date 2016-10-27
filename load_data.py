__author__ = 'PC-LiNing'

import numpy
import util

word_embedding_size = util.word_embedding_size
num_classes = 10

# get e1,e2 position
def  Parse_Sentence(sentence):
    words=sentence.split()
    e1=0
    e2=0
    for word in words:
        if word.startswith('<e1>'):
            e1=words.index(word)
        if word.startswith('<e2>'):
            e2=words.index(word)

    sent=sentence.replace('<e1>','').replace('</e1>','').replace('<e2>','').replace('</e2>','').strip('\n').strip('.').strip('\"')
    return  sent,e1,e2

# relation label to number
# 1 Cause-Effect
# 2 Instrument-Agency
# 3 Product-Producer
# 4 Content-Container
# 5 Entity-Origin
# 6 Entity-Destination
# 7 Component-Whole
# 8 Member-Collection
# 9 Message-Topic
# 10 Other
def transfer_label(label):
    if label.startswith('Cause-Effect'):
        return 0
    if label.startswith('Instrument-Agency'):
        return 1
    if label.startswith('Product-Producer'):
        return 2
    if label.startswith('Content-Container'):
        return 3
    if label.startswith('Entity-Origin'):
        return 4
    if label.startswith('Entity-Destination'):
        return 5
    if label.startswith('Component-Whole'):
        return 6
    if label.startswith('Member-Collection'):
        return 7
    if label.startswith('Message-Topic'):
        return 8
    if label.startswith('Other'):
        return 9

# read SemEval train file
# (sentence,e1_position,e2_position,label)
def  SemEval_train_data():
     file = open("SemEval/TRAIN_FILE.TXT")
     sentence=[]
     label=[]
     i=1
     for line in file.readlines():
         if i % 4 == 1:
             sentence.append(line.split('	')[1])
         if i % 4 == 2:
             label.append(line)
         i+=1
     # parse
     train_data=[]
     for i in range(0,len(sentence)):
         sen=sentence[i]
         type=label[i]
         s,e1,e2=Parse_Sentence(sen)
         relation=transfer_label(type)
         train_data.append((s,e1,e2,relation))
     return train_data

# read SemEval test file
# (sentence,e1_position,e2_position,label)
def  SemEval_test_data():
    file = open("SemEval/TEST_FILE_FULL.TXT")
    sentence=[]
    label=[]
    i=1
    for line in file.readlines():
        if i % 4 == 1:
            sentence.append(line.split('	')[1])
        if i % 4 == 2:
            label.append(line)
        i+=1
    # parse
    test_data=[]
    for i in range(0,len(sentence)):
        sen=sentence[i]
        type=label[i]
        s,e1,e2=Parse_Sentence(sen)
        relation=transfer_label(type)
        test_data.append((s,e1,e2,relation))

    return test_data


# get max length of sentences
def get_Max_length(texts):
    return max([len(x[0].split(" ")) for x in texts])

def get_Min_length(texts):
    return min([len(x[0].split(" ")) for x in texts])

MAX_DOCUMENT_LENGTH = max(get_Max_length(SemEval_train_data()),get_Max_length(SemEval_test_data()))


# change class number  to (num_class,) vector
def getLabelVector(number,num_class):
    vec = numpy.zeros(num_class)
    vec[number] = 1.0
    return vec

# parse SemEval train data
def load_train_data():
    semeval_data = SemEval_train_data()
    Train_Size = len(semeval_data)
    train_data = numpy.ndarray(shape=(Train_Size,MAX_DOCUMENT_LENGTH,word_embedding_size),dtype=numpy.float32)
    train_label = numpy.ndarray(shape=(Train_Size,num_classes),dtype=numpy.float32)
    i = 0
    for one in semeval_data:
        sentence = one[0]
        train_data[i]=util.getSentence_matrix(sentence,MAX_DOCUMENT_LENGTH)
        train_label[i]=getLabelVector(one[3],num_class=num_classes)
        i+=1

    return train_data,train_label

# parse SemEval test data
def load_test_data():
    semeval_data = SemEval_test_data()
    Train_Size = len(semeval_data)
    train_data = numpy.ndarray(shape=(Train_Size,MAX_DOCUMENT_LENGTH,word_embedding_size),dtype=numpy.float32)
    train_label = numpy.ndarray(shape=(Train_Size,num_classes),dtype=numpy.float32)
    i = 0
    for one in semeval_data:
        sentence = one[0]
        train_data[i]=util.getSentence_matrix(sentence,MAX_DOCUMENT_LENGTH)
        train_label[i]=getLabelVector(one[3],num_class=num_classes)
        i+=1

    return train_data,train_label

