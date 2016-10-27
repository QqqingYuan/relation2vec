__author__ = 'PC-LiNing'

import re

import numpy

import util
import spacy_parser

word_embedding_size = util.word_embedding_size
num_classes = 19


def preprocess_sent(sentence):
    # ' 's 't
    strip_list = ['\n','\"','.']
    replace_list = [',',';','\'']
    delete_list = ['-','\"',':','%','(',')','.','\'t','!','$']
    for strip in strip_list:
        sentence = sentence.strip(strip)
    for replace in replace_list:
        sentence = sentence.replace(replace,' '+replace)
    for delete in delete_list:
        sentence = sentence.replace(delete,' ')

    return ' '.join(sentence.split())


# get e1,e2 position
def  Parse_Sentence(sentence):
    pattern1 = re.compile(r'<e1>(.*)</e1>')
    pattern2 = re.compile(r'<e2>(.*)</e2>')
    clean_sent = preprocess_sent(sentence)
    match1 = pattern1.search(clean_sent)
    match2 = pattern2.search(clean_sent)
    if match1:
        entity_e1 = match1.group(1).split()
    if match2:
        entity_e2 = match2.group(1).split()

    sent = clean_sent.replace('<e1>','').replace('</e1>','').replace('<e2>','').replace('</e2>','')
    words = sent.split()
    try:
        e1 = words.index(entity_e1[0])
        e2 = words.index(entity_e2[0])
    except Exception as e:
        print(sentence)

    # dependency path
    path =sentence.replace('<e1>','').replace('</e1>','').replace('<e2>','').replace('</e2>','').strip('\n').strip('.').strip('\"')
    try:
        path = spacy_parser.parse_sent(sent,entity_e1,e1,entity_e2,e2)
    except Exception as e:
        print(sentence)
        # pass
    return  path,e1,e2

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
    if label.startswith('Cause-Effect(e1,e2)'):
        return 0
    if label.startswith('Cause-Effect(e2,e1)'):
        return 1
    if label.startswith('Instrument-Agency(e1,e2)'):
        return 2
    if label.startswith('Instrument-Agency(e2,e1)'):
        return 3
    if label.startswith('Product-Producer(e1,e2)'):
        return 4
    if label.startswith('Product-Producer(e2,e1)'):
        return 5
    if label.startswith('Content-Container(e1,e2)'):
        return 6
    if label.startswith('Content-Container(e2,e1)'):
        return 7
    if label.startswith('Entity-Origin(e1,e2)'):
        return 8
    if label.startswith('Entity-Origin(e2,e1)'):
        return 9
    if label.startswith('Entity-Destination(e1,e2)'):
        return 10
    if label.startswith('Entity-Destination(e2,e1)'):
        return 11
    if label.startswith('Component-Whole(e1,e2)'):
        return 12
    if label.startswith('Component-Whole(e2,e1)'):
        return 13
    if label.startswith('Member-Collection(e1,e2)'):
        return 14
    if label.startswith('Member-Collection(e2,e1)'):
        return 15
    if label.startswith('Message-Topic(e1,e2)'):
        return 16
    if label.startswith('Message-Topic(e2,e1)'):
        return 17
    if label.startswith('Other'):
        return 18
    print('error')


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
