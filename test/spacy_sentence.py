__author__ = 'PC-LiNing'

import spacy_parser
import re


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
    e1 = words.index(entity_e1[0])
    e2 = words.index(entity_e2[0])
    # dependency path
    path =sentence.replace('<e1>','').replace('</e1>','').replace('<e2>','').replace('</e2>','').strip('\n').strip('.').strip('\"')
    try:
        path = spacy_parser.parse_sent(sent,entity_e1,e1,entity_e2,e2)
    except Exception as e:
        print(sentence)
        # pass
    return  path,e1,e2

sentence = "The chavettes are the female of the species, usually have a '<e1>passel</e1> of <e2>brats</e2> with different dads before they turn 20 which ensures the perpetuation of the disease."
clean_sent = preprocess_sent(sentence)
print('clean sent: ')
print(clean_sent)
path = Parse_Sentence(sentence)
print('result: ')
print(path)


