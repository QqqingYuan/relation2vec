__author__ = 'PC-LiNing'

import re
import numpy as np
import spacy_parser

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
        path,d_e1,d_e2 = spacy_parser.parse_sent(sent,entity_e1,e1,entity_e2,e2)
    except Exception as e:
        print(sentence)
        # pass
    return  path,d_e1,d_e2

sentence = "The <e1>lawsonite</e1> was contained in a <e2>platinum crucible</e2> and the counter-weight was a plastic crucible with metal pieces."
print(Parse_Sentence(sentence))