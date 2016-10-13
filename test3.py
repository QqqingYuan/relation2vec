__author__ = 'PC-LiNing'


def preprocess_sent(sentence):
    # ' 's 't
    strip_list = ['\n','\"','.']
    replace_list = [',',';','\'']
    delete_list = ['-','\"',':','%','(',')','.','\'t','!']
    for strip in strip_list:
        sentence = sentence.strip(strip)
    for replace in replace_list:
        sentence = sentence.replace(replace,' '+replace)
    for delete in delete_list:
        sentence = sentence.replace(delete,' ')

    return ' '.join(sentence.split())


def  Parse_Sentence(sentence):
    words=preprocess_sent(sentence).split()
    e1=-1
    e2=-1
    for word in words:
        if word.startswith('<e1>'):
            e1=words.index(word)
        if word.startswith('<e2>'):
            e2=words.index(word)

    sent=preprocess_sent(sentence).replace('<e1>','').replace('</e1>','').replace('<e2>','').replace('</e2>','')
    print(sent)
    print(e1)
    print(e2)

# 9 , 11 , 15 , 18

sentence = "Also, very rarely, hypnotherapy leads to the development of 'false <e1>memories</e1>' fabricated by the unconscious <e2>mind</e2>; these are called confabulations."
Parse_Sentence(sentence)
# print(preprocess_sent(sentence))