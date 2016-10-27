__author__ = 'PC-LiNing'

import spacy_parser


sentence = "Fall planting cannot be completed before the ground freezes , so I stored the happy seeds in a stratification unit until spring"
entity_e1 = ['happy','seeds']
e1 = 14
entity_e2 = ['stratification','unit']
e2= 18
result = spacy_parser.parse_sent(sentence,entity_e1,e1,entity_e2,e2)
print(result)
