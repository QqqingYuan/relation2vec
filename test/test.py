__author__ = 'PC-LiNing'

import re

# re.escape
sentence = "Fall planting cannot be completed before the ground freezes, so I stored the <e1>seeds</e1> in a <e2>stratification unit</e2> until spring."

pattern1 = re.compile(r'<e1>(.*)</e1>')
pattern2 = re.compile(r'<e2>(.*)</e2>')
match = pattern1.search(sentence)

if match:
    print(match.group(1).split()[0])