__author__ = 'PC-LiNing'

from spacy.en import English
import networkx as nx

parser = English()

# parse sentence
def  parse_sent(sentence,position1,position2):
    e1 = sentence.split()[position1]+'-'+str(position1)
    e2 = sentence.split()[position2]+'-'+str(position2)
    parsedEx = parser(sentence)
    edges = []
    for token in parsedEx:
        edges.append((token.head.orth_+'-'+str(token.head.i),token.orth_+'-'+str(token.i)))
    graph = nx.Graph(edges)
    path = nx.shortest_path(graph,source=e1,target=e2)
    path2 = [item.split('-')[0] for item in path]
    sent = ' '.join(path2)
    return sent




