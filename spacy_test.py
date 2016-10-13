__author__ = 'PC-LiNing'

from spacy.en import English
import networkx as nx



parser = English()

example = 'As in the popular movie Deep Impact , the action of the Perseid meteor shower is caused by a comet , in this case periodic comet Swift Tuttle'
position1 = 13
position2 = 19
e1 = example.split()[position1]+'-'+str(position1)
e2 = example.split()[position2]+'-'+str(position2)

parsedEx = parser(example)
edges = []

for token in parsedEx:
    print((token.head.orth_+'-'+str(token.head.i),token.orth_+'-'+str(token.i)))
    edges.append((token.head.orth_+'-'+str(token.head.i),token.orth_+'-'+str(token.i)))

graph = nx.Graph(edges)
path = nx.shortest_path(graph,source=e1,target=e2)
path2 = [item.split('-')[0] for item in path]
sent = ' '.join(path2)
print(sent)