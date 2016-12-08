__author__ = 'PC-LiNing'

from spacy.en import English
import networkx as nx

parser = English()

def find_all_element(words,word):
    return [i for i,a in enumerate(words) if a == word]

def find_closest_index(locations,pl_e):
    distance = -1
    target = -1
    for index,value in enumerate(locations):
        if distance < 0 :
            distance = abs(value - pl_e)
            target = value
        elif abs(value-pl_e) < distance :
            distance = abs(value-pl_e)
            target = value

    return target


def get_entity_index(words,entity_e,pl_e):
    locations = find_all_element(words,entity_e[0])
    return find_closest_index(locations,pl_e)


# parse sentence
def  parse_sent(sentence,entity_e1,pl_e1,entity_e2,pl_e2):
    # print(entity_e1)
    # print(pl_e1)
    # print(entity_e2)
    # print(pl_e2)
    parsedEx = parser(sentence)
    edges = []
    words = [None for i in range(100)]
    for token in parsedEx:
        if words[token.head.i] is None:
            words[token.head.i] = token.head.orth_
        if words[token.i] is None:
            words[token.i] = token.orth_
        # print((token.head.orth_+'-'+str(token.head.i),token.orth_+'-'+str(token.i)))
        edges.append((token.head.orth_+'-'+str(token.head.i),token.orth_+'-'+str(token.i)))

    # compute e1 , e2
    e1 = get_entity_index(words,entity_e1,pl_e1)
    # print(e1)
    e2 = get_entity_index(words,entity_e2,pl_e2)
    # print(e2)
    e1_item = entity_e1[0]+'-'+str(e1)
    e2_item = entity_e2[0]+'-'+str(e2)
    graph = nx.Graph(edges)
    path = nx.shortest_path(graph,source=e1_item,target=e2_item)
    path2 = [item.split('-')[0] for item in path]
    # sort entity_e2
    path3 = []
    for word in path2:
        if word not in entity_e2:
            path3.append(word)
    for word in entity_e2:
        path3.append(word)

    e1 = path3.index(entity_e1[0])
    e2 = path3.index(entity_e2[0])
    sent = ' '.join(path3)
    return sent,e1,e2




