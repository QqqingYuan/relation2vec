__author__ = 'PC-LiNing'

import re
import networkx as nx
from practnlptools.tools import Annotator


annotator = Annotator()

text = "For us the term artists' book simply means a book made by an artist a book made as a work of art rather than as a literary artifact ."

e1 = ''
e2 = ''

dep_parse = annotator.getAnnotations(text,dep_parse=True)['dep_parse']

print(dep_parse)


dp_list = dep_parse.split('\n')

pattern = re.compile(r'.+?\((.+?), (.+?)\)')
edges = []

for dep in dp_list:
	m = pattern.search(dep)
	edges.append((m.group(1),m.group(2)))

graph = nx.Graph(edges)


length = nx.shortest_path_length(graph,source='artists-5',target='book-7')

path = nx.shortest_path(graph,source='artists-5',target='book-7')

print(length)

print(path)