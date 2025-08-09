from graph import *
from to_tex import *
from itertools import product
import matplotlib.pyplot as plt
import os, pickle, io
import numpy as np
from math import pi
import sqlite3

conn = sqlite3.connect('graph.db')
cursor = conn.cursor()

# interp = lambda strs: f'''
# {chr(92)}begin{{tabular}}{{ |c|c|c|c| }}
#     {chr(92)}hline
#     nodes & covered & t & cuts {chr(92)}{chr(92)}
#     {chr(92)}hline
#         {("" + chr(92) + chr(92) + chr(10)).join(strs)}{chr(92)}{chr(92)}
# {chr(92)}end{{tabular}}
# '''

# D = {
#     'graph_n': [],
#     'ilp_n': [],
#     'dt': [],
#     'n_cuts': []
# }
# S = []
# n = 10
# data = n
# while data is not None:
#     cursor.execute(f'SELECT graph_n, ilp_n, dt, n_cuts FROM graphs WHERE id = ?', (n,))
#     data = cursor.fetchone()
#     if data is not None:
#         graph_n, ilp_n, dt, n_cuts = data
#         D['graph_n'].append(graph_n)
#         D['ilp_n'].append(ilp_n)
#         D['dt'].append(dt)
#         D['n_cuts'].append(n_cuts)
#         S.append(' & '.join([str(graph_n), str(ilp_n), str(round(dt, 1)), str(n_cuts)]))
#     n += 1

# file = open('..\draft\examples\data.tex', 'w')
# file.write(interp(S))
# file.close()

# print(D['n_cuts'])

# plt.plot(range(10, 10+len(D['dt'])), D['dt'], label='dettime')
# plt.plot(range(10, 10+len(D['n_cuts'])), D['n_cuts'], label='# fractional cuts')
# plt.plot(range(10, 10+len(D['graph_n'])), [g - i for (g, i) in zip(D['graph_n'], D['ilp_n'])], label='nodes - covered' )
# plt.legend(loc='upper left')
# plt.xlabel('# faces')
# plt.show()

# exit()

i = 0
for n in range(50, 60):
    cursor.execute(f"SELECT graph, ilp FROM graphs WHERE id = ?", (n,))
    data = cursor.fetchone()
    graph, ilp = data
    graph, ilp = SolidGridGraph().decompress(graph), SolidGridGraph().decompress(ilp)
    # ilp = graph.test(k=k)
    # ilp = graph.combination_ilp()
    # print()
    if len(ilp.nodes) < len(graph.nodes):
        # k = len(graph.nodes) - len(ilp.nodes)
        ilp = graph.test()
        TikZGraph(*graph, *ilp, *graph.dual())\
            .add_edges([TikZEdge.DIRECTED(TikZEdge(edge, force=True)) for edge in ilp])\
            .add_edges([TikZEdge.GRAY(TikZEdge.DOTTED(edge)) for edge in graph.single_directed()])\
            .on_nodes(lambda node: TikZNode.NONE(node) if node not in ilp.dual().interior else node)\
            .write(f'ilp_example_{i}')
        i += 1