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
#     graph n & cycle n & ilp dt & lp dt {chr(92)}{chr(92)}
#     {chr(92)}hline
#         {("" + chr(92) + chr(92) + chr(10)).join(strs)}{chr(92)}{chr(92)}
# {chr(92)}end{{tabular}}
# '''

# D = {
#     'graph_n': [],
#     'cycle_n': [],
#     'lc_dt': [],
#     'ilp_dt': [],
#     'lp_dt': []
# }
# S = []
# n = 10
# data = n
# while data is not None:
#     cursor.execute(f'SELECT graph_n, cycle_n, lc_dt, ilp_dt, lp_dt FROM graphs WHERE id = ?', (n,))
#     data = cursor.fetchone()
#     if data is not None:
#         graph_n, cycle_n, lc_dt, ilp_dt, lp_dt = data
#         D['graph_n'].append(graph_n)
#         D['cycle_n'].append(cycle_n)
#         D['lc_dt'].append(lc_dt)
#         D['ilp_dt'].append(ilp_dt)
#         D['lp_dt'].append(lp_dt)
#         S.append(' & '.join([str(graph_n), str(cycle_n), str(round(ilp_dt, 1)), str(lp_dt)]))
#     n += 1

# file = open('..\draft\examples\data.tex', 'w')
# file.write(interp(S))
# file.close()

# norm = lambda A: (np.array(A) - np.min(A)) / (np.max(A) - np.min(A))

# plt.plot(range(10, 10+len(D['ilp_dt'])), norm(D['ilp_dt']), label='ilp')
# plt.plot(range(10, 10+len(D['lp_dt'])), norm(D['lp_dt']), label='lp')
# plt.plot(range(10, 10+len(D['lc_dt'])), norm(D['lc_dt']), label='lc')
# plt.legend(loc='upper left')
# plt.xlabel('# faces')
# plt.show()

# exit()

# for n in range(50, 1000):
#     hc = Graph.undirected_hamcycle(n)
#     for i in range(int(sqrt(n)), ((n*(n-1))//2)-n):
#         for combo in combinations( ((a, b) for a in hc.nodes for b in hc.nodes if a[0] < b[0]-1), i):
#             graph = hc.copy()
#             for edge in combo:
#                 graph += Edge(*edge)
#             graph.polytime_hamiltonian_cycle()
#             print('done')
# exit()

i = 0
for n in range(100, 110):
    cursor.execute(f"SELECT graph, cycle_n FROM graphs WHERE id = ?", (n,))
    data = cursor.fetchone()
    graph, lc_n = data
    graph = SolidGridGraph().decompress(graph)


    ilp = graph.cycle_of_length_n_minus_boundary(lc_n, ilp=True)
    TikZGraph(*graph, *graph.dual())\
        .add_edges([TikZEdge.DIRECTED(TikZEdge(edge, force=True)) for edge in ilp])\
        .add_edges([TikZEdge.GRAY(TikZEdge.DOTTED(edge)) for edge in graph.single_directed()])\
        .add_background(*[f'\\fill[fill=black!10] {str(node - (0.5, 0.5))} rectangle ++(1, 1);' for node in ilp.dual().interior.nodes])\
        .on_nodes(lambda node: TikZNode.NONE(node))\
        .write(f'solution_{n}')


    # TikZGraph(*graph, *graph.dual())\
    #     .add_edges([TikZEdge.DIRECTED(TikZEdge(edge, force=True)) for edge in lp])\
    #     .add_edges([TikZEdge.GRAY(TikZEdge.DOTTED(edge)) for edge in graph.single_directed()])\
    #     .add_edges([TikZOptions.COLOR(f'red!{round(lp.model.solution[str(edge)]*100)}', TikZEdge.DIRECTED(TikZEdge(edge, force=True))) for edge in graph if is_nonintegral(lp.model.solution[str(edge)])])\
    #     .on_nodes(lambda node: TikZOptions.COLOR(f"red!{round(lp.model.solution[f'{node}_interior']*100)}", node) if node in graph.dual().interior and is_nonintegral(lp.model.solution[f'{node}_interior']) else node if node in graph.dual().interior and lp.model.solution[f'{node}_interior'] == 1 else TikZNode.NONE(node))\
    #     .write(f'boundary_solution_{i}')

    # print(f'ilp: {ilp.model.solve_details.dettime} lp: {lp.model.solve_details.dettime}')

    i += 1