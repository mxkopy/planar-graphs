from graph import *
from to_tex import *
from itertools import product
import matplotlib.pyplot as plt
import os
import numpy as np
from math import pi


def shaded_faces(graph, faces, style='fill=black!10'):
    for face in faces:
        x, y = graph.midpoint_at_face(face)
        yield f"{chr(92)}fill[{style}] ({x-0.5}cm-1pt, {y-0.5}cm-1pt) rectangle ++(1cm+2pt, 1cm+2pt);"

def shade_two_factor(graph, style='fill=black!10'):
    yield from shaded_faces(graph, (face for face in graph.faces if graph.two_factor.test_interior(graph.midpoint_at_face(face))), style=style)

def visualize_hamiltonian_cycle(name, graph):
    i = 0
    tgraph = TikZGraph(*list(graph))
    tgraph.make_bipartite()
    tgraph.two_factor = graph.two_factor
    shade_two_factor(tgraph)
    tfc = Graph(*graph.two_factor.as_cycle_facing_inwards()).nodes
    while set(tfc) != set(graph.nodes) and len(tfc) != 0:
        strip = graph.static_alternating_strip()
        tgraph.add_edges([TikZEdge(edge) for edge in strip])
        tgraph.add_edges([TikZEdge.GRAY(TikZEdge.DOTTED(edge)) for edge in tgraph if not tgraph.two_factor.test_interior(edge.midpoint()) and edge not in tgraph.two_factor])
        tgraph.write(f'{name}{i}')
        graph.edge_flip(strip)
        tfc = Graph(*graph.two_factor.as_cycle_facing_inwards()).nodes
        tgraph = TikZGraph(*list(graph))
        tgraph.make_bipartite()
        tgraph.two_factor = graph.two_factor
        shade_two_factor(tgraph)
        i += 1
    tgraph.write(f'{name}{i}')

# visualize_hamiltonian_cycle('uot', sgg)

strategy = SolidGridGraph()
for r in range(3):
    for c in range(3):
        strategy[(r, c), assign]

for edge in list(SGG_iterator(strategy.nodes, 5))[2]:
    strategy += Edge(*edge)

tikz_strategy = TikZGraph(*strategy)

tikz_strategy.make_bipartite()\
    .add_edges(TikZEdge.DIRECTED(edge) for edge in tikz_strategy)\
    .write("strategy")

i = 0
# for graph in solid_grid_graph_iterator(4, 4, minimum=12):
for graph in [
    SolidGridGraph(((6, 6), (6, 7)), ((6, 7), (6, 6)), ((6, 7), (7, 7)), ((7, 7), (6, 7)), ((7, 7), (7, 8)), ((7, 8), (7, 7)), ((7, 8), (8, 8)), ((8, 8), (7, 8)), ((8, 8), (8, 9)), ((8, 9), (8, 8)), ((8, 9), (9, 9)), ((9, 9), (8, 9)), ((9, 9), (9, 8)), ((9, 8), (9, 9)), ((9, 8), (9, 7)), ((9, 7), (9, 8)), ((9, 7), (10, 7)), ((10, 7), (9, 7)), ((10, 7), (11, 7)), ((11, 7), (10, 7)), ((11, 7), (11, 6)), ((11, 6), (11, 7)), ((11, 6), (11, 5)), ((11, 5), (11, 6)), ((11, 5), (12, 5)), ((12, 5), (11, 5)), ((12, 5), (13, 5)), ((13, 5), (12, 5)), ((13, 5), (13, 4)), ((13, 4), (13, 5)), ((13, 4), (13, 3)), ((13, 3), (13, 4)), ((13, 3), (14, 3)), ((14, 3), (13, 3)), ((14, 3), (15, 3)), ((15, 3), (14, 3)), ((15, 3), (15, 2)), ((15, 2), (15, 3)), ((15, 2), (15, 1)), ((15, 1), (15, 2)), ((15, 1), (14, 1)), ((14, 1), (15, 1)), ((14, 1), (14, 0)), ((14, 0), (14, 1)), ((14, 0), (13, 0)), ((13, 0), (14, 0)), ((13, 0), (12, 0)), ((12, 0), (13, 0)), ((12, 0), (12, 1)), ((12, 1), (12, 0)), ((12, 1), (12, 2)), ((12, 2), (12, 1)), ((12, 2), (11, 2)), ((11, 2), (12, 2)), ((11, 2), (10, 2)), ((10, 2), (11, 2)), ((10, 2), (10, 3)), ((10, 3), (10, 2)), ((10, 3), (10, 4)), ((10, 4), (10, 3)), ((10, 4), (9, 4)), ((9, 4), (10, 4)), ((9, 4), (8, 4)), ((8, 4), (9, 4)), ((8, 4), (8, 5)), ((8, 5), (8, 4)), ((8, 5), (8, 6)), ((8, 6), (8, 5)), ((7, 6), (8, 6)), ((8, 6), (7, 6)), ((6, 6), (7, 6)), ((7, 6), (6, 6)), ((7, 7), (7, 6)), ((7, 6), (7, 7)), ((7, 7), (8, 7)), ((8, 7), (7, 7)), ((8, 7), (8, 8)), ((8, 8), (8, 7)), ((8, 7), (9, 7)), ((9, 7), (8, 7)), ((8, 8), (9, 8)), ((9, 8), (8, 8)), ((8, 7), (8, 6)), ((8, 6), (8, 7)), ((8, 6), (9, 6)), ((9, 6), (8, 6)), ((9, 7), (9, 6)), ((9, 6), (9, 7)), ((9, 6), (9, 5)), ((9, 5), (9, 6)), ((8, 5), (9, 5)), ((9, 5), (8, 5)), ((9, 6), (10, 6)), ((10, 6), (9, 6)), ((10, 6), (10, 7)), ((10, 7), (10, 6)), ((10, 6), (10, 5)), ((10, 5), (10, 6)), ((9, 5), (10, 5)), ((10, 5), (9, 5)), ((9, 5), (9, 4)), ((9, 4), (9, 5)), ((10, 5), (10, 4)), ((10, 4), (10, 5)), ((10, 5), (11, 5)), ((11, 5), (10, 5)), ((10, 6), (11, 6)), ((11, 6), (10, 6)), ((11, 5), (11, 4)), ((11, 4), (11, 5)), ((12, 5), (12, 4)), ((12, 4), (12, 5)), ((12, 4), (12, 3)), ((12, 3), (12, 4)), ((11, 3), (12, 3)), ((12, 3), (11, 3)), ((11, 4), (11, 3)), ((11, 3), (11, 4)), ((10, 3), (11, 3)), ((11, 3), (10, 3)), ((11, 3), (11, 2)), ((11, 2), (11, 3)), ((12, 3), (13, 3)), ((13, 3), (12, 3)), ((12, 3), (12, 2)), ((12, 2), (12, 3)), ((12, 2), (13, 2)), ((13, 2), (12, 2)), ((13, 3), (13, 2)), ((13, 2), (13, 3)), ((13, 2), (14, 2)), ((14, 2), (13, 2)), ((14, 2), (14, 3)), ((14, 3), (14, 2)), ((14, 2), (15, 2)), ((15, 2), (14, 2)), ((14, 2), (14, 1)), ((14, 1), (14, 2)), ((13, 1), (14, 1)), ((14, 1), (13, 1)), ((13, 2), (13, 1)), ((13, 1), (13, 2)), ((12, 1), (13, 1)), ((13, 1), (12, 1)), ((13, 1), (13, 0)), ((13, 0), (13, 1)), ((13, 4), (12, 4)), ((12, 4), (13, 4)), ((10, 4), (11, 4)), ((11, 4), (10, 4)), ((11, 4), (12, 4)), ((12, 4), (11, 4))),
    # SolidGridGraph(((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 2), (1, 2)), ((1, 2), (1, 1)), ((0, 1), (1, 1)), ((0, 0), (1, 0)), ((1, 1), (1, 0)), ((1, 2), (2, 2)), ((2, 2), (2, 1)), ((1, 1), (2, 1)), ((2, 1), (2, 0)), ((1, 0), (2, 0)), ((2, 2), (3, 2)), ((3, 2), (3, 1)), ((2, 1), (3, 1)), ((3, 1), (3, 0)), ((2, 0), (3, 0)), ((0, 3), (0, 2)), ((0, 3), (1, 3)), ((1, 3), (1, 2)), ((1, 3), (2, 3)), ((2, 3), (2, 2)), ((2, 3), (3, 3)), ((3, 3), (3, 2))),
    # SolidGridGraph(((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 2), (1, 2)), ((1, 2), (1, 1)), ((0, 1), (1, 1)), ((0, 0), (1, 0)), ((1, 1), (1, 0)), ((1, 2), (2, 2)), ((2, 2), (2, 1)), ((1, 1), (2, 1)), ((2, 1), (2, 0)), ((1, 0), (2, 0)), ((2, 2), (3, 2)), ((3, 2), (3, 1)), ((2, 1), (3, 1)), ((3, 1), (3, 0)), ((2, 0), (3, 0))),
    # SolidGridGraph(((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 2), (1, 2)), ((1, 2), (1, 1)), ((0, 1), (1, 1)), ((0, 0), (1, 0)), ((1, 1), (1, 0)), ((1, 2), (2, 2)), ((2, 2), (2, 1)), ((1, 1), (2, 1)), ((2, 1), (2, 0)), ((1, 0), (2, 0)), ((2, 2), (3, 2)), ((3, 2), (3, 1)), ((2, 1), (3, 1)), ((3, 1), (3, 0)), ((2, 0), (3, 0)), ((3, 2), (4, 2)), ((4, 2), (4, 1)), ((3, 1), (4, 1)), ((4, 1), (4, 0)), ((3, 0), (4, 0))),
    # SolidGridGraph(((0, 2), (0, 1)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((1, 2), (1, 1)), ((1, 2), (2, 2)), ((2, 2), (2, 1)), ((1, 1), (2, 1)), ((2, 1), (2, 0)), ((1, 1), (1, 0)), ((1, 0), (2, 0)), ((0, 1), (0, 0)), ((0, 0), (1, 0))),
    # SolidGridGraph(((3, 3), (4, 3)), ((4, 3), (3, 3)), ((4, 3), (5, 3)), ((5, 3), (4, 3)), ((5, 3), (5, 4)), ((5, 4), (5, 3)), ((5, 4), (5, 5)), ((5, 5), (5, 4)), ((5, 5), (4, 5)), ((4, 5), (5, 5)), ((4, 5), (4, 6)), ((4, 6), (4, 5)), ((4, 6), (4, 7)), ((4, 7), (4, 6)), ((4, 7), (5, 7)), ((5, 7), (4, 7)), ((5, 7), (5, 8)), ((5, 8), (5, 7)), ((5, 8), (4, 8)), ((4, 8), (5, 8)), ((4, 8), (3, 8)), ((3, 8), (4, 8)), ((3, 8), (3, 9)), ((3, 9), (3, 8)), ((3, 9), (3, 10)), ((3, 10), (3, 9)), ((3, 10), (4, 10)), ((4, 10), (3, 10)), ((4, 10), (4, 11)), ((4, 11), (4, 10)), ((4, 11), (3, 11)), ((3, 11), (4, 11)), ((3, 11), (2, 11)), ((2, 11), (3, 11)), ((2, 11), (1, 11)), ((1, 11), (2, 11)), ((1, 11), (1, 10)), ((1, 10), (1, 11)), ((1, 10), (2, 10)), ((2, 10), (1, 10)), ((2, 10), (2, 9)), ((2, 9), (2, 10)), ((2, 9), (2, 8)), ((2, 8), (2, 9)), ((2, 8), (1, 8)), ((1, 8), (2, 8)), ((1, 8), (0, 8)), ((0, 8), (1, 8)), ((0, 8), (0, 7)), ((0, 7), (0, 8)), ((0, 7), (1, 7)), ((1, 7), (0, 7)), ((1, 7), (1, 6)), ((1, 6), (1, 7)), ((1, 6), (2, 6)), ((2, 6), (1, 6)), ((2, 6), (2, 5)), ((2, 5), (2, 6)), ((2, 5), (3, 5)), ((3, 5), (2, 5)), ((3, 5), (3, 4)), ((3, 4), (3, 5)), ((3, 4), (3, 3)), ((3, 3), (3, 4)), ((3, 4), (4, 4)), ((4, 4), (3, 4)), ((4, 4), (5, 4)), ((5, 4), (4, 4)), ((4, 4), (4, 3)), ((4, 3), (4, 4)), ((4, 4), (4, 5)), ((4, 5), (4, 4)), ((3, 5), (4, 5)), ((4, 5), (3, 5)), ((3, 6), (3, 5)), ((3, 5), (3, 6)), ((3, 6), (4, 6)), ((4, 6), (3, 6)), ((2, 6), (3, 6)), ((3, 6), (2, 6)), ((3, 7), (3, 6)), ((3, 6), (3, 7)), ((3, 7), (4, 7)), ((4, 7), (3, 7)), ((4, 8), (4, 7)), ((4, 7), (4, 8)), ((3, 8), (3, 7)), ((3, 7), (3, 8)), ((2, 8), (2, 7)), ((2, 7), (2, 8)), ((1, 7), (2, 7)), ((2, 7), (1, 7)), ((2, 7), (3, 7)), ((3, 7), (2, 7)), ((2, 8), (3, 8)), ((3, 8), (2, 8)), ((2, 9), (3, 9)), ((3, 9), (2, 9)), ((3, 11), (3, 10)), ((3, 10), (3, 11)), ((2, 10), (3, 10)), ((3, 10), (2, 10)), ((2, 11), (2, 10)), ((2, 10), (2, 11)), ((1, 8), (1, 7)), ((1, 7), (1, 8)), ((2, 6), (2, 7)), ((2, 7), (2, 6)))
]:
    uot = graph.union_of_tours()
    # print(uot.is_tour())
    # exit()
    # for node in uot.nodes:
    uot = uot.kernel_patch(next(iter(uot.nodes))+Node(1, 1))
    TikZGraph(*list(graph)).make_bipartite()\
        .add_edges([TikZEdge.DIRECTED(edge) for edge in uot])\
        .write(f'uot{i}')
    # uot = graph.union_of_tours()
    # graph = TikZGraph(*list(graph)).make_bipartite()\
    #     .add_edges([TikZEdge.DIRECTED(edge) for edge in uot])\
    #     .add_edges([TikZEdge.GRAY(TikZEdge.DOTTED(edge)) for edge in graph.remove_directed()])\
    #     .add_edges([TikZOptions.BLUE(edge) for edge in graph.unfurl_uot(uot)])\
    #     .write(f'uot{i}')
    i+=1
    del uot
# exit()

cat_image = plt.imread('cat.png')

def reres(img, new_res=(30, 30)):
    h, w = new_res
    img = img[0:img.shape[0]-(img.shape[0]%h), 0:img.shape[1]-(img.shape[1]%w), :]
    newimg = np.zeros((h, w), int)
    for (i, r), (j, c) in product(enumerate(range(0, img.shape[0], img.shape[0]//h)), enumerate(range(0, img.shape[1], img.shape[1]//w))):
        newimg[i, j] = np.average(img[r:r+h, c:c+w, 3]) > 0.25
    return newimg

cat = reres(cat_image)
cat_graph = TikZGraph()
for r, c in product(*[range(0, s) for s in cat.shape]):
    if cat[r, c]:
        cat_graph.nodes[TikZNode(c, cat.shape[1]-r), assign]

for d in SolidGridGraph.directions:
    for node in cat_graph.nodes:
        if node+d in cat_graph.nodes:
            cat_graph[TikZEdge(Edge(node, node+d)), assign]

# cat_graph = TikZGraph(*cat_edges)
for node in cat_graph.nodes:
    node.style = TikZOptions({
        'scale': '0.5',
        'draw': None,
        'color': 'black',
        'fill': 'black',
        'circle': None
    })
for edge in cat_graph:
    edge.style['draw'] = 'none'
    edge.style['line width'] = '1mm'

cat_graph.add_edges(list(cat_graph))
cat_graph.write('cat')
del cat_image, cat_graph

example_graph_length = 2
example_graph = TikZGraph(((0, 0), (example_graph_length, 0)), ((example_graph_length, 0), (example_graph_length / 2, ((example_graph_length ** 2) - ((example_graph_length / 2) ** 2))**0.5 )), ((example_graph_length / 2, ((example_graph_length ** 2) - ((example_graph_length / 2) ** 2))**0.5), (0, 0)))
midpoint = sum(node[0] for node in example_graph.nodes)/3, sum(node[1] for node in example_graph.nodes)/3
for edge in list(example_graph):
    example_graph[TikZEdge(Edge(edge.t, midpoint)), assign]

example_graph.add_edges(
    [TikZEdge((example_graph_length / 2, ((example_graph_length ** 2) - ((example_graph_length / 2) ** 2))**0.5), (0, 0)), TikZEdge((example_graph_length, 0), midpoint)]
)
example_graph.add_edges([TikZOptions.GRAY(TikZEdge.DOTTED(edge)) for edge in example_graph])    
example_graph.write('graph')
del example_graph

# first example of a trail
trail_graph = TikZGraph(((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 0), (1, 1)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((1, 2), (1, 1)))
trail_graph.add_edges(
    [
        TikZEdge.DIRECTED(TikZEdge.SHORT(((0, 1), (1, 1))))
    ],
    [
        TikZEdge.DIRECTED(edge) for edge in TikZGraph.vertex_sequence_to_edges([(0, 1), (1, 1), (1, 2), (0, 2), (0, 1), (0, 0), (1, 0)])
    ]
)
trail_graph.write('trail')
del trail_graph

# trail_graph = Graph(edges=[((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 0), (1, 1)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((1, 2), (1, 1))])
# trail_graph.make_bipartite = lambda: ([], trail_graph.nodes, trail_graph.nodes)
# write_tex('trail', trail_graph, styles=[(['black', '->'], lambda _ : Graph.vertex_sequence_to_edges([(0, 1), (1, 1), (1, 2), (0, 2), (0, 1), (0, 0)])), (['black!30'], getEdges)])

# first example of a walk
# walk_graph = Graph(edges=[((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 0), (1, 1)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((1, 2), (1, 1))])
# walk_graph.make_bipartite = lambda: ([], walk_graph.nodes, walk_graph.nodes)
# write_tex('walk', walk_graph, styles=[(['black', '->'], lambda _ : list(walk_graph.src_sink_paths((0, 0), (0, 2) ))[3]), (['black!30'], getEdges)])

path_graph = TikZGraph(((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 0), (1, 1)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((1, 2), (1, 1))).undirected()
path_graph.add_edges(
    [
        TikZEdge.DIRECTED(edge) for edge in [((1, 0), (1, 1)), ((1, 1), (0, 1)), ((0, 1), (0, 2))]
    ],
    # [
    #     TikZEdge.GRAY(edge) for edge in path_graph
    # ]
)
path_graph.write('path')
del path_graph

# first example of a bipartite graph
solid_grid_graph = TikZGraph(((0, 0), (0, 1)), ((0, 1), (1, 1)), ((1, 1), (1, 0)), ((0, 0), (1, 0)), ((1, 1), (1, 2)), ((1, 2), (2, 2)), ((2, 2), (2, 1)), ((1, 1), (2, 1)), ((1, 2), (1, 3)), ((1, 3), (2, 3)), ((2, 3), (2, 2)), ((2, 1), (3, 1)), ((3, 1), (3, 2)), ((3, 2), (2, 2)), ((2, 3), (3, 3)), ((3, 3), (3, 2)), ((3, 4), (3, 3)), ((3, 4), (4, 4)), ((4, 4), (4, 3)), ((3, 3), (4, 3))).undirected()
solid_grid_graph.make_bipartite()
solid_grid_graph.add_edges(
    [
        TikZEdge(edge) for edge in solid_grid_graph.remove_directed()
    ]
)
solid_grid_graph.write('solid')
del solid_grid_graph

grid_graph = TikZGraph(((0, 0), (0, 1)), ((0, 1), (1, 1)), ((1, 1), (1, 0)), ((0, 0), (1, 0)), ((1, 1), (1, 2)), ((1, 2), (1, 3)), ((1, 3), (2, 3)), ((2, 3), (3, 3)), ((3, 3), (3, 2)), ((3, 2), (3, 1)), ((2, 1), (3, 1)), ((1, 1), (2, 1)), ((3, 4), (3, 3)), ((3, 4), (4, 4)), ((4, 4), (4, 3)), ((3, 3), (4, 3))).undirected()
grid_graph.make_bipartite()
grid_graph.add_edges(
    [
        TikZEdge(edge) for edge in grid_graph.remove_directed()
    ]
)
grid_graph.write('grid')
del grid_graph

# first example of a Hamiltonian path & Hamiltonian cycle
hamiltonian_graph = TikZGraph(((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 0), (1, 1)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((1, 2), (1, 1))).undirected()
hamiltonian_graph.make_bipartite()
hamiltonian_graph.add_edges(
    [
        TikZEdge.DIRECTED(edge) for edge in list(hamiltonian_graph.hamiltonian_paths((1, 0)))[-2]
    ],
    [
        TikZEdge.GRAY(TikZEdge.DOTTED(edge)) for edge in hamiltonian_graph.remove_directed()
    ]
)
hamiltonian_graph.write('hamiltonian_path')

hamiltonian_graph.edgesets[0] = [
    TikZEdge(edge) for edge in next(hamiltonian_graph.hamiltonian_cycles())
]
hamiltonian_graph.write('hamiltonian_cycle')
del hamiltonian_graph

# first example of the TST
travelling_salesman_tour_graph = TikZGraph(((0, 2), (1, 2)), ((1, 2), (2, 2)), ((2, 2), (2, 1)), ((2, 1), (1, 1)), ((0, 2), (0, 1)), ((0, 1), (1, 1)), ((1, 2), (1, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 1), (1, 0)), ((1, 0), (2, 0)), ((2, 1), (2, 0))).undirected()
travelling_salesman_tour_graph.make_bipartite()
travelling_salesman_tour_graph.add_edges(
    [
        TikZEdge(edge) for edge in next(SolidGridGraph(*travelling_salesman_tour_graph).travelling_salesman_tours())
    ],
    [
        TikZEdge.GRAY(TikZEdge.DOTTED(edge)) for edge in travelling_salesman_tour_graph.remove_directed()
    ]
)
travelling_salesman_tour_graph.write('travelling_salesman_tour')

# one_factor.tex
one_factor_graph = TikZGraph(((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 1), (1, 0)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((1, 2), (1, 1)), ((1, 2), (2, 2)), ((2, 2), (2, 1)), ((1, 1), (2, 1)), ((1, 0), (2, 0)), ((2, 0), (2, 1)), ((2, 2), (3, 2)), ((3, 2), (3, 1)), ((2, 1), (3, 1)), ((3, 1), (3, 0)), ((2, 0), (3, 0))).undirected()
one_factor_graph.make_bipartite()
one_factor_graph.add_edges(
    [
        TikZEdge(edge) for edge in [((0, 2), (1, 2)), ((2, 0), (3, 0)), ((2, 2), (2, 1)), ((3, 2), (3, 1)), ((0, 1), (0, 0)), ((1, 1), (1, 0))]
    ]
)
one_factor_graph.write('one_factor')
del one_factor_graph

# no_one_factor.tex
no_one_factor_graph = TikZGraph(*Graph.vertex_sequence_to_edges([(0, 1), (1.7293088488585, 2.0047342460869), (1.7347798055927, 0.0047417289428), (0, 1)]))
no_one_factor_graph.add_edges(
    [
        TikZEdge.GRAY(TikZEdge.DOTTED(edge)) for edge in no_one_factor_graph
    ]
)
no_one_factor_graph.write('no_one_factor')
del no_one_factor_graph

# two_factor.tex
two_factor_graph = TikZGraph(((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 1), (1, 0)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((1, 2), (1, 1)), ((1, 2), (2, 2)), ((2, 2), (2, 1)), ((1, 1), (2, 1)), ((1, 0), (2, 0)), ((2, 0), (2, 1)), ((2, 2), (3, 2)), ((3, 2), (3, 1)), ((2, 1), (3, 1)), ((3, 1), (3, 0)), ((2, 0), (3, 0))).undirected()
two_factor_graph.make_bipartite()
two_factor_graph.add_edges(
    [
        TikZEdge(edge) for edge in [((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 0), (1, 1)), ((1, 2), (1, 1)), ((0, 2), (1, 2)), ((2, 2), (2, 1)), ((2, 1), (2, 0)), ((2, 0), (3, 0)), ((3, 0), (3, 1)), ((3, 1), (3, 2)), ((2, 2), (3, 2))]
    ]
)
two_factor_graph.write('two_factor')
del two_factor_graph

# no_two_factor.tex
no_two_factor_graph = TikZGraph(*list(travelling_salesman_tour_graph.keys())).undirected()
no_two_factor_graph.make_bipartite()
no_two_factor_graph.add_edges(
    [
        TikZEdge.GRAY(TikZEdge.DOTTED(edge)) for edge in no_two_factor_graph.remove_directed()
    ]
)
no_two_factor_graph.write('no_two_factor')
del no_two_factor_graph, travelling_salesman_tour_graph

import os
import pickle
big_two_factor_graph = SolidGridGraph()
if not os.path.exists('big_two_factor.graph'):
    for _ in range(20):
        big_two_factor_graph.add_random_face(15, 15)

    while len(big_two_factor_graph.two_factor) == 0:
        delattr(big_two_factor_graph, '_two_factor')
        big_two_factor_graph.add_random_face(15, 15)
    big_tf = {
        'nodes': list(tuple(node) for node in big_two_factor_graph.nodes),
        'edges': list((tuple(edge.s), tuple(edge.t)) for edge in big_two_factor_graph),
        'twof': list((tuple(edge.s), tuple(edge.t)) for edge in big_two_factor_graph.two_factor)
    }
    pickle.dump(big_tf, open('big_two_factor.graph', 'wb'))
else:
    big_tf = pickle.load(open('big_two_factor.graph', 'rb'))
    for edge in big_tf['edges']:
        big_two_factor_graph[Edge(Node(edge[0][0], edge[0][1]), Node(edge[1][0], edge[1][1])), assign]
    big_two_factor_graph.two_factor = Graph(*big_tf['twof'])

big_two_factor_graph = big_two_factor_graph.undirected()

strips = big_two_factor_graph.get_alternating_strips()
static = big_two_factor_graph.static_alternating_strip()

shaded = shade_two_factor(big_two_factor_graph)
big_two_factor_graph = TikZGraph(*list(big_two_factor_graph)).undirected()
big_two_factor_graph.add_background(*shaded)

big_two_factor_graph.add_edges(
    [
        TikZOptions.GRAY(TikZEdge.DOTTED(TikZOptions.style('shorten <', '0', TikZOptions.style('shorten >', '0', TikZEdge(edge))))) for edge in big_two_factor_graph.two_factor.remove_directed()
    ]
)

for node in big_two_factor_graph.nodes:
    big_two_factor_graph.nodes[node].style["draw"] = 'none'
    big_two_factor_graph.nodes[node].style["fill"] = 'none'

big_two_factor_graph.write('big_two_factor')
del big_two_factor_graph


x1 = (0, 2)
x2 = (0, 1)
x3 = (0, 0)
y1 = (1, 2)
y2 = (1, 1)
y3 = (1, 0)

cubic_graph = TikZGraph((x1, y1), (x1, y3), (x2, y2), (x2, y3), (x3, y1), (x3, y2), (x3, y3)).undirected()
cubic_graph.make_bipartite()
cubic_graph.add_edges(
    [TikZOptions.COLOR('red', edge) for edge in cubic_graph.remove_directed() if edge.s == x2]
)

cubic_graph.add_edges(
    [TikZOptions.COLOR('blue', edge) for edge in cubic_graph.remove_directed() if edge.s == x3]
)

cubic_graph.add_edges(list(cubic_graph.undirected()))

cubic_graph.write('cubic_graph')
del cubic_graph

cubic_embedding = TikZGraph(((0, 4), (1, 4)), ((1, 4), (1, 3)), ((1, 3), (0, 3)), ((0, 4), (0, 3)), ((1, 4), (2, 4)), ((2, 4), (2, 3)), ((1, 3), (2, 3)), ((2, 3), (3, 3)), ((2, 4), (3, 4)), ((3, 4), (3, 3)), ((3, 4), (4, 4)), ((4, 4), (4, 3)), ((3, 3), (4, 3)), ((0, 3), (0, 2)), ((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((0, 1), (1, 1)), ((1, 1), (1, 0)), ((1, 2), (1, 1)), ((0, 2), (1, 2)), ((1, 3), (1, 2)), ((1, 2), (2, 2)), ((2, 2), (2, 3)), ((2, 1), (3, 1)), ((3, 2), (3, 1)), ((2, 2), (2, 1)), ((2, 2), (3, 2)), ((3, 2), (3, 3)), ((3, 2), (4, 2)), ((4, 2), (4, 1)), ((3, 1), (4, 1)), ((4, 2), (4, 3)), ((1, 1), (2, 1)), ((2, 1), (2, 0)), ((1, 0), (2, 0)), ((2, 0), (3, 0)), ((3, 0), (3, 1)), ((3, 0), (4, 0)), ((4, 0), (4, 1)))
cubic_embedding.make_bipartite()
cubic_embedding.add_edges(
    [TikZOptions.COLOR('red', TikZEdge(edge)) for edge in [((2, 2), (1, 2)), ((1, 2), (1, 3)), ((1, 3), (1, 4)), ((2, 2), (3, 2)), ((2, 2), (2, 1)), ((2, 1), (3, 1)), ((3, 1), (3, 0))]]
)

cubic_embedding.add_edges(
    [TikZOptions.COLOR('blue', TikZEdge(edge)) for edge in [((2, 4), (3, 4)), ((3, 4), (4, 4)), ((4, 4), (4, 3)), ((4, 3), (4, 2)), ((4, 2), (4, 1)), ((4, 1), (4, 0)), ((4, 0), (3, 0)), ((2, 4), (1, 4)), ((2, 4), (2, 3)), ((2, 3), (3, 3)), ((3, 3), (3, 2))]]
)

cubic_embedding.add_edges(
    [TikZEdge(edge) for edge in [((0, 0), (0, 1)), ((0, 1), (0, 2)), ((0, 2), (0, 3)), ((0, 3), (0, 4)), ((0, 4), (1, 4)), ((0, 0), (1, 0)), ((1, 0), (2, 0)), ((2, 0), (3, 0))]]
)

cubic_embedding.write('cubic_embedding')
del cubic_embedding

# type_3_before_flip.tex
type_3_before_flip = TikZGraph(((0, 1), (1, 1)), ((1, 1), (1, 0)), ((0, 0), (1, 0)), ((2, 1), (3, 1)), ((2, 1), (2, 0)), ((2, 0), (3, 0)), ((1, 1), (2, 1)), ((1, 0), (2, 0))).undirected()
type_3_before_flip.make_bipartite()
type_3_before_flip.add_edges(
    [
        TikZEdge(edge) for edge in [((0, 1), (1, 1)), ((1, 1), (1, 0)), ((0, 0), (1, 0)), ((2, 1), (3, 1)), ((2, 1), (2, 0)), ((2, 0), (3, 0))]
    ],
    [
        TikZEdge.GRAY(TikZEdge.DOTTED(edge)) for edge in type_3_before_flip.remove_directed()
    ]
)
type_3_before_flip.add_background("\\fill[fill=black!10] (-0.1, 0) rectangle ++(1.1, 1);", "\\fill[fill=black!10] (2, 0) rectangle ++(1.1, 1);")
type_3_before_flip.add_background("\\draw node at (1.5, 0.5) {\\textbf{III}};")
type_3_before_flip.write('type_3_before_flip')
del type_3_before_flip

type_3_after_flip = TikZGraph(((0, 1), (1, 1)), ((1, 1), (1, 0)), ((0, 0), (1, 0)), ((2, 1), (3, 1)), ((2, 1), (2, 0)), ((2, 0), (3, 0)), ((1, 1), (2, 1)), ((1, 0), (2, 0))).undirected()
type_3_after_flip.make_bipartite()
type_3_after_flip.add_edges(
    [
        TikZEdge(edge) for edge in [((0, 1), (1, 1)), ((1, 1), (2, 1)), ((2, 1), (3, 1)), ((0, 0), (1, 0)), ((1, 0), (2, 0)), ((2, 0), (3, 0))]
    ],
    [
        TikZEdge.GRAY(TikZEdge.DOTTED(edge)) for edge in type_3_after_flip.remove_directed()
    ]
)
type_3_after_flip.add_background("\\fill[fill=black!10] (-0.1, 0) rectangle ++(3.1, 1);")
type_3_after_flip.write('type_3_after_flip')
del type_3_after_flip

short_odd_alternating_strip = TikZGraph(((0, 1), (0, 0)), ((1, 1), (1, 0)))
short_odd_alternating_strip.add_edges(list(short_odd_alternating_strip))
for node in short_odd_alternating_strip.nodes:
    short_odd_alternating_strip.nodes[node].style["fill"] = "black"
short_odd_alternating_strip.add_background("\\useasboundingbox (0, 0) rectangle ++(7, 1);")
short_odd_alternating_strip.write('short_odd_alternating_strip')
del short_odd_alternating_strip

longer_odd_alternating_strip = TikZGraph(((0, 1), (0, 0)), ((1, 1), (2, 1)), ((2, 1), (2, 0)), ((2, 0), (1, 0)), ((3, 1), (3, 0)))
longer_odd_alternating_strip.add_edges(list(longer_odd_alternating_strip))
for node in longer_odd_alternating_strip.nodes:
    longer_odd_alternating_strip.nodes[node].style["fill"] = "black"
longer_odd_alternating_strip.add_background("\\useasboundingbox (0, 0) rectangle ++(7, 1);")
longer_odd_alternating_strip.write('longer_odd_alternating_strip')
del longer_odd_alternating_strip

general_odd_alternating_strip = TikZGraph(((5, -7), (5, -8)), ((6, -7), (7, -7)), ((7, -7), (7, -8)), ((6, -8), (7, -8)), ((10, -7), (11, -7)), ((11, -7), (11, -8)), ((10, -8), (11, -8)), ((12, -7), (12, -8)))
general_odd_alternating_strip.add_edges(list(general_odd_alternating_strip))
general_odd_alternating_strip.add_foreground("\draw node[scale=4.0] at (8.5, -7.5) {...};")
for node in general_odd_alternating_strip.nodes:
    general_odd_alternating_strip.nodes[node].style["fill"] = "black"
general_odd_alternating_strip.add_background("\\useasboundingbox (5, -8) rectangle ++(7, 1);")
general_odd_alternating_strip.write('general_odd_alternating_strip')
del general_odd_alternating_strip

short_even_alternating_strip = TikZGraph(((0, 1), (0, 0)), ((1, 1), (2, 1)), ((1, 0), (2, 0)))
short_even_alternating_strip.add_edges(list(short_even_alternating_strip))
for node in short_even_alternating_strip.nodes:
    short_even_alternating_strip.nodes[node].style["fill"] = "black"
short_even_alternating_strip.add_background("\\useasboundingbox (0, 0) rectangle ++(8, 1);")
short_even_alternating_strip.write('short_even_alternating_strip')
del short_even_alternating_strip

longer_even_alternating_strip = TikZGraph(((0, 1), (0, 0)), ((1, 1), (2, 1)), ((2, 1), (2, 0)), ((1, 0), (2, 0)), ((3, 1), (4, 1)), ((3, 0), (4, 0)))
longer_even_alternating_strip.add_edges(list(longer_even_alternating_strip))
for node in longer_even_alternating_strip.nodes:
    longer_even_alternating_strip.nodes[node].style["fill"] = "black"
longer_even_alternating_strip.add_background("\\useasboundingbox (0, 0) rectangle ++(8, 1);")
longer_even_alternating_strip.write('longer_even_alternating_strip')
del longer_even_alternating_strip

general_even_alternating_strip = TikZGraph(((0, 1), (0, 0)), ((1, 1), (2, 1)), ((2, 1), (2, 0)), ((1, 0), (2, 0)), ((5, 1), (6, 1)), ((6, 1), (6, 0)), ((6, 0), (5, 0)), ((7, 1), (8, 1)), ((7, 0), (8, 0)))
general_even_alternating_strip.add_edges(list(general_even_alternating_strip))
general_even_alternating_strip.add_foreground("\draw node[scale=4.0] at (3.5, 0.5) {...};")
for node in general_even_alternating_strip.nodes:
    general_even_alternating_strip.nodes[node].style["fill"] = "black"
general_even_alternating_strip.add_background("\\useasboundingbox (0, 0) rectangle ++(8, 1);")
general_even_alternating_strip.write('general_even_alternating_strip')
del general_even_alternating_strip

alternating_strip_before_flip = SolidGridGraph(((1, 0), (2, 0)), ((1, 1), (2, 1)), ((1, 2), (1, 1)), ((2, 2), (2, 1)), ((1, 4), (1, 3)), ((1, 3), (2, 3)), ((2, 4), (2, 3)), ((1, 6), (1, 5)), ((1, 5), (2, 5)), ((2, 6), (2, 5)), ((1, 7), (2, 7)), ((0, 0), (1, 0)), ((2, 0), (3, 0)), ((3, 0), (3, 1)), ((3, 1), (3, 2)), ((3, 2), (2, 2)), ((1, 2), (0, 2)), ((0, 2), (0, 1)), ((0, 1), (0, 0)), ((1, 4), (0, 4)), ((0, 4), (0, 5)), ((0, 5), (0, 6)), ((0, 6), (1, 6)), ((2, 6), (3, 6)), ((3, 6), (3, 5)), ((3, 5), (3, 4)), ((3, 4), (2, 4)), ((1, 7), (1, 8)), ((1, 8), (0, 8)), ((0, 8), (0, 9)), ((0, 9), (1, 9)), ((1, 9), (2, 9)), ((2, 9), (3, 9)), ((3, 9), (3, 8)), ((3, 8), (2, 8)), ((2, 8), (2, 7)), ((2, 7), (2, 6)), ((2, 6), (1, 6)), ((1, 6), (1, 7)), ((0, 5), (1, 5)), ((1, 5), (1, 4)), ((1, 4), (2, 4)), ((2, 5), (2, 4)), ((2, 5), (3, 5)), ((1, 8), (2, 8)), ((1, 9), (1, 8)), ((2, 9), (2, 8)), ((1, 3), (1, 2)), ((1, 2), (2, 2)), ((2, 2), (2, 3)), ((0, 1), (1, 1)), ((1, 1), (1, 0)), ((2, 0), (2, 1)), ((2, 1), (3, 1)))

alternating_strip_before_flip.two_factor = SolidGridGraph(((0, 0), (1, 0)), ((1, 0), (2, 0)), ((2, 0), (3, 0)), ((3, 0), (3, 1)), ((3, 1), (3, 2)), ((3, 2), (2, 2)), ((2, 2), (2, 1)), ((2, 1), (1, 1)), ((1, 1), (1, 2)), ((1, 2), (0, 2)), ((0, 2), (0, 1)), ((0, 1), (0, 0)), ((1, 3), (2, 3)), ((2, 3), (2, 4)), ((2, 4), (3, 4)), ((3, 4), (3, 5)), ((3, 5), (3, 6)), ((3, 6), (2, 6)), ((2, 6), (2, 5)), ((2, 5), (1, 5)), ((1, 5), (1, 6)), ((0, 6), (1, 6)), ((0, 6), (0, 5)), ((0, 5), (0, 4)), ((0, 4), (1, 4)), ((1, 4), (1, 3)), ((1, 7), (2, 7)), ((2, 7), (2, 8)), ((2, 8), (3, 8)), ((3, 8), (3, 9)), ((3, 9), (2, 9)), ((2, 9), (1, 9)), ((1, 9), (0, 9)), ((0, 9), (0, 8)), ((0, 8), (1, 8)), ((1, 8), (1, 7)))

longest_strip = max(SolidGridGraph.get_alternating_strips(alternating_strip_before_flip), key=len)
alternating_strip_after_flip = alternating_strip_before_flip.copy().edge_flip(longest_strip)

shaded = list(shade_two_factor(alternating_strip_before_flip))
alternating_strip_before_flip = TikZGraph(*alternating_strip_before_flip)
alternating_strip_before_flip.make_bipartite()
alternating_strip_before_flip.add_edges(
    [
        TikZEdge(edge) for edge in longest_strip.remove_directed()
    ],
    [
        TikZOptions.GRAY(TikZEdge.DOTTED(edge)) for edge in alternating_strip_before_flip.two_factor.remove_directed()
    ]
)
for node in alternating_strip_before_flip.nodes:
    if node not in longest_strip:
        node.style["draw"] = "none"
        node.style["fill"] = "none"
alternating_strip_before_flip.add_background(*shaded)
alternating_strip_before_flip.write('alternating_strip_before_flip')

longest_strip_flipped = longest_strip.flipped_perimeter + longest_strip - longest_strip.perimeter

shaded = list(shade_two_factor(alternating_strip_after_flip))
alternating_strip_after_flip = TikZGraph(*alternating_strip_after_flip)
alternating_strip_after_flip.two_factor = alternating_strip_before_flip.two_factor - longest_strip.perimeter + longest_strip.flipped_perimeter
alternating_strip_after_flip.make_bipartite()
alternating_strip_after_flip.add_edges(
    [
        TikZEdge(edge) for edge in longest_strip_flipped.remove_directed()
    ],
    [
        TikZOptions.GRAY(TikZEdge.DOTTED(edge)) for edge in alternating_strip_after_flip.two_factor.remove_directed()
    ]
)

for node in alternating_strip_after_flip.nodes:
    if node not in longest_strip_flipped:
        node.style["draw"] = "none"
        node.style["fill"] = "none"

alternating_strip_after_flip.add_background(*shaded)
alternating_strip_after_flip.write('alternating_strip_after_flip')
del alternating_strip_after_flip, alternating_strip_before_flip
    
# visualize_hamiltonian_cycle('tst', SolidGridGraph(*[Edge(edge.s, edge.t) for edge in big_two_factor_graph]))

# tst1 = TikZGraph(
#     ((0, 2), (0, 1)), ((0, 0), (1, 0)), ((0, 2), (1, 2)), ((1, 2), (1, 1)), ((0, 1), (1, 1)), ((0, 1), (0, 0)), ((1, 1), (1, 0)), ((1, 2), (2, 2)), ((2, 2), (2, 1)), ((1, 1), (2, 1)), ((1, 0), (2, 0)), ((2, 1), (2, 0))
# )
# tst1.add_edges(
#     [TikZEdge.DIRECTED(TikZEdge(edge.s, edge.t, d=True)) for edge in SolidGridGraph.tst(tst1)]
# )
# tst1.add_edges(
#     [TikZEdge.DOTTED(TikZEdge.GRAY(TikZEdge(*edge))) for edge in tst1]
# )
# tst1.write('tst1')


# tst2 = TikZGraph(
#     ((0, 2), (0, 1)), ((0, 0), (1, 0)), ((0, 2), (1, 2)), ((1, 2), (1, 1)), ((0, 1), (1, 1)), ((0, 1), (0, 0)), ((1, 1), (1, 0)), ((1, 2), (2, 2)), ((2, 2), (2, 1)), ((1, 1), (2, 1)), ((1, 0), (2, 0)), ((2, 1), (2, 0)), ((0, 2), (0, 3)), ((0, 3), (1, 3)), ((1, 3), (1, 2)), ((2, 3), (2, 2)), ((1, 3), (2, 3)), ((2, 3), (3, 3)), ((3, 3), (3, 2)), ((2, 2), (3, 2)), ((3, 2), (3, 1)), ((2, 1), (3, 1)), ((3, 1), (3, 0)), ((2, 0), (3, 0)), ((3, 3), (4, 3)), ((4, 3), (4, 2)), ((4, 2), (3, 2)), ((3, 1), (4, 1)), ((4, 2), (4, 1))
# )
# tst2.add_edges(
#     [TikZEdge.DIRECTED(TikZEdge(edge.s, edge.t, d=True)) for edge in SolidGridGraph.tst(tst2)]
# )
# tst2.add_edges(
#     [TikZEdge.DOTTED(TikZEdge.GRAY(TikZEdge(*edge))) for edge in tst2]
# )
# tst2.write('tst2')


# tst3 = TikZGraph(
#     ((0, 1), (1, 1)), ((1, 1), (1, 0)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 2), (1, 1)), ((1, 2), (2, 2)), ((2, 2), (2, 1)), ((1, 1), (2, 1)), ((2, 2), (3, 2)), ((3, 2), (3, 1)), ((2, 1), (3, 1)), ((2, 1), (2, 0)), ((2, 0), (3, 0)), ((3, 0), (3, 1)), ((3, 2), (4, 2)), ((4, 2), (4, 1)), ((3, 1), (4, 1)), ((4, 1), (4, 0)), ((3, 0), (4, 0)))
# tst3.add_edges(
#     [TikZEdge.DIRECTED(TikZEdge(edge.s, edge.t, d=True)) for edge in SolidGridGraph.tst(tst3)]
# )
# tst3.add_edges(
#     [TikZEdge.DOTTED(TikZEdge.GRAY(TikZEdge(*edge))) for edge in tst3]
# )
# tst3.write('tst3')
