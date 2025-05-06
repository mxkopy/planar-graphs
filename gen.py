from graph import *
from to_tex import *
from itertools import product
import matplotlib.pyplot as plt
import os
import numpy as np
from math import pi

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
            cat_graph[TikZEdge(node, node+d), assign]

# cat_graph = TikZGraph(*cat_edges)
for node in cat_graph.nodes:
    node.style = TiKZOptions({
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

# print(cat_image[round(cat_image.shape[0] / 2), round(cat_image.shape[1] / 2)])
# print(cat_image[0, 0])


example_graph_length = 2
example_graph = TikZGraph(((0, 0), (example_graph_length, 0)), ((example_graph_length, 0), (example_graph_length / 2, ((example_graph_length ** 2) - ((example_graph_length / 2) ** 2))**0.5 )), ((example_graph_length / 2, ((example_graph_length ** 2) - ((example_graph_length / 2) ** 2))**0.5), (0, 0)))
# example_graph.nodes[Node(example_graph_length* sin(pi/180.0), example_graph_length* cos(pi/180.0)), assign]
midpoint = sum(node[0] for node in example_graph.nodes)/3, sum(node[1] for node in example_graph.nodes)/3
for edge in list(example_graph):
    example_graph[TikZEdge(edge.t, midpoint), assign]
# example_graph[]

example_graph.add_edges(
    [TikZEdge((example_graph_length / 2, ((example_graph_length ** 2) - ((example_graph_length / 2) ** 2))**0.5), (0, 0)), TikZEdge((example_graph_length, 0), midpoint)]
)
example_graph.add_edges([TiKZOptions.GRAY(TikZEdge.DOTTED(edge)) for edge in example_graph])    
example_graph.write('graph')


# first example of a trail
trail_graph = TikZGraph(((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 0), (1, 1)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((1, 2), (1, 1)))
trail_graph.add_edges(
    [
        TikZEdge.DIRECTED(TikZEdge.SHORT(((0, 1), (1, 1))))
    ],
    [
        TikZEdge.DIRECTED(TikZEdge(*edge)) for edge in TikZGraph.vertex_sequence_to_edges([(0, 1), (1, 1), (1, 2), (0, 2), (0, 1), (0, 0), (1, 0)])
    ],
    # [
    #     TikZEdge.GRAY(edge) for edge in trail_graph
    # ]
)
trail_graph.write('trail')

# trail_graph = Graph(edges=[((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 0), (1, 1)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((1, 2), (1, 1))])
# trail_graph.make_bipartite = lambda: ([], trail_graph.nodes, trail_graph.nodes)
# write_tex('trail', trail_graph, styles=[(['black', '->'], lambda _ : Graph.vertex_sequence_to_edges([(0, 1), (1, 1), (1, 2), (0, 2), (0, 1), (0, 0)])), (['black!30'], getEdges)])

# first example of a walk
# walk_graph = Graph(edges=[((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 0), (1, 1)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((1, 2), (1, 1))])
# walk_graph.make_bipartite = lambda: ([], walk_graph.nodes, walk_graph.nodes)
# write_tex('walk', walk_graph, styles=[(['black', '->'], lambda _ : list(walk_graph.src_sink_paths((0, 0), (0, 2) ))[3]), (['black!30'], getEdges)])

path_graph = TikZGraph(((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 0), (1, 1)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((1, 2), (1, 1)))
path_graph.add_edges(
    [
        TikZEdge.DIRECTED(TikZEdge(*edge)) for edge in list(path_graph.src_sink_paths((0, 0), (0, 2)))[3]
    ],
    # [
    #     TikZEdge.GRAY(edge) for edge in path_graph
    # ]
)
path_graph.nodes[(1, 0)].draw['draw'] = 'none'
path_graph.nodes[(1, 0)].style['draw'] = 'none'
path_graph.write('path')

# first example of a bipartite graph
bipartite_graph = TikZGraph(((2, 3), (2, 2)), ((2, 2), (3, 2)), ((2, 3), (3, 3)), ((3, 3), (3, 2)), ((3, 2), (3, 1)), ((2, 1), (3, 1)), ((2, 2), (2, 1)), ((3, 1), (4, 1)), ((3, 2), (4, 2)), ((4, 2), (4, 1)), ((4, 3), (4, 2)), ((4, 3), (5, 3)), ((5, 3), (5, 2)), ((4, 2), (5, 2)), ((2, 2), (1, 2)), ((1, 2), (1, 1)), ((1, 1), (2, 1)), ((2, 1), (2, 0)), ((2, 0), (1, 0)), ((1, 0), (1, 1)), ((1, 2), (0, 2)), ((0, 2), (0, 1)), ((0, 1), (1, 1)))
bipartite_graph.make_bipartite()
bipartite_graph.add_edges(
    [
        TikZEdge(*edge) for edge in bipartite_graph
    ]
)
bipartite_graph.write('bipartite')

# first example of a Hamiltonian path & Hamiltonian cycle
hamiltonian_graph = TikZGraph(((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 0), (1, 1)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((1, 2), (1, 1)))
hamiltonian_graph.make_bipartite()
hamiltonian_graph.add_edges(
    [
        TikZEdge.DIRECTED(TikZEdge(*edge)) for edge in hamiltonian_graph.hamiltonian_paths((1, 0))[-2]
    ],
    [
        TikZEdge.GRAY(TikZEdge.DOTTED(TikZEdge(*edge))) for edge in hamiltonian_graph.keys()
    ]
)
hamiltonian_graph.write('hamiltonian_path')

hamiltonian_graph.edgesets[0] = [
    TikZEdge(*edge) for edge in hamiltonian_graph.hamiltonian_cycles()[0]
]
hamiltonian_graph.write('hamiltonian_cycle')

# first example of the TST
travelling_salesman_tour_graph = TikZGraph(((0, 2), (1, 2)), ((1, 2), (2, 2)), ((2, 2), (2, 1)), ((2, 1), (1, 1)), ((0, 2), (0, 1)), ((0, 1), (1, 1)), ((1, 2), (1, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 1), (1, 0)), ((1, 0), (2, 0)), ((2, 1), (2, 0)))
travelling_salesman_tour_graph.make_bipartite()
travelling_salesman_tour_graph.add_edges(
    [
        TikZEdge(*edge, d=True) for edge in travelling_salesman_tour_graph.travelling_salesman_tours()[0]
    ],
    [
        TikZEdge.GRAY(TikZEdge.DOTTED(edge)) for edge in travelling_salesman_tour_graph
    ]
)
travelling_salesman_tour_graph.write('travelling_salesman_tour')

# one_factor.tex
one_factor_graph = TikZGraph(((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 1), (1, 0)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((1, 2), (1, 1)), ((1, 2), (2, 2)), ((2, 2), (2, 1)), ((1, 1), (2, 1)), ((1, 0), (2, 0)), ((2, 0), (2, 1)), ((2, 2), (3, 2)), ((3, 2), (3, 1)), ((2, 1), (3, 1)), ((3, 1), (3, 0)), ((2, 0), (3, 0)))
one_factor_graph.make_bipartite()
one_factor_graph.add_edges(
    [
        TikZEdge(*edge) for edge in [((0, 2), (1, 2)), ((2, 0), (3, 0)), ((2, 2), (2, 1)), ((3, 2), (3, 1)), ((0, 1), (0, 0)), ((1, 1), (1, 0))]
    ]
)
one_factor_graph.write('one_factor')

# no_one_factor.tex
no_one_factor_graph = TikZGraph(*Graph.vertex_sequence_to_edges([(0, 1), (1.7293088488585, 2.0047342460869), (1.7347798055927, 0.0047417289428), (0, 1)]))
no_one_factor_graph.add_edges(
    [
        TikZEdge.GRAY(TikZEdge.DOTTED(edge)) for edge in no_one_factor_graph
    ]
)
no_one_factor_graph.write('no_one_factor')


# two_factor.tex
two_factor_graph = TikZGraph(((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 1), (1, 0)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((1, 2), (1, 1)), ((1, 2), (2, 2)), ((2, 2), (2, 1)), ((1, 1), (2, 1)), ((1, 0), (2, 0)), ((2, 0), (2, 1)), ((2, 2), (3, 2)), ((3, 2), (3, 1)), ((2, 1), (3, 1)), ((3, 1), (3, 0)), ((2, 0), (3, 0)))
two_factor_graph.make_bipartite()
two_factor_graph.add_edges(
    [
        TikZEdge(*edge) for edge in [((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 0), (1, 1)), ((1, 2), (1, 1)), ((0, 2), (1, 2)), ((2, 2), (2, 1)), ((2, 1), (2, 0)), ((2, 0), (3, 0)), ((3, 0), (3, 1)), ((3, 1), (3, 2)), ((2, 2), (3, 2))]
    ]
)
two_factor_graph.write('two_factor')


# no_two_factor.tex
no_two_factor_graph = TikZGraph(*list(travelling_salesman_tour_graph.keys()))
no_two_factor_graph.make_bipartite()
no_two_factor_graph.add_edges(
    [
        TikZEdge.GRAY(TikZEdge.DOTTED(edge)) for edge in no_two_factor_graph
    ]
)
no_two_factor_graph.write('no_two_factor')

def shade_faces(graph, faces, style='fill=black!10'):
    for face in faces:
        x, y = sum((edge.s for edge in face), start=Node(0, 0)) * 0.25
        graph.add_background(f"{chr(92)}fill[{style}] ({x-0.5}cm-1pt, {y-0.5}cm-1pt) rectangle ++(1cm+2pt, 1cm+2pt);")

def shade_two_factor(graph, style='fill=black!10'):
    faces = [face for face in SolidGridGraph(*[(edge.s, edge.t) for edge in graph]).faces if graph.two_factor.test_interior(sum((edge.s for edge in face), start=Node(0, 0)) * 0.25)]
    shade_faces(graph, faces, style=style)

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
    # for node in big_tf['nodes']:
    #     big_two_factor_graph[Node(node[0], node[1]), assign]
    for edge in big_tf['edges']:
        big_two_factor_graph[Edge(Node(edge[0][0], edge[0][1]), Node(edge[1][0], edge[1][1])), assign]
    big_two_factor_graph.two_factor = EdgeSet(*big_tf['twof'])

# big_two_factor_graph = SolidGridGraph(*[Edge(edge.s, edge.t) for edge in two_factor_graph])
# big_two_factor_graph = SolidGridGraph()
# big_two_factor_graph.two_factor = EdgeSet(((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 0), (1, 1)), ((1, 2), (1, 1)), ((0, 2), (1, 2)), ((2, 2), (2, 1)), ((2, 1), (2, 0)), ((2, 0), (3, 0)), ((3, 0), (3, 1)), ((3, 1), (3, 2)), ((2, 2), (3, 2)))

strips = big_two_factor_graph.get_alternating_strips()
static = big_two_factor_graph.static_alternating_strip()
# big_two_factor_graph = big_two_factor_graph.reduce_two_factor()


# dual = big_two_factor_graph.get_dual()
big_two_factor_graph = TikZGraph(*list(big_two_factor_graph))
# big_two_factor_graph.add_edges([TikZEdge(*edge) for edge in static])
# big_two_factor_graph.two_factor = EdgeSet(((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 0), (1, 1)), ((1, 2), (1, 1)), ((0, 2), (1, 2)), ((2, 2), (2, 1)), ((2, 1), (2, 0)), ((2, 0), (3, 0)), ((3, 0), (3, 1)), ((3, 1), (3, 2)), ((2, 2), (3, 2)))
# big_two_factor_graph.make_bipartite()

# for strip in statics:
#     if hasattr(strip, 'c'):
#         big_two_factor_graph.add_edges([TikZEdge(edge.s, edge.t) for edge in strip])
#         big_two_factor_graph.add_foreground(f"\draw node[circle, draw] at ({strip.c[0]}, {strip.c[1]}) {{c}};")
#         big_two_factor_graph.add_foreground(f"\draw node[circle, blue, draw] at ({strip.c1[0]}, {strip.c1[1]}) {{c1}};")
#         big_two_factor_graph.add_foreground(f"\draw node[circle, blue, draw] at ({strip.c2[0]}, {strip.c2[1]}) {{c2}};")

# big_two_factor_graph.add_edges(
#     TiKZOptions.BLUE(TikZEdge(*edge)) for edge in dual if edge.s.interior and edge.t.interior
# )
# for strip in statics:
#     big_two_factor_graph.add_edges([TikZEdge(edge.s, edge.t) for edge in strip])
    # if not strip.chain:
    # big_two_factor_graph.add_foreground(f"\draw node[circle, red, draw] at ({strip.start[0]}, {strip.start[1]}){{{'begin' if not strip.chain else 'chain'}}};")
    # big_two_factor_graph.add_foreground(f"\draw node[circle, draw] at ({strip.end[0]}, {strip.end[1]}){{{'odd' if strip.odd else 'even'}}};")


big_two_factor_graph.add_edges(
    [
        TiKZOptions.GRAY(TikZEdge.DOTTED(TikZEdge(edge.s, edge.t, style={'shorten <': '0', 'shorten >': '0'}))) for edge in big_two_factor_graph.two_factor
    ]
)

# big_two_factor_graph.add_edges(
#     [
#         TiKZOptions.GRAY(TikZEdge.DOTTED(TikZEdge(edge.s, edge.t, style={'shorten <': '0', 'shorten >': '0'}))) for edge in big_two_factor_graph
#     ]
# )


for node in big_two_factor_graph.nodes:
    big_two_factor_graph.nodes[node].style["draw"] = 'none'
    big_two_factor_graph.nodes[node].style["fill"] = 'none'


shade_two_factor(big_two_factor_graph)
big_two_factor_graph.write('big_two_factor')

del big_two_factor_graph


x1 = (0, 2)
x2 = (0, 1)
x3 = (0, 0)
y1 = (1, 2)
y2 = (1, 1)
y3 = (1, 0)

cubic_graph = TikZGraph((x1, y1), (x1, y3), (x2, y2), (x2, y3), (x3, y1), (x3, y2), (x3, y3))
cubic_graph.make_bipartite()
cubic_graph.add_edges(
    [TiKZOptions.COLOR('red', edge) for edge in cubic_graph if edge.s == x2]
)

cubic_graph.add_edges(
    [TiKZOptions.COLOR('blue', edge) for edge in cubic_graph if edge.s == x3]
)

cubic_graph.add_edges(list(cubic_graph))

cubic_graph.write('cubic_graph')

cubic_embedding = TikZGraph(((0, 4), (1, 4)), ((1, 4), (1, 3)), ((1, 3), (0, 3)), ((0, 4), (0, 3)), ((1, 4), (2, 4)), ((2, 4), (2, 3)), ((1, 3), (2, 3)), ((2, 3), (3, 3)), ((2, 4), (3, 4)), ((3, 4), (3, 3)), ((3, 4), (4, 4)), ((4, 4), (4, 3)), ((3, 3), (4, 3)), ((0, 3), (0, 2)), ((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((0, 1), (1, 1)), ((1, 1), (1, 0)), ((1, 2), (1, 1)), ((0, 2), (1, 2)), ((1, 3), (1, 2)), ((1, 2), (2, 2)), ((2, 2), (2, 3)), ((2, 1), (3, 1)), ((3, 2), (3, 1)), ((2, 2), (2, 1)), ((2, 2), (3, 2)), ((3, 2), (3, 3)), ((3, 2), (4, 2)), ((4, 2), (4, 1)), ((3, 1), (4, 1)), ((4, 2), (4, 3)), ((1, 1), (2, 1)), ((2, 1), (2, 0)), ((1, 0), (2, 0)), ((2, 0), (3, 0)), ((3, 0), (3, 1)), ((3, 0), (4, 0)), ((4, 0), (4, 1)))
cubic_embedding.make_bipartite()
cubic_embedding.add_edges(
    [TiKZOptions.COLOR('red', TikZEdge(*edge)) for edge in [((2, 2), (1, 2)), ((1, 2), (1, 3)), ((1, 3), (1, 4)), ((2, 2), (3, 2)), ((2, 2), (2, 1)), ((2, 1), (3, 1)), ((3, 1), (3, 0))]]
)

cubic_embedding.add_edges(
    [TiKZOptions.COLOR('blue', TikZEdge(*edge)) for edge in [((2, 4), (3, 4)), ((3, 4), (4, 4)), ((4, 4), (4, 3)), ((4, 3), (4, 2)), ((4, 2), (4, 1)), ((4, 1), (4, 0)), ((4, 0), (3, 0)), ((2, 4), (1, 4)), ((2, 4), (2, 3)), ((2, 3), (3, 3)), ((3, 3), (3, 2))]]
)

cubic_embedding.add_edges(
    [TikZEdge(*edge) for edge in [((0, 0), (0, 1)), ((0, 1), (0, 2)), ((0, 2), (0, 3)), ((0, 3), (0, 4)), ((0, 4), (1, 4)), ((0, 0), (1, 0)), ((1, 0), (2, 0)), ((2, 0), (3, 0))]]
)

cubic_embedding.write('cubic_embedding')

# type_3_before_flip.tex
type_3_before_flip = TikZGraph(((0, 1), (1, 1)), ((1, 1), (1, 0)), ((0, 0), (1, 0)), ((2, 1), (3, 1)), ((2, 1), (2, 0)), ((2, 0), (3, 0)), ((1, 1), (2, 1)), ((1, 0), (2, 0)))
type_3_before_flip.make_bipartite()
type_3_before_flip.add_edges(
    [
        TikZEdge(*edge) for edge in [((0, 1), (1, 1)), ((1, 1), (1, 0)), ((0, 0), (1, 0)), ((2, 1), (3, 1)), ((2, 1), (2, 0)), ((2, 0), (3, 0))]
    ],
    [
        TikZEdge.GRAY(TikZEdge.DOTTED(edge)) for edge in type_3_before_flip
    ]
)
type_3_before_flip.add_background("\\fill[fill=black!10] (-0.1, 0) rectangle ++(1.1, 1);", "\\fill[fill=black!10] (2, 0) rectangle ++(1.1, 1);")
type_3_before_flip.add_background("\\draw node at (1.5, 0.5) {\\textbf{III}};")
type_3_before_flip.write('type_3_before_flip')

type_3_after_flip = TikZGraph(((0, 1), (1, 1)), ((1, 1), (1, 0)), ((0, 0), (1, 0)), ((2, 1), (3, 1)), ((2, 1), (2, 0)), ((2, 0), (3, 0)), ((1, 1), (2, 1)), ((1, 0), (2, 0)))
type_3_after_flip.make_bipartite()
type_3_after_flip.add_edges(
    [
        TikZEdge(*edge) for edge in [((0, 1), (1, 1)), ((1, 1), (2, 1)), ((2, 1), (3, 1)), ((0, 0), (1, 0)), ((1, 0), (2, 0)), ((2, 0), (3, 0))]
    ],
    [
        TikZEdge.GRAY(TikZEdge.DOTTED(edge)) for edge in type_3_after_flip
    ]
)
type_3_after_flip.add_background("\\fill[fill=black!10] (-0.1, 0) rectangle ++(3.1, 1);")
type_3_after_flip.write('type_3_after_flip')

short_odd_alternating_strip = TikZGraph(((0, 1), (0, 0)), ((1, 1), (1, 0)))
short_odd_alternating_strip.add_edges(list(short_odd_alternating_strip))
for node in short_odd_alternating_strip.nodes:
    short_odd_alternating_strip.nodes[node].style["fill"] = "black"
short_odd_alternating_strip.add_background("\\useasboundingbox (0, 0) rectangle ++(7, 1);")
short_odd_alternating_strip.write('short_odd_alternating_strip')

longer_odd_alternating_strip = TikZGraph(((0, 1), (0, 0)), ((1, 1), (2, 1)), ((2, 1), (2, 0)), ((2, 0), (1, 0)), ((3, 1), (3, 0)))
longer_odd_alternating_strip.add_edges(list(longer_odd_alternating_strip))
for node in longer_odd_alternating_strip.nodes:
    longer_odd_alternating_strip.nodes[node].style["fill"] = "black"
longer_odd_alternating_strip.add_background("\\useasboundingbox (0, 0) rectangle ++(7, 1);")
longer_odd_alternating_strip.write('longer_odd_alternating_strip')

general_odd_alternating_strip = TikZGraph(((5, -7), (5, -8)), ((6, -7), (7, -7)), ((7, -7), (7, -8)), ((6, -8), (7, -8)), ((10, -7), (11, -7)), ((11, -7), (11, -8)), ((10, -8), (11, -8)), ((12, -7), (12, -8)))
general_odd_alternating_strip.add_edges(list(general_odd_alternating_strip))
general_odd_alternating_strip.add_foreground("\draw node[scale=4.0] at (8.5, -7.5) {...};")
for node in general_odd_alternating_strip.nodes:
    general_odd_alternating_strip.nodes[node].style["fill"] = "black"
general_odd_alternating_strip.add_background("\\useasboundingbox (5, -8) rectangle ++(7, 1);")
general_odd_alternating_strip.write('general_odd_alternating_strip')


short_even_alternating_strip = TikZGraph(((0, 1), (0, 0)), ((1, 1), (2, 1)), ((1, 0), (2, 0)))
short_even_alternating_strip.add_edges(list(short_even_alternating_strip))
for node in short_even_alternating_strip.nodes:
    short_even_alternating_strip.nodes[node].style["fill"] = "black"
short_even_alternating_strip.add_background("\\useasboundingbox (0, 0) rectangle ++(8, 1);")
short_even_alternating_strip.write('short_even_alternating_strip')

longer_even_alternating_strip = TikZGraph(((0, 1), (0, 0)), ((1, 1), (2, 1)), ((2, 1), (2, 0)), ((1, 0), (2, 0)), ((3, 1), (4, 1)), ((3, 0), (4, 0)))
longer_even_alternating_strip.add_edges(list(longer_even_alternating_strip))
for node in longer_even_alternating_strip.nodes:
    longer_even_alternating_strip.nodes[node].style["fill"] = "black"
longer_even_alternating_strip.add_background("\\useasboundingbox (0, 0) rectangle ++(8, 1);")
longer_even_alternating_strip.write('longer_even_alternating_strip')

general_even_alternating_strip = TikZGraph(((0, 1), (0, 0)), ((1, 1), (2, 1)), ((2, 1), (2, 0)), ((1, 0), (2, 0)), ((5, 1), (6, 1)), ((6, 1), (6, 0)), ((6, 0), (5, 0)), ((7, 1), (8, 1)), ((7, 0), (8, 0)))
general_even_alternating_strip.add_edges(list(general_even_alternating_strip))
general_even_alternating_strip.add_foreground("\draw node[scale=4.0] at (3.5, 0.5) {...};")
for node in general_even_alternating_strip.nodes:
    general_even_alternating_strip.nodes[node].style["fill"] = "black"
general_even_alternating_strip.add_background("\\useasboundingbox (0, 0) rectangle ++(8, 1);")
general_even_alternating_strip.write('general_even_alternating_strip')

# odd_alternating_strip = SolidGridGraph(((0, 1), (0, 0)), ((0, 1), (1, 1)), ((0, 0), (1, 0)), ((1, 0), (1, 1)), ((1, 1), (2, 1)), ((2, 1), (2, 0)), ((1, 0), (2, 0)), ((2, 0), (3, 0)), ((2, 1), (3, 1)), ((3, 1), (3, 0)), ((3, 0), (4, 0)), ((3, 1), (4, 1)), ((4, 1), (4, 0)), ((4, 0), (5, 0)), ((4, 1), (5, 1)), ((5, 1), (5, 0)), ((5, 0), (6, 0)), ((5, 1), (6, 1)), ((6, 1), (6, 0)), ((6, 0), (7, 0)), ((6, 1), (7, 1)), ((7, 1), (7, 0)))
# odd_alternating_strip.longest_strip = odd_alternating_strip.get_alternating_strip(Edge((0, 0), (0, 1)))
# odd_alternating_strip = TikZGraph(*odd_alternating_strip.longest_strip)
# odd_alternating_strip.add_edges(list(odd_alternating_strip))
# for node in odd_alternating_strip.nodes:
#     odd_alternating_strip.nodes[node].style["fill"] = "black"
# odd_alternating_strip.write('odd_alternating_strip')


alternating_strip_before_flip = TikZGraph(((1, 0), (2, 0)), ((1, 1), (2, 1)), ((1, 2), (1, 1)), ((2, 2), (2, 1)), ((1, 4), (1, 3)), ((1, 3), (2, 3)), ((2, 4), (2, 3)), ((1, 6), (1, 5)), ((1, 5), (2, 5)), ((2, 6), (2, 5)), ((1, 7), (2, 7)), ((0, 0), (1, 0)), ((2, 0), (3, 0)), ((3, 0), (3, 1)), ((3, 1), (3, 2)), ((3, 2), (2, 2)), ((1, 2), (0, 2)), ((0, 2), (0, 1)), ((0, 1), (0, 0)), ((1, 4), (0, 4)), ((0, 4), (0, 5)), ((0, 5), (0, 6)), ((0, 6), (1, 6)), ((2, 6), (3, 6)), ((3, 6), (3, 5)), ((3, 5), (3, 4)), ((3, 4), (2, 4)), ((1, 7), (1, 8)), ((1, 8), (0, 8)), ((0, 8), (0, 9)), ((0, 9), (1, 9)), ((1, 9), (2, 9)), ((2, 9), (3, 9)), ((3, 9), (3, 8)), ((3, 8), (2, 8)), ((2, 8), (2, 7)), ((2, 7), (2, 6)), ((2, 6), (1, 6)), ((1, 6), (1, 7)), ((0, 5), (1, 5)), ((1, 5), (1, 4)), ((1, 4), (2, 4)), ((2, 5), (2, 4)), ((2, 5), (3, 5)), ((1, 8), (2, 8)), ((1, 9), (1, 8)), ((2, 9), (2, 8)), ((1, 3), (1, 2)), ((1, 2), (2, 2)), ((2, 2), (2, 3)), ((0, 1), (1, 1)), ((1, 1), (1, 0)), ((2, 0), (2, 1)), ((2, 1), (3, 1)))
alternating_strip_before_flip.make_bipartite()
# alternating_strip_before_flip.faces = 

alternating_strip_before_flip.two_factors = [EdgeSet(*two_factor) for two_factor in [
    [((0, 0), (1, 0)), ((1, 0), (2, 0)), ((2, 0), (3, 0)), ((3, 0), (3, 1)), ((3, 1), (3, 2)), ((3, 2), (2, 2)), ((2, 2), (2, 1)), ((2, 1), (1, 1)), ((1, 1), (1, 2)), ((1, 2), (0, 2)), ((0, 2), (0, 1)), ((0, 1), (0, 0))],
    [((1, 3), (2, 3)), ((2, 3), (2, 4)), ((2, 4), (3, 4)), ((3, 4), (3, 5)), ((3, 5), (3, 6)), ((3, 6), (2, 6)), ((2, 6), (2, 5)), ((2, 5), (1, 5)), ((1, 5), (1, 6)), ((0, 6), (1, 6)), ((0, 6), (0, 5)), ((0, 5), (0, 4)), ((0, 4), (1, 4)), ((1, 4), (1, 3))],
    [((1, 7), (2, 7)), ((2, 7), (2, 8)), ((2, 8), (3, 8)), ((3, 8), (3, 9)), ((3, 9), (2, 9)), ((2, 9), (1, 9)), ((1, 9), (0, 9)), ((0, 9), (0, 8)), ((0, 8), (1, 8)), ((1, 8), (1, 7))]
]]
alternating_strip_before_flip.two_factor = sum(alternating_strip_before_flip.two_factors, start=EdgeSet())
alternating_strip_before_flip.alternating_strips = SolidGridGraph.get_alternating_strips(alternating_strip_before_flip)
longest_strip = max(alternating_strip_before_flip.alternating_strips, key=len)

perimeter = SolidGridGraph.get_perimeter_of_alternating_strip(longest_strip)

alternating_strip_after_flip = SolidGridGraph.edge_flip(alternating_strip_before_flip, longest_strip)

# for strip in alternating_strip_before_flip.alternating_strips:
#     alternating_strip_before_flip.add_edges(
#         [
#             TikZEdge(*edge) for edge in strip
#         ]
#     )
    # alternating_strip_before_flip.add_foreground(f"\draw node[draw, circle, black] at ({strip.start[0]}, {strip.start[1]}){{}};")
    # alternating_strip_before_flip.add_foreground(f"\draw node[draw, circle, red] at ({strip.end[0]}, {strip.end[1]}){{}};")


alternating_strip_before_flip.add_edges(
    [
        TikZEdge(*edge) for edge in longest_strip
    ],
    [
        TiKZOptions.GRAY(TikZEdge.DOTTED(TikZEdge(edge.s, edge.t))) for edge in alternating_strip_before_flip if edge in alternating_strip_before_flip.two_factor
    ]
)
for node in alternating_strip_before_flip.nodes:
    if node not in longest_strip:
        node.style["draw"] = "none"
        node.style["fill"] = "none"

dual = SolidGridGraph(*[(edge.s, edge.t) for edge in alternating_strip_before_flip]).get_dual()

shade_two_factor(alternating_strip_before_flip)
alternating_strip_before_flip.write('alternating_strip_before_flip')

longest_strip_flipped = SolidGridGraph.flip_perimeter_of_alternating_strip(perimeter) + longest_strip - perimeter
alternating_strip_after_flip.add_edges(
    [
        TikZEdge(*edge) for edge in longest_strip_flipped
    ],
    [
        TiKZOptions.GRAY(TikZEdge.DOTTED(TikZEdge(edge.s, edge.t))) for edge in alternating_strip_after_flip if edge in alternating_strip_after_flip.two_factor
    ]
)

for node in alternating_strip_after_flip.nodes:
    if node not in longest_strip_flipped:
        node.style["draw"] = "none"
        node.style["fill"] = "none"


shade_two_factor(alternating_strip_after_flip)
alternating_strip_after_flip.write('alternating_strip_after_flip')



tst1 = TikZGraph(
    ((0, 2), (0, 1)), ((0, 0), (1, 0)), ((0, 2), (1, 2)), ((1, 2), (1, 1)), ((0, 1), (1, 1)), ((0, 1), (0, 0)), ((1, 1), (1, 0)), ((1, 2), (2, 2)), ((2, 2), (2, 1)), ((1, 1), (2, 1)), ((1, 0), (2, 0)), ((2, 1), (2, 0))
)
tst1.add_edges(
    [TikZEdge.DIRECTED(TikZEdge(edge.s, edge.t, d=True)) for edge in SolidGridGraph.tst(tst1)]
)
tst1.add_edges(
    [TikZEdge.DOTTED(TikZEdge.GRAY(TikZEdge(*edge))) for edge in tst1]
)
tst1.write('tst1')


tst2 = TikZGraph(
    ((0, 2), (0, 1)), ((0, 0), (1, 0)), ((0, 2), (1, 2)), ((1, 2), (1, 1)), ((0, 1), (1, 1)), ((0, 1), (0, 0)), ((1, 1), (1, 0)), ((1, 2), (2, 2)), ((2, 2), (2, 1)), ((1, 1), (2, 1)), ((1, 0), (2, 0)), ((2, 1), (2, 0)), ((0, 2), (0, 3)), ((0, 3), (1, 3)), ((1, 3), (1, 2)), ((2, 3), (2, 2)), ((1, 3), (2, 3)), ((2, 3), (3, 3)), ((3, 3), (3, 2)), ((2, 2), (3, 2)), ((3, 2), (3, 1)), ((2, 1), (3, 1)), ((3, 1), (3, 0)), ((2, 0), (3, 0)), ((3, 3), (4, 3)), ((4, 3), (4, 2)), ((4, 2), (3, 2)), ((3, 1), (4, 1)), ((4, 2), (4, 1))
)
tst2.add_edges(
    [TikZEdge.DIRECTED(TikZEdge(edge.s, edge.t, d=True)) for edge in SolidGridGraph.tst(tst2)]
)
tst2.add_edges(
    [TikZEdge.DOTTED(TikZEdge.GRAY(TikZEdge(*edge))) for edge in tst2]
)
tst2.write('tst2')


tst3 = TikZGraph(
    ((0, 1), (1, 1)), ((1, 1), (1, 0)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 2), (1, 1)), ((1, 2), (2, 2)), ((2, 2), (2, 1)), ((1, 1), (2, 1)), ((2, 2), (3, 2)), ((3, 2), (3, 1)), ((2, 1), (3, 1)), ((2, 1), (2, 0)), ((2, 0), (3, 0)), ((3, 0), (3, 1)), ((3, 2), (4, 2)), ((4, 2), (4, 1)), ((3, 1), (4, 1)), ((4, 1), (4, 0)), ((3, 0), (4, 0)))
tst3.add_edges(
    [TikZEdge.DIRECTED(TikZEdge(edge.s, edge.t, d=True)) for edge in SolidGridGraph.tst(tst3)]
)
tst3.add_edges(
    [TikZEdge.DOTTED(TikZEdge.GRAY(TikZEdge(*edge))) for edge in tst3]
)
tst3.write('tst3')


# polygon = TikZGraph(edges=Graph.vertex_sequence_to_edges([(0.4811533788947,1.8225175674737), (1.4446055899665,1.9117261055359), (1.7568354731842,1.4032374385813), (1.0877714377177,1.0553241401387), (0.37410313322,1.2515829238756), (0.4811533788947,1.8225175674737)]))
# for node in polygon.nodes:
#     node.style = TiKZOptions({})
# for edge in polygon:
#     print(edge.s.style)
# polygon.add_edges(list(polygon.keys()))
# for edge in polygon.edgesets[0]:
#     print(edge.s.style)
# polygon.write('polygon')






# tours = getTravellingSalesmanTour(travelling_salesman_tour_graph)
# write_tex('travelling_salesman_tour', travelling_salesman_tour_graph, styles=[(['black'], getTravellingSalesmanTour), (['black!30', 'dotted'], getEdges)])


# hamiltonian_graph = Graph(edges=[((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 0), (1, 1)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((1, 2), (1, 1))])
# # hamiltonian_graph.make_bipartite = lambda: ([], hamiltonian_graph.nodes, hamiltonian_graph.nodes)
# write_tex('hamiltonian_path', hamiltonian_graph, styles=[(['black', '->'], lambda _: hamiltonian_graph.hamiltonian_paths((1, 0))[-2]), (['black!30', 'dotted'], getEdges)])
# write_tex('hamiltonian_cycle', hamiltonian_graph, styles=[(['black'], getHamiltonianCycle), (['black!30', 'dotted'], getEdges)])
# write_tex('edge_cover', hamiltonian_graph, styles=[(['black'], lambda _: [Edge((0, 0), (1, 0)), Edge((0, 1), (1, 1)), Edge((0, 2), (1, 2))]), (['black!30', 'dotted'], getEdges)])


# two_factor_cover_graph = Graph(edges=[((0, 2), (0, 1)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((1, 2), (1, 1)), ((1, 1), (1, 0)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 0), (2, 0)), ((2, 0), (3, 0)), ((3, 1), (3, 0)), ((2, 1), (2, 0)), ((2, 1), (3, 1)), ((2, 2), (2, 1)), ((2, 2), (3, 2)), ((3, 2), (3, 1)), ((1, 1), (2, 1)), ((1, 2), (2, 2))])
# two_factor_cover_graph.make_bipartite = lambda: ([], two_factor_cover_graph.nodes, two_factor_cover_graph.nodes)
# write_tex('two_factor_cover', two_factor_cover_graph, styles=[(['black'], getTwoFactor), (['black!30', 'dotted'], getEdges)])

# first example of a travelling salesman tour 
# travelling_salesman_tour_graph = Graph(edges=[((0, 2), (1, 2)), ((1, 2), (2, 2)), ((2, 2), (2, 1)), ((2, 1), (1, 1)), ((0, 2), (0, 1)), ((0, 1), (1, 1)), ((1, 2), (1, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 1), (1, 0)), ((1, 0), (2, 0)), ((2, 1), (2, 0))])
# travelling_salesman_tour_graph.make_bipartite = lambda: ([], travelling_salesman_tour_graph.nodes, travelling_salesman_tour_graph.nodes)
# tours = getTravellingSalesmanTour(travelling_salesman_tour_graph)
# write_tex('travelling_salesman_tour', travelling_salesman_tour_graph, styles=[(['black'], getTravellingSalesmanTour), (['black!30', 'dotted'], getEdges)])


# interior_exterior_graph = Graph(edges=[((0, 2), (0, 1)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((1, 2), (1, 1)), ((1, 1), (2, 1)), ((1, 2), (2, 2)), ((2, 2), (2, 1)), ((2, 1), (3, 1)), ((2, 2), (3, 2)), ((3, 2), (3, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 1), (1, 0)), ((1, 0), (2, 0)), ((2, 1), (2, 0)), ((2, 0), (3, 0)), ((3, 1), (3, 0)), ((0, 2), (0, 3)), ((0, 3), (1, 3)), ((1, 3), (1, 2)), ((1, 3), (2, 3)), ((2, 3), (2, 2)), ((2, 3), (3, 3)), ((3, 3), (3, 2))])
# print(Edge((1, 1), (1, 0), d=True).axis())