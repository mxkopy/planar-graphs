from graph import *
from to_tex import *


example_graph_length = 2
example_graph = TeXGraph(edges=[((0, 0), (example_graph_length, 0)), ((example_graph_length, 0), (example_graph_length / 2, ((example_graph_length ** 2) - ((example_graph_length / 2) ** 2))**0.5 )), ((example_graph_length / 2, ((example_graph_length ** 2) - ((example_graph_length / 2) ** 2))**0.5), (0, 0))])
example_graph.add_edges(list(example_graph.keys()))
example_graph.write('graph')


# first example of a trail
trail_graph = TeXGraph(edges=[((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 0), (1, 1)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((1, 2), (1, 1))])
trail_graph.add_edges(
    [
        TeXGraph.DIRECTED(TeXGraph.SHORT(TikZEdge((0, 1), (1, 1))))
    ],
    [
        TeXGraph.DIRECTED(TikZEdge(*edge)) for edge in TeXGraph.vertex_sequence_to_edges([(0, 1), (1, 1), (1, 2), (0, 2), (0, 1), (0, 0)])
    ],
    [
        TeXGraph.GRAY(edge) for edge in trail_graph
    ]
)
trail_graph.write('trail')

# trail_graph = Graph(edges=[((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 0), (1, 1)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((1, 2), (1, 1))])
# trail_graph.make_bipartite = lambda: ([], trail_graph.nodes, trail_graph.nodes)
# write_tex('trail', trail_graph, styles=[(['black', '->'], lambda _ : Graph.vertex_sequence_to_edges([(0, 1), (1, 1), (1, 2), (0, 2), (0, 1), (0, 0)])), (['black!30'], getEdges)])

# first example of a walk
# walk_graph = Graph(edges=[((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 0), (1, 1)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((1, 2), (1, 1))])
# walk_graph.make_bipartite = lambda: ([], walk_graph.nodes, walk_graph.nodes)
# write_tex('walk', walk_graph, styles=[(['black', '->'], lambda _ : list(walk_graph.src_sink_paths((0, 0), (0, 2) ))[3]), (['black!30'], getEdges)])

path_graph = TeXGraph(edges=[((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 0), (1, 1)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((1, 2), (1, 1))])
path_graph.add_edges(
    [
        TeXGraph.DIRECTED(TikZEdge(*edge)) for edge in list(path_graph.src_sink_paths((0, 0), (0, 2)))[3]
    ],
    [
        TeXGraph.GRAY(edge) for edge in path_graph
    ]
)
path_graph.write('path')

# first example of a bipartite graph
bipartite_graph = TeXGraph(edges=[((2, 3), (2, 2)), ((2, 2), (3, 2)), ((2, 3), (3, 3)), ((3, 3), (3, 2)), ((3, 2), (3, 1)), ((2, 1), (3, 1)), ((2, 2), (2, 1)), ((3, 1), (4, 1)), ((3, 2), (4, 2)), ((4, 2), (4, 1)), ((4, 3), (4, 2)), ((4, 3), (5, 3)), ((5, 3), (5, 2)), ((4, 2), (5, 2)), ((2, 2), (1, 2)), ((1, 2), (1, 1)), ((1, 1), (2, 1)), ((2, 1), (2, 0)), ((2, 0), (1, 0)), ((1, 0), (1, 1)), ((1, 2), (0, 2)), ((0, 2), (0, 1)), ((0, 1), (1, 1))])
bipartite_graph.BIPARTITE()
bipartite_graph.add_edges(
    [
        TikZEdge(*edge) for edge in bipartite_graph
    ]
)
bipartite_graph.write('bipartite')

# first example of a Hamiltonian path & Hamiltonian cycle

hamiltonian_graph = TeXGraph(edges=[((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 0), (1, 1)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((1, 2), (1, 1))])
hamiltonian_graph.BIPARTITE()
hamiltonian_graph.add_edges(
    [
        TeXGraph.DIRECTED(TikZEdge(*edge)) for edge in hamiltonian_graph.hamiltonian_paths((1, 0))[-2]
    ],
    [
        TeXGraph.GRAY(TeXGraph.DOTTED(TikZEdge(*edge))) for edge in hamiltonian_graph.keys()
    ]
)
hamiltonian_graph.write('hamiltonian_path')

hamiltonian_graph.edgesets[0] = [
    TikZEdge(*edge) for edge in hamiltonian_graph.hamiltonian_cycles()[0]
]
hamiltonian_graph.write('hamiltonian_cycle')

hamiltonian_graph.edgesets[0] = [
    TikZEdge((0, 0), (1, 0)), TikZEdge((0, 1), (1, 1)), TikZEdge((0, 2), (1, 2))
]
hamiltonian_graph.write('edge_cover')

# first example of the TST
travelling_salesman_tour_graph = TeXGraph(edges=[((0, 2), (1, 2)), ((1, 2), (2, 2)), ((2, 2), (2, 1)), ((2, 1), (1, 1)), ((0, 2), (0, 1)), ((0, 1), (1, 1)), ((1, 2), (1, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 1), (1, 0)), ((1, 0), (2, 0)), ((2, 1), (2, 0))])
travelling_salesman_tour_graph.BIPARTITE()
travelling_salesman_tour_graph.add_edges(
    [
        TikZEdge(*edge, d=True) for edge in travelling_salesman_tour_graph.travelling_salesman_tours()[0]
    ],
    [
        TeXGraph.GRAY(TeXGraph.DOTTED(edge)) for edge in travelling_salesman_tour_graph
    ]
)
travelling_salesman_tour_graph.write('travelling_salesman_tour')

# one_factor.tex
one_factor_graph = TeXGraph(edges=[((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 1), (1, 0)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((1, 2), (1, 1)), ((1, 2), (2, 2)), ((2, 2), (2, 1)), ((1, 1), (2, 1)), ((1, 0), (2, 0)), ((2, 0), (2, 1)), ((2, 2), (3, 2)), ((3, 2), (3, 1)), ((2, 1), (3, 1)), ((3, 1), (3, 0)), ((2, 0), (3, 0))])
one_factor_graph.BIPARTITE()
one_factor_graph.add_edges(
    [
        TikZEdge(*edge) for edge in [((0, 2), (1, 2)), ((2, 0), (3, 0)), ((2, 2), (2, 1)), ((3, 2), (3, 1)), ((0, 1), (0, 0)), ((1, 1), (1, 0))]
    ]
)
one_factor_graph.write('one_factor')

# no_one_factor.tex
no_one_factor_graph = TeXGraph(edges=Graph.vertex_sequence_to_edges([(0, 1), (1.7293088488585, 2.0047342460869), (1.7347798055927, 0.0047417289428), (0, 1)]))
# no_one_factor_graph.BIPARTITE()
no_one_factor_graph.add_edges(
    [
        TeXGraph.GRAY(TeXGraph.DOTTED(edge)) for edge in no_one_factor_graph
    ]
)
no_one_factor_graph.write('no_one_factor')


# two_factor.tex
two_factor_graph = TeXGraph(edges=[((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 1), (1, 0)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((1, 2), (1, 1)), ((1, 2), (2, 2)), ((2, 2), (2, 1)), ((1, 1), (2, 1)), ((1, 0), (2, 0)), ((2, 0), (2, 1)), ((2, 2), (3, 2)), ((3, 2), (3, 1)), ((2, 1), (3, 1)), ((3, 1), (3, 0)), ((2, 0), (3, 0))])
two_factor_graph.BIPARTITE()
two_factor_graph.add_edges(
    [
        TikZEdge(*edge) for edge in [((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 0), (1, 1)), ((1, 2), (1, 1)), ((0, 2), (1, 2)), ((2, 2), (2, 1)), ((2, 1), (2, 0)), ((2, 0), (3, 0)), ((3, 0), (3, 1)), ((3, 1), (3, 2)), ((2, 2), (3, 2))]
    ]
)
two_factor_graph.write('two_factor')


# no_two_factor.tex
no_two_factor_graph = TeXGraph(edges=list(travelling_salesman_tour_graph.keys()))
no_two_factor_graph.BIPARTITE()
no_two_factor_graph.add_edges(
    [
        TeXGraph.GRAY(TeXGraph.DOTTED(edge)) for edge in no_two_factor_graph
    ]
)
no_two_factor_graph.write('no_two_factor')


# type_3_before_flip.tex
type_3_before_flip = TeXGraph(edges=[((0, 1), (1, 1)), ((1, 1), (1, 0)), ((0, 0), (1, 0)), ((2, 1), (3, 1)), ((2, 1), (2, 0)), ((2, 0), (3, 0)), ((1, 1), (2, 1)), ((1, 0), (2, 0))])
type_3_before_flip.BIPARTITE()
type_3_before_flip.add_edges(
    [
        TikZEdge(*edge) for edge in [((0, 1), (1, 1)), ((1, 1), (1, 0)), ((0, 0), (1, 0)), ((2, 1), (3, 1)), ((2, 1), (2, 0)), ((2, 0), (3, 0))]
    ],
    [
        TeXGraph.GRAY(TeXGraph.DOTTED(edge)) for edge in type_3_before_flip
    ]
)
type_3_before_flip.add_background("\\fill[fill=black!10] (-0.1, 0) rectangle ++(1.1, 1);", "\\fill[fill=black!10] (2, 0) rectangle ++(1.1, 1);")
type_3_before_flip.add_background("\\draw node at (1.5, 0.5) {\\textbf{III}};")
type_3_before_flip.write('type_3_before_flip')

type_3_after_flip = TeXGraph(edges=[((0, 1), (1, 1)), ((1, 1), (1, 0)), ((0, 0), (1, 0)), ((2, 1), (3, 1)), ((2, 1), (2, 0)), ((2, 0), (3, 0)), ((1, 1), (2, 1)), ((1, 0), (2, 0))])
type_3_after_flip.BIPARTITE()
type_3_after_flip.add_edges(
    [
        TikZEdge(*edge) for edge in [((0, 1), (1, 1)), ((1, 1), (2, 1)), ((2, 1), (3, 1)), ((0, 0), (1, 0)), ((1, 0), (2, 0)), ((2, 0), (3, 0))]
    ],
    [
        TeXGraph.GRAY(TeXGraph.DOTTED(edge)) for edge in type_3_after_flip
    ]
)
type_3_after_flip.add_background("\\fill[fill=black!10] (-0.1, 0) rectangle ++(3.1, 1);")
type_3_after_flip.write('type_3_after_flip')


alternating_strip_before_flip = TeXGraph(edges=[((1, 0), (2, 0)), ((1, 1), (2, 1)), ((1, 2), (1, 1)), ((2, 2), (2, 1)), ((1, 4), (1, 3)), ((1, 3), (2, 3)), ((2, 4), (2, 3)), ((1, 6), (1, 5)), ((1, 5), (2, 5)), ((2, 6), (2, 5)), ((1, 7), (2, 7)), ((0, 0), (1, 0)), ((2, 0), (3, 0)), ((3, 0), (3, 1)), ((3, 1), (3, 2)), ((3, 2), (2, 2)), ((1, 2), (0, 2)), ((0, 2), (0, 1)), ((0, 1), (0, 0)), ((1, 4), (0, 4)), ((0, 4), (0, 5)), ((0, 5), (0, 6)), ((0, 6), (1, 6)), ((2, 6), (3, 6)), ((3, 6), (3, 5)), ((3, 5), (3, 4)), ((3, 4), (2, 4)), ((1, 7), (1, 8)), ((1, 8), (0, 8)), ((0, 8), (0, 9)), ((0, 9), (1, 9)), ((1, 9), (2, 9)), ((2, 9), (3, 9)), ((3, 9), (3, 8)), ((3, 8), (2, 8)), ((2, 8), (2, 7)), ((2, 7), (2, 6)), ((2, 6), (1, 6)), ((1, 6), (1, 7)), ((0, 5), (1, 5)), ((1, 5), (1, 4)), ((1, 4), (2, 4)), ((2, 5), (2, 4)), ((2, 5), (3, 5)), ((1, 8), (2, 8)), ((1, 9), (1, 8)), ((2, 9), (2, 8)), ((1, 3), (1, 2)), ((1, 2), (2, 2)), ((2, 2), (2, 3)), ((0, 1), (1, 1)), ((1, 1), (1, 0)), ((2, 0), (2, 1)), ((2, 1), (3, 1))])
alternating_strip_before_flip.BIPARTITE()
alternating_strip_before_flip.add_edges(
    # [
    #     TikZEdge(*edge) for edge in [((1, 0), (2, 0)), ((1, 1), (2, 1)), ((1, 2), (1, 1)), ((2, 2), (2, 1)), ((1, 4), (1, 3)), ((1, 3), (2, 3)), ((2, 4), (2, 3)), ((1, 6), (1, 5)), ((1, 5), (2, 5)), ((2, 6), (2, 5)), ((1, 7), (2, 7))]
    # ],
    # [
    #     TeXGraph.GRAY(TikZEdge(*edge)) for edge in [((1, 7), (2, 7)), ((1, 6), (1, 5)), ((1, 5), (2, 5)), ((2, 5), (2, 6)), ((1, 4), (1, 3)), ((1, 3), (2, 3)), ((2, 3), (2, 4)), ((1, 2), (1, 1)), ((1, 1), (2, 1)), ((2, 1), (2, 2)), ((1, 0), (2, 0)), ((1, 7), (1, 8)), ((2, 7), (2, 8)), ((2, 8), (3, 8)), ((3, 8), (3, 9)), ((3, 9), (2, 9)), ((2, 9), (1, 9)), ((1, 9), (0, 9)), ((0, 9), (0, 8)), ((0, 8), (1, 8)), ((1, 6), (0, 6)), ((0, 6), (0, 5)), ((0, 5), (0, 4)), ((0, 4), (1, 4)), ((2, 4), (3, 4)), ((3, 4), (3, 5)), ((3, 5), (3, 6)), ((2, 6), (3, 6)), ((1, 2), (0, 2)), ((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((2, 0), (3, 0)), ((3, 0), (3, 1)), ((3, 1), (3, 2)), ((3, 2), (2, 2))]
    # ],
    [
        TeXGraph.GRAY(TeXGraph.DOTTED(edge)) for edge in alternating_strip_before_flip
    ]
)

for node in alternating_strip_before_flip.nodes:
    node.style['fill'] = 'black'

for edge in alternating_strip_before_flip:
    print(edge.s.style['fill'])
    print(edge.t.style['fill'])

# alternating_strip_before_flip_boundary = EdgeSet(edges=alternating_strip_before_flip.edgesets[0] + alternating_strip_before_flip.edgesets[1])
# for node in alternating_strip_before_flip_boundary.nodes:
#     if (node[0] + 1, node[1] + 1) in alternating_strip_before_flip_boundary.nodes:
#         print('ayo')
#         alternating_strip_before_flip.add_background(f"{chr(92)}fill[fill=black!10] {str(node)} rectangle ++(1, 1);")

# alternating_strip_before_flip.write('alternating_strip_before_flip')


# polygon = TeXGraph(edges=Graph.vertex_sequence_to_edges([(0.4811533788947,1.8225175674737), (1.4446055899665,1.9117261055359), (1.7568354731842,1.4032374385813), (1.0877714377177,1.0553241401387), (0.37410313322,1.2515829238756), (0.4811533788947,1.8225175674737)]))
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