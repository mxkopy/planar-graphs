from graph import *
from to_tex import *


example_graph_length = 2
example_graph = Graph(edges=[((0, 0), (example_graph_length, 0)), ((example_graph_length, 0), (example_graph_length / 2, ((example_graph_length ** 2) - ((example_graph_length / 2) ** 2))**0.5 )), ((example_graph_length / 2, ((example_graph_length ** 2) - ((example_graph_length / 2) ** 2))**0.5), (0, 0))])
example_graph.make_bipartite = lambda: ([], example_graph.nodes, example_graph.nodes)
# example_graph = TeXGraph(edges=[edge for edge in example_graph.keys()])
# example_graph.add_edges(
#     [
#         TeXGraph.UNDIRECTED_BLACK(edge) for edge in example_graph.keys()
#     ]
# )
# example_graph.write('graph')

# write_tex('graph', example_graph, styles=[(['black'], getEdges)])

# first example of a trail
trail_graph = Graph(edges=[((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 0), (1, 1)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((1, 2), (1, 1))])
trail_graph.make_bipartite = lambda: ([], trail_graph.nodes, trail_graph.nodes)
# write_tex('trail', trail_graph, styles=[(['black', '->'], lambda _ : Graph.vertex_sequence_to_edges([(0, 1), (1, 1), (1, 2), (0, 2), (0, 1), (0, 0)])), (['black!30'], getEdges)])

# first example of a walk
walk_graph = Graph(edges=[((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 0), (1, 1)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((1, 2), (1, 1))])
walk_graph.make_bipartite = lambda: ([], walk_graph.nodes, walk_graph.nodes)
write_tex('walk', walk_graph, styles=[(['black', '->'], lambda _ : list(walk_graph.src_sink_paths((0, 0), (0, 2) ))[3]), (['black!30'], getEdges)])

# first example of a bipartite graph
bipartite_graph = Graph(edges=[((0, 2), (1, 2)), ((1, 2), (2, 2)), ((2, 2), (2, 1)), ((2, 1), (1, 1)), ((0, 2), (0, 1)), ((0, 1), (1, 1)), ((1, 2), (1, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 1), (1, 0)), ((1, 0), (2, 0)), ((2, 1), (2, 0))])
write_tex('bipartite', bipartite_graph, styles=[(['black'], getEdges)])

# first example of a Hamiltonian path & Hamiltonian cycle
hamiltonian_graph = Graph(edges=[((0, 2), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 0), (1, 1)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((1, 2), (1, 1))])
# hamiltonian_graph.make_bipartite = lambda: ([], hamiltonian_graph.nodes, hamiltonian_graph.nodes)
write_tex('hamiltonian_path', hamiltonian_graph, styles=[(['black', '->'], lambda _: hamiltonian_graph.hamiltonian_paths((1, 0))[-2]), (['black!30', 'dotted'], getEdges)])
write_tex('hamiltonian_cycle', hamiltonian_graph, styles=[(['black'], getHamiltonianCycle), (['black!30', 'dotted'], getEdges)])
write_tex('edge_cover', hamiltonian_graph, styles=[(['black'], lambda _: [Edge((0, 0), (1, 0)), Edge((0, 1), (1, 1)), Edge((0, 2), (1, 2))]), (['black!30', 'dotted'], getEdges)])


# two_factor_cover_graph = Graph(edges=[((0, 2), (0, 1)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((1, 2), (1, 1)), ((1, 1), (1, 0)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 0), (2, 0)), ((2, 0), (3, 0)), ((3, 1), (3, 0)), ((2, 1), (2, 0)), ((2, 1), (3, 1)), ((2, 2), (2, 1)), ((2, 2), (3, 2)), ((3, 2), (3, 1)), ((1, 1), (2, 1)), ((1, 2), (2, 2))])
# two_factor_cover_graph.make_bipartite = lambda: ([], two_factor_cover_graph.nodes, two_factor_cover_graph.nodes)
# write_tex('two_factor_cover', two_factor_cover_graph, styles=[(['black'], getTwoFactor), (['black!30', 'dotted'], getEdges)])

# first example of a travelling salesman tour 
travelling_salesman_tour_graph = Graph(edges=[((0, 2), (1, 2)), ((1, 2), (2, 2)), ((2, 2), (2, 1)), ((2, 1), (1, 1)), ((0, 2), (0, 1)), ((0, 1), (1, 1)), ((1, 2), (1, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 1), (1, 0)), ((1, 0), (2, 0)), ((2, 1), (2, 0))])
# travelling_salesman_tour_graph.make_bipartite = lambda: ([], travelling_salesman_tour_graph.nodes, travelling_salesman_tour_graph.nodes)
tours = getTravellingSalesmanTour(travelling_salesman_tour_graph)
write_tex('travelling_salesman_tour', travelling_salesman_tour_graph, styles=[(['black'], getTravellingSalesmanTour), (['black!30', 'dotted'], getEdges)])

((0, 2), (1, 2)), ((1, 2), (1, 1)), ((0, 2), (0, 1)), ((0, 1), (1, 1)), ((1, 1), (2, 1)), ((1, 2), (2, 2)), ((2, 2), (2, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 1), (1, 0))


# interior_exterior_graph = Graph(edges=[((0, 2), (0, 1)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((1, 2), (1, 1)), ((1, 1), (2, 1)), ((1, 2), (2, 2)), ((2, 2), (2, 1)), ((2, 1), (3, 1)), ((2, 2), (3, 2)), ((3, 2), (3, 1)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((1, 1), (1, 0)), ((1, 0), (2, 0)), ((2, 1), (2, 0)), ((2, 0), (3, 0)), ((3, 1), (3, 0)), ((0, 2), (0, 3)), ((0, 3), (1, 3)), ((1, 3), (1, 2)), ((1, 3), (2, 3)), ((2, 3), (2, 2)), ((2, 3), (3, 3)), ((3, 3), (3, 2))])
# print(Edge((1, 1), (1, 0), d=True).axis())