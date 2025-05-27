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
    tfc = EdgeSet(*graph.two_factor.as_cycle_facing_inwards()).nodes
    while set(tfc) != set(graph.nodes) and len(tfc) != 0:
        strip = graph.static_alternating_strip()
        tgraph.add_edges([TikZEdge(*edge) for edge in strip])
        tgraph.add_edges([TikZEdge.GRAY(TikZEdge.DOTTED(edge)) for edge in tgraph if not tgraph.two_factor.test_interior(edge.midpoint()) and edge not in tgraph.two_factor])
        tgraph.write(f'{name}{i}')
        graph.edge_flip(strip)
        tfc = EdgeSet(*graph.two_factor.as_cycle_facing_inwards()).nodes
        tgraph = TikZGraph(*list(graph))
        tgraph.make_bipartite()
        tgraph.two_factor = graph.two_factor
        shade_two_factor(tgraph)
        i += 1
    tgraph.write(f'{name}{i}')


alternating_strip_before_flip = SolidGridGraph(((1, 0), (2, 0)), ((1, 1), (2, 1)), ((1, 2), (1, 1)), ((2, 2), (2, 1)), ((1, 4), (1, 3)), ((1, 3), (2, 3)), ((2, 4), (2, 3)), ((1, 6), (1, 5)), ((1, 5), (2, 5)), ((2, 6), (2, 5)), ((1, 7), (2, 7)), ((0, 0), (1, 0)), ((2, 0), (3, 0)), ((3, 0), (3, 1)), ((3, 1), (3, 2)), ((3, 2), (2, 2)), ((1, 2), (0, 2)), ((0, 2), (0, 1)), ((0, 1), (0, 0)), ((1, 4), (0, 4)), ((0, 4), (0, 5)), ((0, 5), (0, 6)), ((0, 6), (1, 6)), ((2, 6), (3, 6)), ((3, 6), (3, 5)), ((3, 5), (3, 4)), ((3, 4), (2, 4)), ((1, 7), (1, 8)), ((1, 8), (0, 8)), ((0, 8), (0, 9)), ((0, 9), (1, 9)), ((1, 9), (2, 9)), ((2, 9), (3, 9)), ((3, 9), (3, 8)), ((3, 8), (2, 8)), ((2, 8), (2, 7)), ((2, 7), (2, 6)), ((2, 6), (1, 6)), ((1, 6), (1, 7)), ((0, 5), (1, 5)), ((1, 5), (1, 4)), ((1, 4), (2, 4)), ((2, 5), (2, 4)), ((2, 5), (3, 5)), ((1, 8), (2, 8)), ((1, 9), (1, 8)), ((2, 9), (2, 8)), ((1, 3), (1, 2)), ((1, 2), (2, 2)), ((2, 2), (2, 3)), ((0, 1), (1, 1)), ((1, 1), (1, 0)), ((2, 0), (2, 1)), ((2, 1), (3, 1)))

alternating_strip_before_flip.two_factor = SolidGridGraph(((0, 0), (1, 0)), ((1, 0), (2, 0)), ((2, 0), (3, 0)), ((3, 0), (3, 1)), ((3, 1), (3, 2)), ((3, 2), (2, 2)), ((2, 2), (2, 1)), ((2, 1), (1, 1)), ((1, 1), (1, 2)), ((1, 2), (0, 2)), ((0, 2), (0, 1)), ((0, 1), (0, 0)), ((1, 3), (2, 3)), ((2, 3), (2, 4)), ((2, 4), (3, 4)), ((3, 4), (3, 5)), ((3, 5), (3, 6)), ((3, 6), (2, 6)), ((2, 6), (2, 5)), ((2, 5), (1, 5)), ((1, 5), (1, 6)), ((0, 6), (1, 6)), ((0, 6), (0, 5)), ((0, 5), (0, 4)), ((0, 4), (1, 4)), ((1, 4), (1, 3)), ((1, 7), (2, 7)), ((2, 7), (2, 8)), ((2, 8), (3, 8)), ((3, 8), (3, 9)), ((3, 9), (2, 9)), ((2, 9), (1, 9)), ((1, 9), (0, 9)), ((0, 9), (0, 8)), ((0, 8), (1, 8)), ((1, 8), (1, 7)))

longest_strip = max(SolidGridGraph.get_alternating_strips(alternating_strip_before_flip), key=len)
alternating_strip_after_flip = alternating_strip_before_flip.edge_flip(longest_strip)

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
        TikZOptions.GRAY(TikZEdge.DOTTED(TikZEdge(edge))) for edge in alternating_strip_after_flip.two_factor.remove_directed()
    ]
)

for node in alternating_strip_after_flip.nodes:
    if node not in longest_strip_flipped:
        node.style["draw"] = "none"
        node.style["fill"] = "none"

alternating_strip_after_flip.add_background(*shaded)
alternating_strip_after_flip.write('alternating_strip_after_flip')
del alternating_strip_after_flip, alternating_strip_before_flip
