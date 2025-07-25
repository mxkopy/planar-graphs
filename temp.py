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

lt, gt = 0, 0
for n in range(10, 1000):
    cursor.execute(f"SELECT graph, dtree, new, combo FROM graphs WHERE id = ?", (n,))
    graph, dtree, new, combo = cursor.fetchone()
    graph, dtree, new, combo = SolidGridGraph().decompress(graph), SolidGridGraph().decompress(dtree), SolidGridGraph().decompress(new), SolidGridGraph().decompress(combo)
    # if len(dtree) != len(new):
    if len(combo) < len(new) or len(combo) < len(dtree):
        if len(combo) < len(new):
            names = f'dtree_lt_counterexample_{lt}', f'new_gt_counterexample_{lt}', f'combo_dtree_lt_new_{lt}'
            lt += 1
        else:
            names = f'dtree_gt_counterexample_{gt}', f'new_lt_counterexample_{gt}', f'combo_new_lt_dtree_{gt}'
            gt += 1

        dtree, new, combo = graph.dual_tree(), graph.new_ilp(), graph.combination_ilp()

        nodes = set(list(dtree.dual().interior.nodes)).intersection(set(list(new.dual().interior.nodes)))

        TikZGraph(*graph, *dtree, *graph.dual())\
            .add_edges([TikZEdge(edge, force=True) for edge in dtree.remove_directed()])\
            .add_edges([TikZEdge.GRAY(TikZEdge.DOTTED(edge)) for edge in graph.remove_directed()])\
            .on_nodes(lambda node: TikZNode.NONE(node) if node not in dtree.dual().interior else node)\
            .write(names[0])

        TikZGraph(*graph, *new, *graph.dual())\
            .add_edges([TikZEdge(edge, force=True) for edge in new.remove_directed()])\
            .add_edges([TikZEdge.GRAY(TikZEdge.DOTTED(edge)) for edge in graph.remove_directed()])\
            .on_nodes(lambda node: TikZNode.NONE(node) if node not in new.dual().interior else node)\
            .write(names[1])

        TikZGraph(*graph, *combo, *graph.dual())\
            .add_edges([TikZEdge(edge, force=True) for edge in combo.remove_directed()])\
            .add_edges([TikZEdge.GRAY(TikZEdge.DOTTED(edge)) for edge in graph.remove_directed()])\
            .on_nodes(lambda node: TikZNode.NONE(node) if node not in combo.dual().interior else node)\
            .write(names[2])

            # .on_nodes(lambda node: TikZNode.NONE(node) if node not in combo.dual().interior else node)\
