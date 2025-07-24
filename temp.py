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
for n in range(50, 1000):
    cursor.execute(f"SELECT dtree, new FROM graphs WHERE id = ?", (n,))
    dtree, new = cursor.fetchone()
    dtree, new = SolidGridGraph().decompress(dtree), SolidGridGraph().decompress(new)
    if len(dtree) != len(new):
        print(n)
        if len(dtree) < len(new):
            names = f'dtree_lt_counterexample_{lt}', f'new_gt_counterexample_{lt}', f'combo_dtree_lt_new_{lt}'
            lt += 1
        else:
            names = f'dtree_gt_counterexample_{gt}', f'new_lt_counterexample_{gt}', f'combo_new_lt_dtree_{gt}'
            gt += 1

        cursor.execute(f"SELECT data FROM graphs WHERE id = ?", (n,))
        full = cursor.fetchone()[0]
        full = SolidGridGraph().decompress(full)

        # full = SolidGridGraph(((2, 6), (2, 5)), ((3, 6), (3, 5)), ((2, 7), (1, 7)), ((2, 8), (1, 8)), ((3, 8), (3, 9)), ((4, 8), (4, 9)), ((4, 7), (5, 7)), ((4, 6), (5, 6)), ((4, 7), (4, 6)), ((3, 6), (4, 6)), ((2, 6), (3, 6)), ((2, 5), (2, 4)), ((3, 5), (3, 4)), ((2, 5), (3, 5)), ((1, 4), (2, 4)), ((2, 4), (3, 4)), ((1, 4), (1, 3)), ((1, 3), (2, 3)), ((2, 4), (2, 3)), ((3, 4), (3, 3)), ((2, 3), (3, 3)), ((3, 4), (4, 4)), ((4, 4), (4, 3)), ((3, 3), (4, 3)), ((4, 3), (5, 3)), ((5, 3), (5, 2)), ((4, 2), (5, 2)), ((4, 3), (4, 2)), ((1, 3), (0, 3)), ((0, 3), (0, 2)), ((0, 2), (1, 2)), ((1, 2), (1, 3)), ((1, 2), (1, 1)), ((1, 1), (2, 1)), ((2, 1), (2, 2)), ((1, 2), (2, 2)), ((2, 2), (3, 2)), ((3, 2), (3, 1)), ((2, 1), (3, 1)), ((3, 2), (4, 2)), ((4, 2), (4, 1)), ((3, 1), (4, 1)), ((2, 1), (2, 0)), ((2, 0), (3, 0)), ((3, 0), (3, 1)), ((2, 2), (2, 3)), ((3, 3), (3, 2)), ((3, 6), (3, 7)), ((3, 7), (4, 7)), ((2, 7), (2, 6)), ((2, 7), (3, 7)), ((2, 8), (2, 7)), ((2, 8), (3, 8)), ((3, 8), (3, 7)), ((3, 8), (4, 8)), ((4, 8), (4, 7)), ((5, 7), (5, 6)), ((3, 9), (4, 9)), ((1, 8), (1, 7)))

        dtree, new, combo = full.dual_tree(), full.new_ilp(), full.fix_combo(full.combination_ilp())

        nodes = set(list(dtree.dual().interior.nodes)).intersection(set(list(new.dual().interior.nodes)))

        TikZGraph(*full, *dtree, *full.dual())\
            .add_edges([TikZEdge(edge, force=True) for edge in dtree.remove_directed()])\
            .add_edges([TikZEdge.GRAY(TikZEdge.DOTTED(edge)) for edge in full.remove_directed()])\
            .on_nodes(lambda node: TikZNode.NONE(node) if node not in dtree.dual().interior else node)\
            .write(names[0])

        TikZGraph(*full, *new, *full.dual())\
            .add_edges([TikZEdge(edge, force=True) for edge in new.remove_directed()])\
            .add_edges([TikZEdge.GRAY(TikZEdge.DOTTED(edge)) for edge in full.remove_directed()])\
            .on_nodes(lambda node: TikZNode.NONE(node) if node not in new.dual().interior else node)\
            .write(names[1])

        TikZGraph(*full, *combo, *full.dual())\
            .add_edges([TikZEdge(edge, force=True) for edge in combo.remove_directed()])\
            .add_edges([TikZEdge.GRAY(TikZEdge.DOTTED(edge)) for edge in full.remove_directed()])\
            .on_nodes(lambda node: TikZNode.NONE(node) if node not in combo.dual().interior else node)\
            .write(names[2])

            # .on_nodes(lambda node: TikZNode.NONE(node) if node not in combo.dual().interior else node)\
