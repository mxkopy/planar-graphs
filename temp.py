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
    cursor.execute(f"SELECT dtree, new FROM graphs WHERE id = ?", (n,))
    dtree, new = cursor.fetchone()
    dtree, new = SolidGridGraph().decompress(dtree), SolidGridGraph().decompress(new)
    if len(dtree) < len(new):
        cursor.execute(f"SELECT data FROM graphs WHERE id = ?", (n,))
        full = cursor.fetchone()[0]
        full = SolidGridGraph().decompress(full)

        TikZGraph(*full, *dtree)\
            .add_edges([TikZEdge(edge, force=True) for edge in dtree.remove_directed()])\
            .add_edges([TikZEdge.GRAY(TikZEdge.DOTTED(edge)) for edge in full.remove_directed()])\
            .on_nodes(lambda node: TikZNode.NONE(node) if node in full else node)\
            .write(f'dtree_lt_counterexample_{lt}')

        TikZGraph(*full, *dtree)\
            .add_edges([TikZEdge(edge, force=True) for edge in new.remove_directed()])\
            .add_edges([TikZEdge.GRAY(TikZEdge.DOTTED(edge)) for edge in full.remove_directed()])\
            .on_nodes(lambda node: TikZNode.NONE(node) if node in full else node)\
            .write(f'new_gt_counterexample_{lt}')
        lt += 1

    if len(dtree) > len(new):
        cursor.execute(f"SELECT data FROM graphs WHERE id = ?", (n,))
        full = cursor.fetchone()[0]
        full = SolidGridGraph().decompress(full)

        TikZGraph(*full, *dtree)\
            .add_edges([TikZEdge(edge, force=True) for edge in dtree.remove_directed()])\
            .add_edges([TikZEdge.GRAY(TikZEdge.DOTTED(edge)) for edge in full.remove_directed()])\
            .on_nodes(lambda node: TikZNode.NONE(node) if node in full else node)\
            .write(f'dtree_gt_counterexample_{gt}')

        TikZGraph(*full, *dtree)\
            .add_edges([TikZEdge(edge, force=True) for edge in new.remove_directed()])\
            .add_edges([TikZEdge.GRAY(TikZEdge.DOTTED(edge)) for edge in full.remove_directed()])\
            .on_nodes(lambda node: TikZNode.NONE(node) if node in full else node)\
            .write(f'new_lt_counterexample_{gt}')

        gt += 1
