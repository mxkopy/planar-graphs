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
cursor.execute('''CREATE TABLE IF NOT EXISTS graphs(
               id INTEGER PRIMARY KEY,
               graph BLOB NOT NULL,
               dtree BLOB NOT NULL,
               new BLOB NOT NULL,
               combo BLOB NOT NULL,
               dtree_dt REAL NOT NULL,
               new_dt REAL NOT NULL,
               combo_dt REAL NOT NULL,
               dtree_len INTEGER NOT NULL,
               new_len INTEGER NOT NULL,
               combo_len INTEGER NOT NULL
               );''')

for n in range(40, 1000):
    sgg = SolidGridGraph.random_thick_solid_grid_graph(n=n)

    while len(sgg.allowed_four_fulls()) > 0:
        sgg = SolidGridGraph.random_thick_solid_grid_graph(n=n)

    cursor.execute(f"SELECT graph FROM graphs WHERE id = ?", (n,))
    data = cursor.fetchone()

    if data is None:
        sgg = SolidGridGraph.random_thick_solid_grid_graph(n=n)
        dtree, new, combo = sgg.dual_tree(), sgg.new_ilp(), sgg.fix_combo(sgg.combination_ilp())
        dtree_dt, new_dt, combo_dt = dtree.model.solve_details.dettime, new.model.solve_details.dettime, combo.model.solve_details.dettime
        cursor.execute('''INSERT INTO graphs 
                       (id, graph, dtree, new, combo, dtree_dt, new_dt, combo_dt, dtree_len, new_len, combo_len) 
                       VALUES 
                       (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                       (
                            n, 
                            sgg.compress(), 
                            dtree.compress(), 
                            new.compress(), 
                            combo.compress(),
                            dtree_dt, 
                            new_dt, 
                            combo_dt,
                            len(dtree),
                            len(new),
                            len(combo)
                        )
        )
        print(f'{n} dtree_dt: {dtree_dt} new_dt: {new_dt} combo_dt: {combo_dt} dtree_len: {len(dtree)} new_len: {len(new)} combo_len: {len(combo)}\n')
        conn.commit()

    if data is not None:
        sgg = SolidGridGraph().decompress(data[0])
        dtree, new, combo = sgg.dual_tree(), sgg.new_ilp(), sgg.fix_combo(sgg.combination_ilp())
        dtree_dt, new_dt, combo_dt = dtree.model.solve_details.dettime, new.model.solve_details.dettime, combo.model.solve_details.dettime
        cursor.execute('''UPDATE graphs 
                       SET dtree = ?, new = ?, combo = ?, dtree_dt = ?, new_dt = ?, combo_dt = ?, dtree_len = ?, new_len = ?, combo_len = ?
                       WHERE id = ?
                       ''',
                       (
                            dtree.compress(), 
                            new.compress(),
                            combo.compress(),
                            dtree_dt, 
                            new_dt, 
                            combo_dt,
                            len(dtree),
                            len(new),
                            len(combo),
                            n
                        )
        )
        print(f'{n} dtree_dt: {dtree_dt} new_dt: {new_dt} combo_dt: {combo_dt} dtree_len: {len(dtree)} new_len: {len(new)} combo_len: {len(combo)}\n')
        conn.commit()

