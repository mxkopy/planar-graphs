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
               data BLOB NOT NULL,
               dtree BLOB NOT NULL,
               new BLOB NOT NULL,
               dtree_dt REAL NOT NULL,
               new_dt REAL NOT NULL,
               dtree_len INTEGER NOT NULL,
               new_len INTEGER NOT NULL
               );''')

for n in range(10, 1000):
    cursor.execute(f"SELECT data FROM graphs WHERE id = ?", (n,))
    if cursor.fetchone() is None:
        sgg = SolidGridGraph.random_thick_solid_grid_graph(n=n)
        dtree, new = sgg.dual_tree(), sgg.new_ilp()
        dtree_dt, new_dt = dtree.model.solve_details.dettime, new.model.solve_details.dettime
        cursor.execute('''INSERT INTO graphs 
                       (id, data, dtree, new, dtree_dt, new_dt, dtree_len, new_len) 
                       VALUES 
                       (?, ?, ?, ?, ?, ?, ?, ?)''',
                       (
                            n, 
                            sgg.compress(), 
                            dtree.compress(), 
                            new.compress(), 
                            dtree_dt, 
                            new_dt, 
                            len(dtree),
                            len(new)
                        )
        )
        print(f'{n} dtree_dt: {dtree_dt} new_dt: {new_dt} dtree_len: {len(dtree)} new_len: {len(new)}\n')
        conn.commit()
