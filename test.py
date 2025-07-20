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
               dtree_dt REAL,
               new_dt REAL
               );''')

for n in range(10, 1000):
    cursor.execute(f"SELECT data FROM graphs WHERE id = ?", (n,))
    data = cursor.fetchone()
    if data is None:
        sgg = SolidGridGraph.random_thick_solid_grid_graph(n=n)
        cursor.execute("INSERT INTO graphs (id, data) VALUES (?,?)", (n, sgg.compress(),))
    else:
        sgg = SolidGridGraph().decompress(data)

    dtree_dt = sgg.dual_tree().model.solve_details.dettime
    new_dt = sgg.new_ilp().model.solve_details.dettime
    print(f'{n} {dtree_dt} {new_dt}\n')
    cursor.execute('''UPDATE graphs
                   SET dtree_dt = ?, new_dt = ?
                   WHERE id = ?
                   ''', 
                   (dtree_dt, new_dt, n))
    conn.commit()
