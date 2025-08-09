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
               graph_n INTEGER NOT NULL,
               ilp BLOB NOT NULL,
               ilp_n INTEGER NOT NULL,
               dt REAL NOT NULL,
               n_cuts INTEGER NOT NULL
               );''')

for n in range(10, 1000):

    cursor.execute(f"SELECT graph FROM graphs WHERE id = ?", (n,))
    data = cursor.fetchone()

    if data is None:
        graph = SolidGridGraph.random_thick_solid_grid_graph(n=n)
        while len(graph.allowed_four_fulls()) > 0:
            graph = SolidGridGraph.random_thick_solid_grid_graph(n=n)
        graph_n = len(graph.nodes)
        ilp = graph.combination_ilp()
        ilp_n = len(ilp.nodes)
        dt = ilp.model.solve_details.dettime
        n_cuts = sum(ilp.model.get_cuts().values())
        cursor.execute('''INSERT INTO graphs 
                       (id, graph, graph_n, ilp, ilp_n, dt, n_cuts) 
                       VALUES 
                       (?, ?, ?, ?, ?, ?, ?)''',
                       (
                            n, 
                            graph.compress(),
                            graph_n, 
                            ilp.compress(), 
                            ilp_n,
                            dt, 
                            n_cuts
                        )
        )
        print(f'n: {n} nodes: {graph_n} covered: {ilp_n} dt: {dt} n_cuts: {n_cuts}\n')
        conn.commit()