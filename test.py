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
cursor.execute('''
                 CREATE TABLE IF NOT EXISTS graphs(
                   id INTEGER PRIMARY KEY, 
                   graph BLOB NOT NULL,
                   graph_n INTEGER NOT NULL,
                   cycle_n INTEGER NOT NULL,
                   lc_dt REAL NOT NULL,
                   lp_dt REAL NOT NULL,
                   ilp_dt REAL NOT NULL
                 );''')


for n in range(10, 1020):

    print(n)
    cursor.execute(f"SELECT graph FROM graphs WHERE id = ?", (n,))
    data = cursor.fetchone()

    if data is None:
        graph = SolidGridGraph.random_thick_solid_grid_graph(n=n)
        while len(graph.allowed_four_fulls()) > 0:
            graph = SolidGridGraph.random_thick_solid_grid_graph(n=n)
        lc = graph.longest_cycle()
        lc_n = len(lc)
        lc_dt = lc.model.solve_details.dettime
        lp_dt  = graph.cycle_of_length_n_minus_boundary(lc_n, ilp=False).model.solve_details.dettime
        ilp_dt = graph.cycle_of_length_n_minus_boundary(lc_n, ilp=True).model.solve_details.dettime
        cursor.execute(
            '''INSERT INTO graphs 
                       (id, graph, graph_n, cycle_n, lc_dt, lp_dt, ilp_dt) 
                       VALUES 
                       (?, ?, ?, ?, ?, ?, ?)''',
                       (
                            n, 
                            graph.compress(),
                            len(graph.nodes),
                            lc_n,
                            lc_dt,
                            lp_dt,
                            ilp_dt
                        )
        )
        print(f'n: {n} lc_dt: {lc_dt} lp_dt: {lp_dt} ilp_dt: {ilp_dt}\n')
        conn.commit()