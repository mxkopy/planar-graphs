import random, copy
from typing import Union, TypeVar, Generic

from math import atan2, degrees

class Node(tuple):

    def __new__(cls, *value, edges=[]):
        node = super(Node, cls).__new__(cls, value)
        node.edges = edges
        return node

    def __add__(self, offset):
        return Node(*(self[i] + o for i, o in enumerate(offset)), edges=self.edges)

    def __sub__(self, offset):
        return Node(*(self[i] - o for i, o in enumerate(offset)), edges=self.edges)

    def __gt__(self, other):
        for v, o in zip(self.value, other):
            if not v > o:
                return False
        return True

    def __lt__(self, other):
        for v, o in zip(self.value, other):
            if not v < o:
                return False
        return True
    
    def __ge__(self, other):
        return self == other or self > other
 
    def __le__(self, other):
        return self == other or self < other


class Edge:

    s: Node
    t: Node
    w: int
    d: bool

    def __init__(self, s, t, w=True, d=False):
        self.s = s if isinstance(s, Node) else Node(*s)
        self.t = t if isinstance(t, Node) else Node(*t)
        self.w = w
        self.d = d
        self.s.edges.append(self)
        self.t.edges.append(self)

    # TODO: we want unique directed edges to have unique hashes, but point hash to a single undirected equivalent
    def __hash__(self):
        if self.d:
            return hash((self.s, self.t))
        else:
            return hash((min(self.s, self.t, key=hash), max(self.s, self.t, key=hash)))

    def __eq__(self, other):
        if isinstance(other, Edge):
            if other.d and self.d:
                return hash((self.s, self.t)) == hash((other.s, other.t))
        return hash(self) == hash(other)
    
    def __getitem__(self, idx):
        return [self.s, self.t, self.w, self.d][idx]
        # if idx != 0 and idx != 1:
        #     raise IndexError(f"Attempted to get item {idx} in edge.") 
        # else:
        #     if idx == 0:
        #         return self.s
        #     else: 
        #         return self.t
    
    def __contains__(self, node):
        return self.s == Node(*node) or self.t == Node(*node)
            
    def __str__(self):
        return f"({self.s}, {self.t})" if self.w is True else f"({self.s}, {self.t}, w={self.w})"
    
    def __repr__(self):
        return (self.s, self.t).__repr__() if self.w is True else (self.s, self.t, self.w).__repr__()

    def other(self, node):
        if self.s == Node(*node):
            return self.t
        elif self.t == Node(*node):
            return self.s
        else:
            raise KeyError(f'Attempted to find other endpoint to ${node} in edge ${self}')
        
    def same(self, node):
        if self.s == Node(*node):
            return self.s
        elif self.t == Node(*node):
            return self.t
        else:
            raise KeyError(f'Attempted to find same endpoint as ${node} in edge ${self}')
        
    def switch(self):
        return self.__class__(self.t, self.s, *self[2:])

    def axis(self):
        x = self.t[0] - self.s[0]
        y = self.t[1] - self.s[1]
        return degrees(atan2(y, x))

class assign:
    pass

class setdict(dict):

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        super(dict, self).__setitem__(key, value)

    def __getitem__(self, key, operation):
        if operation is assign:
            self.__setitem__(key, key) 
        return self[key]

    def __init__(self, *args):
        super().__init__(self)
        for arg in args:
            self[arg]

class EdgeSet(dict):

    nodes: dict[Node]

    def __init__(self, edges=[]):
        self.nodes = set()
        for edge in edges:
            self[edge] = True if not isinstance(edge, Edge) else edge.w

    def __getitem__(self, edge):
        edge = edge if isinstance(edge, Edge) else Edge(*edge)
        return super().__getitem__(edge)

    def __setitem__(self, edge, w, d=False):
        if isinstance(edge, Edge):
            edge.w = w
            edge.d = d
        else:
            edge = Edge(*edge, w=w, d=d)
        super().__setitem__(edge, edge)
        self.nodes.add(edge.s)
        self.nodes.add(edge.t)

    # TODO: see if you can use the fact that connected edges are kept track of in nodes to make this more efficient
    def __delitem__(self, edge):
        super().__delitem__(Edge(*edge))
        for node in edge:
            if sum([node in edge for edge in self]) == 0:
                self.nodes.pop(node)

    def __contains__(self, x):
        if isinstance(x, Node):
            for edge in self:
                if x in self:
                    return True
            return False
        else:
            return super().__contains__(x)
    
    def __repr__(self):
        return list(self.keys()).__repr__()
    
    def __str__(self):
        return list(self.keys()).__str__()

    def copy(self):
        return self.__class__(edges=copy.deepcopy(list(self)))

class Graph(EdgeSet):

    def __setitem__(self, edge, weight):
        if self.is_admissible_edge(edge):
            super().__setitem__(edge, weight)
        else:
            raise KeyError(f"Attempted to set an inadmissible edge {edge} to the graph.")
    
    def is_admissible_edge(self, edge):
        return True

    def neighbors(self, node):
        return [edge.other(node) for edge in self if node in edge]

    def paths_iterator(self, paths, filter=lambda path: len(path) == len(set(path))):
        next = []
        for path in paths:
            for n in self.neighbors(path[-1]):
                new_path = path + [n]
                if filter(new_path):
                    next.append(new_path)
        if len(next) > 0:
            yield next
            yield from self.paths_iterator(next, filter=filter)
        else:
            return None

    def paths(self, src, filter=lambda path: len(path) == len(set(path)) ):
        yield from self.paths_iterator([[src]], filter=filter)

    def cycles(self, src):
        for paths in self.paths(src, filter=lambda path: len(path[:-1]) == len(set(path[:-1]))):
            yield list(filter(lambda path: path[-1] == src, paths))

    def vertex_sequence_to_edges(sequence):
        return [(sequence[i], sequence[i+1]) for i in range(len(sequence)-1)]
    
    def src_sink_paths(self, src, sink, filter=lambda path: len(Graph.vertex_sequence_to_edges(path)) == len(set(Graph.vertex_sequence_to_edges(path)))):
        for paths in self.paths(src, filter=filter):
            for path in paths:
                if path[-1] == sink:
                    yield Graph.vertex_sequence_to_edges(path)


    def make_bipartite(self):
        src = list(self.nodes)[0]
        src.color = 0
        L, R = [], []
        D = set([src])
        Q = [src]
        while len(Q) > 0:
            x = Q.pop()
            for neighbor in self.neighbors(x):
                neighbor.color = not x.color
                if neighbor not in D:
                    Q.append(neighbor)
                    D.add(neighbor)
            D.add(x)
        for node in D:
            if node.color == 0:
                L.append(node)
            else:
                R.append(node)
        return L, R, D

    def hungarian_method(self):

        # Creates an initial matching and alternating tree
        M = []
        S = []
        T = []

        # Gets the partitions of the graph
        X, Y, D = self.make_bipartite(list(self.nodes)[0])
        if set(X + Y) != D or len(X) != len(Y):
            return None

        # Turns the graph into a complete bipartite graph
        complete_graph = Graph(edges=list(self))
        for l in X:
            for r in Y:
                if (l, r) not in complete_graph:
                    complete_graph[l, r] = 0

        # Creates an initial labeling
        labeling = {
            **{y: 0 for y in Y},
            **{x: max(edge.w for edge in complete_graph if x in edge) for x in X}
        }

        def compute_equality_graph():
            return Graph(edges=[edge for edge in complete_graph if labeling[edge.s] + labeling[edge.t] == edge.w])
    
        equality_graph = compute_equality_graph()

        def is_perfect_matching():
            for node in self.nodes:
                is_included = [node in edge for edge in M]
                if sum(is_included) != 1:
                    return False
            return True

        def is_free(y):
            for edge in M:
                if y in edge:
                    return False
            return True

        def neighbors():
            return set(sum([equality_graph.neighbors(node) for node in S], start=[]))

        def neighbors_without_T():
            return [node for node in neighbors() if node not in T]

        def improve_labeling():
            a = max(labeling.values())
            for s in S:
                for y in neighbors_without_T():
                    a = min(a, labeling[s] + labeling[y] - self[s, y].w)
            for v in S:
                labeling[v] = labeling[v] - a
            for v in T:
                labeling[v] = labeling[v] + a

        def matching_node(y):
            for edge in M:
                if y in edge:
                    return edge.other(y)
            return None

        def free_vertex():
            free_vertices = [x for x in X if is_free(x)]
            if len(free_vertices) == 0:
                return None
            else:
                return free_vertices[0]

        def augment_M(y):

            def longest_alternating_path(path, S, T):
                neighbors = [node for node in equality_graph.neighbors(path[-1]) if node not in path and node in S]
                for x in neighbors:
                    next_path = longest_alternating_path(path + [x], T, S)
                    if len(next_path) == len(S) + len(T) + 1:
                        return next_path
                return path

            path = longest_alternating_path([y], S, T)
        
            for edge in M[:]:
                s, t = edge
                if s in path or t in path:
                    M.remove(edge)

            for i in range(0, len(path), 2):
                edge = Edge(path[i], path[i+1])
                M.append(edge)

        S.append(free_vertex())
        while not is_perfect_matching():
            len_EG, len_M, len_S = len(equality_graph), len(M), len(S), 
            N = neighbors_without_T()
            if len(N) == 0:
                improve_labeling()
                equality_graph.clear()
                equality_graph.update(compute_equality_graph())
            else:
                y = N[0]
                if is_free(y):
                    augment_M(y)
                    S.clear()
                    T.clear()
                    S.append(free_vertex())
                else:
                    z = matching_node(y)
                    S.append(z)
                    T.append(y)
            if len(equality_graph) == len_EG and len(M) == len_M and len(S) == len_S:
                return None
        return M

    def dijkstra(self, src):
        nodes = list(self.nodes)
        D = {x: 0 if x == src else float('inf') for x in nodes}
        P = {src: src}
        Q = set(nodes)
        while len(Q) > 0:
            x = min(Q, key=lambda node: D[node])
            Q.remove(x)
            for y in filter(lambda y: y in Q, self.neighbors(x, self)):
                alt = D[x] + self[x, y]
                if alt < D[y]:
                    D[y] = alt
                    P[y] = x
        return D, P
        
    def shortest_path(self, src, dest):
        _, p = self.dijkstra(src)
        if dest in p.keys():
            path = [dest]
            x = dest
            while x != src:
                x = p[x]
                path.append(x)
            path.reverse()
            return Graph.vertex_sequence_to_edges(path)
        return None

    def shortest_cycle(self, src):
        cycle = None
        for n in self.neighbors(src, self):
            graph = self.copy()
            del graph[src, n]
            sp = graph.shortest_path(src, n)
            if sp is not None:
                if cycle is None or len(sp) < len(cycle):
                    cycle = sp + [src]
        return cycle

    def hamiltonian_paths(self, src):
        paths_iterator = self.paths(src)
        hamiltonian_paths = []
        for paths in paths_iterator:
            for path in paths:
                if len(path) == len(self.nodes):
                    hamiltonian_paths.append(Graph.vertex_sequence_to_edges(path))
        return hamiltonian_paths

    def hamiltonian_cycles(self):
        nodes = list(self.nodes)
        cycle_iterator = self.cycles(nodes[0])
        hc_cycles = []
        for cycles in cycle_iterator:
            for cycle in filter(lambda cycle: len(set(cycle)) == len(set(nodes)), cycles):
                hc_cycles.append(cycle)
        return [Graph.vertex_sequence_to_edges(cycle) for cycle in hc_cycles]

    def tours(self):
        src    = list(self.nodes)[0]
        filter = lambda path: len(Graph.vertex_sequence_to_edges(path)) == len(set(Graph.vertex_sequence_to_edges(path)))
        for paths in self.paths(src, filter=filter):
            for path in paths:
                if len(set(path)) == len(self.nodes) and path[-1] == path[0]:
                    yield path

    def travelling_salesman_tours(self):
        travelling_salesman_tours = []
        for tour in self.tours():
            if len(travelling_salesman_tours) == 0 or len(travelling_salesman_tours[0]) == len(tour):
                travelling_salesman_tours.append(Graph.vertex_sequence_to_edges(tour))
            if len(travelling_salesman_tours[0]) < len(tour):
                return travelling_salesman_tours

class SolidGridGraph(Graph):

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]    

    def is_admissible_edge(self, edge):
        edge = Edge(*edge)
        s, t = edge
        for d in SolidGridGraph.directions:
            if s + d == t:
                return True
        return False

    def neighbors(self, node: Node):
        return [node + direction for direction in SolidGridGraph.directions if node + direction in self.nodes]

    # if the number of unique four-cycles is less than four, 
    # there should not be a cycle using edges from the missing four-cycles
    def not_holey(graph: Graph):
        for node in graph.nodes:
            cycle_iterator = graph.cycles(node)
            four_cycles = [[tuple(cycle[i:i+2]) for i in range(len(cycle)-2)] for cycle in list(zip(cycle_iterator, range(4)))[-1][0]]
            if len(four_cycles) < 4:
                for cycles in cycle_iterator:
                    for cycle in cycles:
                        cycle_edges = [tuple(cycle[i:i+2]) for i in range(len(cycle)-2)]
                        for four_cycle in four_cycles:
                            if len(set(four_cycle) & set(cycle_edges)) == 0:
                                return False
        return True

    def make_face(node: Node, dx: int, dy: int):
        a = node + (dx, 0)
        b = node + (dx, dy)
        c = node + (0, dy)
        return [
            (node, a),
            (node, c),
            (a, b),
            (c, b)
        ]

    def add_random_face(graph: Graph, m: int, n: int):
        num_edges_before: int = len(graph)
        nodes: list[Node] = [Node(0, 0)] if len(graph) == 0 else list(graph.nodes)
        random.shuffle(nodes)
        while len(graph) == num_edges_before:
            node: Node = nodes.pop()
            directions = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
            random.shuffle(directions)
            while len(directions) > 0:
                dx, dy = directions.pop()
                if (0, 0) <= node + (dx, dy) <= (m, n):
                    for edge in SolidGridGraph.make_face(node, dx, dy):
                        graph[edge] = 1

    def init_plt(self):
        import matplotlib.pyplot as plt
        try:
            return self.fig, self.ax, self.plt
        except AttributeError:
            self.plt = plt
            self.fig, self.ax = self.plt.subplots()
            self.ax.set_axis_off()
            self.ax.set_xlim(-1, max(x for (x, _) in self.nodes) + 1 )
            self.ax.set_ylim(-1, max(y for (_, y) in self.nodes) + 1 )
            self.plt.edges = []
            self.plt.nodes = []
            return self.fig, self.ax, self.plt

    def display_edges(self, edges, **kwargs):
        for edge in edges:
            s, t = edge
            s_x, s_y = s
            t_x, t_y = t
            plot_edge = self.plt.plot([s_x, t_x], [s_y, t_y] , **kwargs)
            self.plt.edges.append(*plot_edge)

    def display_nodes(self, nodes, **kwargs):
        for node in nodes:
            circle = self.plt.Circle(node, 0.05, **kwargs)
            plot_node = self.ax.add_patch(circle)
            self.plt.nodes.append(plot_node)

    def init_display(self):
        self.init_plt()
        self.display_edges(self.keys(), color='black')
        self.display_nodes(self.nodes, color='black')

    def show(self):
        self.plt.gca().set_aspect('equal')
        self.plt.show()

    def display(self):
        self.init_display()
        self.show()


# goal: find longest cycle for some solid grid graph
# subgoal: test hypothesis 
# - the longest cycle is in the union of two maximally disjoint maximum matchings 

class GraphIterator(SolidGridGraph):

    def __init__(self, edges=[]):
        super().__init__(edges=edges)
        self.node = Node(0, 0)

    def node_iterator(self, node=Node(0, 0)):
        node = Node(node[1] + 1, 0) if node[0] == 0 else Node(node[0] - 1, node[1] + 1)
        yield node
        yield from self.node_iterator(node=node)
 

