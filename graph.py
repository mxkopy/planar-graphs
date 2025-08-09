import random, copy
from typing import Union, TypeVar, Generic
from math import atan2, degrees, fmod, sqrt, pow, sin, cos, radians
from itertools import chain, combinations
from docplex.mp.model import Model
import os, pickle
import hashlib

def diff_update_dict(dict_a, dict_b, exclude=[]):
    for key in dict_b.keys():
        if key not in dict_a:
            if hasattr(dict_b[key], 'copy') and not 'lock' in dict_b:
                dict_b['lock'] = True
                dict_a[key] = dict_b[key].copy()
                del dict_b['lock']
            else:
                dict_a[key] = dict_b[key]

class get_is_set(type):

    def make_setitem(cls):
        def setitem(self, key, op=None):
            if isinstance(key, tuple) and len(key) > 1 and key[1] is assign:
                return cls.getitem(self, key[0], op=key[1])
            else:
                return cls.getitem(self, key, op=op)
        return setitem

    def __new__(meta_cls, name, parents, attrs):
        new_cls = super().__new__(meta_cls, name, parents, attrs)
        new_cls.getitem = new_cls.__getitem__
        new_cls.__getitem__ = get_is_set.make_setitem(new_cls)
        return new_cls

class assign:
    pass

class setdict(dict, metaclass=get_is_set):

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        super(setdict, self).__setitem__(key, value)

    def __getitem__(self, key, op=None):
        if op is assign:
            self.__setitem__(key, key)
        return super().__getitem__(key)

    def __init__(self, *args):
        super().__init__()
        for arg in args:
            self[arg, assign]

    def __str__(self):
        return str(list(self.keys()))
    
    def copy(self):
        sd = self.__class__()
        for x in list(self):
            sd[x, assign]
        diff_update_dict(sd.__dict__, self.__dict__)
        return sd

class Node(tuple):

    def __new__(cls, *value):
        while isinstance(value, tuple) and len(value) == 1:
            value = value[0]
        node = super(Node, cls).__new__(cls, value)
        node.parent = node
        node.size = 1
        return node

    def copy(self):
        node = self.__class__(*self)
        diff_update_dict(node.__dict__, self.__dict__, exclude=['parent', 'size'])
        node.parent = node
        node.size = 1
        return node

    def __abs__(self):
        return Node(abs(x) for x in self)

    def __add__(self, other):
        if isinstance(other, (tuple, Node)):
            return Node(self[i]+o for i, o in enumerate(other))
        elif isinstance(other, Edge):
            return other+self
        else:
            return Node(i+other for i in self)

    def __sub__(self, other):
        if isinstance(other, (tuple, Node)):
            return Node(self[i]-o for i, o in enumerate(other))
        else:
            return Node(i-other for i in self)

    def __mul__(self, other):
        if isinstance(other, (tuple, Node)):
            return Node(self[i]*o for i, o in enumerate(other))
        else:
            return Node(i*other for i in self)
    
    def __truediv__(self, other):
        if isinstance(other, (tuple, Node)):
            return Node(0 if i == 0 else i/o for i, o in zip(self, other))
        else:
            return Node(0 if i == 0 else i/other for i in self)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return self - other

    def __rmul__(self, other):
        return self * other

    def __gt__(self, other):
        for v, o in zip(self, other):
            if not v > o:
                return False
        return True

    def __lt__(self, other):
        for v, o in zip(self, other):
            if not v < o:
                return False
        return True
    
    def __ge__(self, other):
        return self == other or self > other
 
    def __le__(self, other):
        return self == other or self < other
    
    def __round__(self, digits=None):
        return self.__class__(round(self[0], digits), round(self[1], digits))

    # def __hash__(self):
    #     print(self)
    #     return super().__hash__() - ((self[0] < 0) + (self[1] < 0))
    
    def id(self):
        return int.from_bytes(hashlib.md5(str(self).encode(), usedforsecurity=False).digest(), byteorder='big')

    def neighbors(self):
        yield from (edge.t for edge in self.edges if edge.s == self)

class Edge:

    s: Node
    t: Node

    def __init__(self, s, t):
        self.s = s if isinstance(s, Node) else Node(*s)
        self.t = t if isinstance(t, Node) else Node(*t)
        self.weight = 1

    def copy(self):
        edge = self.__class__(self.s.copy(), self.t.copy())
        diff_update_dict(edge.__dict__, self.__dict__, exclude=['s', 't'])
        edge.s, edge.t = self.s.copy(), self.t.copy()
        return edge

    def __hash__(self):
        return hash((self.s, self.t))

    def __eq__(self, other):
        return hash(self) == hash(other)
    
    def __getitem__(self, idx):
        return [self.s, self.t][idx]
    
    def __contains__(self, node):
        return self.s == node or self.t == node
            
    def __str__(self):
        return f"({self.s}, {self.t})"
    
    def __repr__(self):
        return (self.s, self.t).__repr__()

    def __add__(self, other):
        if isinstance(other, Edge):
            return self.__class__(self.s+other.s, self.t+other.t)
        elif isinstance(other, (float, int, Node)):
            return self.__class__(self.s+other, self.t+other)
        else:
            raise ValueError(f"Attempted to add {type(other)} to an edge")

    def __sub__(self, other):
        if isinstance(other, Edge):
            return self.__class__(self.s-other.s, self.t-other.t)
        elif isinstance(other, (float, int, Node)):
            return self.__class__(self.s-other, self.t-other)
        else:
            raise ValueError(f"Attempted to add {type(other)} to an edge")

    def __mul__(self, other):
        if isinstance(other, Edge):
            return self.__class__(self.s*other.s, self.t*other.t)
        elif isinstance(other, (float, int, Node)):
            return self.__class__(self.s*other, self.t*other)
        else:
            raise ValueError(f"Attempted to add {type(other)} to an edge")

    def __truediv__(self, other):
        if isinstance(other, Edge):
            return self.__class__(self.s/other.s, self.t/other.t)
        elif isinstance(other, (float, int, Node)):
            return self.__class__(self.s/other, self.t/other)
        else:
            raise ValueError(f"Attempted to add {type(other)} to an edge")

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return self - other

    def __rmul__(self, other):
        return self * other
    
    def __round__(self, digits=None):
        self.s, self.t = round(self.s, digits), round(self.t, digits)
        return self

    def double_directed(self):
        # if not hasattr(self.s, 'color') or not hasattr(self.t, 'color'):
        return self.__class__(min(self.s, self.t, key=hash), max(self.s, self.t, key=hash))
        # else:
            # return self.__class__(min(self.s, self.t, key=lambda node: node.color), max(self.s, self.t, key=lambda node: node.color))

    def meets(self, other):
        return self.s if self.s == other.t else self.t if self.t == other.t or self.t == other.s else None

    def other(self, node):
        if self.s == node:
            return self.t
        elif self.t == node:
            return self.s
        else:
            raise KeyError(f'Attempted to find other endpoint to ${node} in edge ${self}')
        
    def same(self, node):
        return self.other(self.other(node))

    def switch(self):
        return self.__class__(self.t, self.s)

    def dist(self):
        d = self.t - self.s
        return sqrt((d[0]*d[0])+(d[1]*d[1]))

    # TODO: make this radians by default
    def axis(self):
        x = self.t[0] - self.s[0]
        y = self.t[1] - self.s[1]
        return degrees(atan2(y, x))
    
    def axis_in_radians(self):
        x = self.t[0] - self.s[0]
        y = self.t[1] - self.s[1]
        return atan2(y, x)
    
    def direction(self):

        return self.__class__(self.s-self.s, (self.t-self.s) / abs(self.t-self.s) )
    
    def rotate(self, degrees):
        theta = radians(degrees)
        u = self.direction()
        v = u.t*(cos(theta), sin(theta))
        w = u.t*(sin(theta), cos(theta))
        d = (v[0]-v[1], w[0]+w[1])
        if isinstance(self.s[0], int):
            d = round(d[0]), round(d[1])
        else: 
            d = round(d[0], 8), round(d[1], 8)
        return self.__class__(self.s, self.s+d)

    def rotate_right(self):
        return self.rotate(-90)
    
    def rotate_left(self):
        return self.rotate(90)

    def take_right(self):
        return self.t+self.rotate(-90).direction()
    
    def take_left(self):
        return self.t+self.rotate(90).direction()

    def go_forward(self):
        return self+self.direction().t

    def parallel_right(self):
        d = self.direction()
        r = self.rotate(-90)
        return r.t+d
    
    def parallel_left(self):
        d = self.direction()
        l = self.rotate(90)
        return l.t+d

    # Determines if a line segment from point a to b intersects with an edge. If so, returns the point at which it does. 
    # Solution ripped from an old graphics homework. 
    def intersect(self, a, b):
        s = (-self.s[1]*a[0]+self.s[1]*b[0]+self.s[0]*a[1]-self.s[0]*b[1]+b[1]*a[0]-b[0]*a[1])/(-self.s[1]*a[0]+self.t[1]*a[0]+self.s[1]*b[0]-self.t[1]*b[0]+self.s[0]*a[1]-self.t[0]*a[1]-self.s[0]*b[1]+self.t[0]*b[1])
        t = ((self.t[0]-self.s[0])*(a[1]-self.s[1])-(self.t[1]-self.s[1])*(a[0]-self.s[0]))/((self.t[0]-self.s[0])*(a[1]-b[1])-(self.t[1]-self.s[1])*(a[0]-b[0]))
        if 0.0 < s and 0.0 < 1.0 and  0.0 < t and t < 1.0:
            return (self.s[0] + (s * (self.t[0] - self.s[0])), self.s[1] + (s * (self.t[1] - self.s[1])))
        return None
    
    def midpoint(self):
        return (self.s[0]+self.t[0])/2, (self.s[1]+self.t[1])/2

    def collinear(self, other):
        other = Edge(*other)
        if abs(self.direction().t[0]) == abs(other.direction().t[0]):
            other_axis = 1 if abs(self.direction().t[0]) == 1 else 0
            return self.s[other_axis] == other.s[other_axis]
        return False

    def id(self):
        return int.from_bytes(hashlib.md5(str(self).encode(), usedforsecurity=False).digest(), byteorder='big')

class EdgeSet(setdict):

    nodes: setdict

    def __init__(self, *args):
        self.nodes = setdict()
        super().__init__()
        for arg in args:
            self[arg, assign]
        
    def copy(self):
        graph = self.__class__()
        graph += list(self.nodes)
        graph += list(self)
        return graph

    def set_node(self, key):
        return self.nodes[key.copy() if isinstance(key, Node) else Node(*key), assign]

    def get_node(self, key):
        return self.nodes[key]

    def set_edge(self, key):
        edge = key.copy() if isinstance(key, Edge) else Edge(*key)
        edge = super().__getitem__(edge, op=assign)
        if edge.s not in self:
            edge.s = EdgeSet.set_node(self, edge.s)
        else:
            edge.s = self.nodes[edge.s]
        if edge.t not in self:
            edge.t = EdgeSet.set_node(self, edge.t)
        else:
            edge.t = self.nodes[edge.t]
        return edge

    def get_edge(self, key):
        # key = key if isinstance(key, Edge) else Edge(*key)
        return super().__getitem__(key, op=None)

    def __getitem__(self, key, op=None):
        keytype = Edge if list(map(lambda t: isinstance(t, tuple), key)) == [True, True] else Node
        ko = keytype, op
        if ko == (Node, assign):
            return self.set_node(key)
        elif ko == (Edge, assign):
            return self.set_edge(key)
        elif ko == (Node, None):
            return self.get_node(key)
        elif ko == (Edge, None):
            return self.get_edge(key)
 
    def del_node(self, key):
        for edge in list(self):
            if key in edge:
                del self[edge]
        del self.nodes[key]

    def del_edge(self, key):
        super().__delitem__(key)

    def __delitem__(self, key):
        keytype = Edge if list(map(lambda t: isinstance(t, tuple), key)) == [True, True] else Node
        if keytype == Node:
            self.del_node(key)
        else:
            self.del_edge(key)

    def __contains__(self, x):
        if isinstance(x, EdgeSet):
            return sum(item not in self for item in x) == 0 and sum(node not in self for node in x.nodes) == 0
        return x in self.nodes or super().__contains__(x)
    
    def __repr__(self):
        return list(self.keys()).__repr__()
    
    def __str__(self):
        return list(self.keys()).__str__()
    
    def __hash__(self):
        edges = [hash(edge) for edge in self]
        edges.sort()
        return hash(tuple(edges))

    def __eq__(self, other):
        if isinstance(other, EdgeSet):
            return other in self and self in other
        else:
            raise ValueError(f"Attempted to test if object of type {type(other)} is equal to an EdgeSet.")

    def __iadd__(self, other):
        if isinstance(other, EdgeSet):
            for node in other.nodes:
                self.nodes[node.copy(), assign]
        if isinstance(other, (EdgeSet, list)):
            for item in list(self) + list(other):
                self[item.copy(), assign]
            return self
        elif isinstance(other, Edge) or isinstance(other, Node):
            self[other.copy(), assign]
            return self
        else:
            raise ValueError(f"Attempted to add object of type {type(other)} to an EdgeSet.")

    def __isub__(self, other):
        if isinstance(other, (EdgeSet, list)):
            for item in other:
                if item in self:
                    del self[item]
            return self
        elif isinstance(other, Edge) or isinstance(other, Node): 
            del self[other]
            return self
        else:
            raise ValueError(f"Attempted to subtract object of type {type(other)} to an EdgeSet.")

    def __add__(self, other):
        if isinstance(other, (EdgeSet, Edge, Node, list)):
            graph = self.copy()
            graph += other
            return graph.copy()
        else:
            raise ValueError(f"Attempted to add object of type {type(other)} to an EdgeSet.")
        
    def __sub__(self, other):
        if isinstance(other, (EdgeSet, Edge, Node, list)):
            graph = self.copy()
            graph -= other
            return graph.copy()
            return self.__class__(*(edge for edge in self if edge not in other))
        else:
            raise ValueError(f"Attempted to subtract object of type {type(other)} to an EdgeSet.")

    def covered_nodes(self):
        covered = set()
        for edge in self:
            covered.add(edge.s)
            covered.add(edge.t)
        return covered

    def double_directed(self):
        graph = self.copy()
        for edge in list(graph):
            graph[edge.switch(), assign]
        return graph

    def single_directed(self):
        graph = self.double_directed()
        for edge in list(graph):
            if edge.double_directed().switch() in graph:
                del graph[edge.double_directed().switch()]
        return graph

    def intersect(self, other):
        graph = self.copy()
        graph += other
        for node in list(graph.nodes):
            if node not in self or node not in other:
                del graph[node]
        for edge in list(graph):
            if edge not in self or edge not in other:
                del graph[edge]
        return graph
        return self.__class__(*chain((edge for edge in self if edge in other), (edge for edge in other if edge in self)))
        return EdgeSet(*([edge for edge in self if edge in other]+[edge for edge in other if edge in self]))

    def symmetric_difference(self, other):
        return self.__class__(*chain((edge for edge in self if edge not in other), (edge for edge in other if edge not in self)))
        return EdgeSet(*([edge for edge in self if edge not in other]+[edge for edge in other if edge not in self]))

    def id(self):
        return int.from_bytes(hashlib.md5(' '.join(str(id) for id in sorted(edge.id() for edge in self)).encode(), usedforsecurity=False).digest(), byteorder='big')
        return sum(edge_hash(edge) for edge in sorted(self, key=lambda edge: edge.id())) % 10**(sys.get_int_max_str_digits()-1)

    def save_to_disk(self, filename):
        graph_dict = {
            'nodes': list(tuple(node) for node in self.nodes),
            'edges': list((tuple(edge.s), tuple(edge.t)) for edge in self)
        }
        pickle.dump(graph_dict, open(filename, 'wb'))

    def load_from_disk(filename):
        graph_dict = pickle.load(open(filename, 'rb'))
        graph = EdgeSet()
        graph += [Edge(*edge) for edge in graph_dict['edges']]
        graph += [Node(*node) for node in graph_dict['nodes']]
        return graph

class Graph(EdgeSet):

    def set_edge(self, key):
        key = EdgeSet.set_edge(self, key)
        self.union(self[key.s], self[key.t])
        return key

    def neighbors(self, node):
        return [n for n in self.nodes if (node, n) in self]

    def degree(self, node):
        return len(self.neighbors(node))

    def find(self, node):
        if node in self:
            if self[node].parent != node:
                self[node].parent = self.find(self[node].parent)
                return self[node].parent
            else:
                return self[node]
        else:
            return node

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return None
        if x.size < y.size:
            (x, y) = (y, x)
        y.parent = x
        x.size += y.size
        return x

    def component(self, node):
        component = Graph()
        for edge in self:
            if self.find(edge.s) == self.find(node):
                component += edge
        return component

    def components(self):
        components = {}
        for x in list(self) + list(self.nodes):
            parent = self.find(x) if isinstance(x, Node) else self.find(x.s)
            if parent not in components:
                components[parent] = self.__class__(x)
            else:
                components[parent] += x
        return components

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

    # Returns a cycle such that the rightwards-perpendicular direction of each edge faces inwards
    # assumes cycle is a list of nodes
    def right_faces_inwards(cycle):
        graph = Graph(*Graph.vertex_sequence_to_edges(cycle))
        edge = graph[(cycle[0], cycle[1])]
        point = Edge(edge.midpoint(), edge.parallel_right().midpoint()).midpoint()
        if not graph.test_interior(point):
            return cycle[::-1]
        return cycle

    def vertex_sequence_to_edges(sequence):
        return [(sequence[i], sequence[i+1]) for i in range(len(sequence)-1)]

    def src_sink_paths(self, src, sink, filter=lambda path: len(Graph.vertex_sequence_to_edges(path)) == len(set(Graph.vertex_sequence_to_edges(path)))):
        for paths in self.paths(src, filter=filter):
            for path in paths:
                if path[-1] == sink:
                    yield Graph.vertex_sequence_to_edges(path)

    def hamiltonian_paths(self, src):
        paths_iterator = self.paths(src)
        for paths in paths_iterator:
            for path in paths:
                if len(path) == len(self.nodes):
                    yield Graph.vertex_sequence_to_edges(path)

    def hamiltonian_cycles(self):
        for cycles in self.cycles(next(iter(self.nodes))):
            for cycle in filter(lambda cycle: len(set(cycle)) == len(set(list(self.nodes))), cycles):
                yield Graph.vertex_sequence_to_edges(cycle)

    def tours(self):
        src    = list(self.nodes)[0]
        filter = lambda path: len(Graph.vertex_sequence_to_edges(path)) == len(set(Graph.vertex_sequence_to_edges(path)))
        for paths in self.paths(src, filter=filter):
            for path in paths:
                if len(set(path)) == len(self.nodes) and path[-1] == path[0]:
                    yield path

    def travelling_salesman_tours(self):
        tlen = float('inf')
        for tour in self.tours():
            if len(tour) <= tlen:
                tlen = len(tour)
                yield Graph.vertex_sequence_to_edges(tour)
            if len(tour) > tlen:
                raise StopIteration

    def bfs(self, src):
        tree = Graph() + src
        queue = [src]
        while len(queue) > 0:
            x = queue.pop(0)
            for neighbor in self.neighbors(x):
                if neighbor not in tree:
                    tree[(x, neighbor), assign]
                    queue.append(neighbor)
        return tree

    def make_bipartite(self):
        for src in self.nodes:
            tree = self.bfs(src)
            for node in tree.nodes:
                self[node].color = len(tree.shortest_path(src, node)) % 2
        return self

    def maximum_flow(self, srcs, sinks, src_sink_weight=float('inf')):
        graph = self.copy()
        graph.antiparallel = {}
        for edge in graph.single_directed():
            if edge.switch() in graph:
                del graph[edge.switch()]
                edge_1 = graph[Edge(edge.midpoint(), edge.s), assign]
                edge_2 = graph[Edge(edge.t, edge.midpoint()), assign]
                graph.antiparallel[edge_1] = edge.switch()
                graph.antiparallel[edge_2] = edge.switch()
        graph.src = Node(None, -1)
        graph.sink = Node(None, 1)
        for src in srcs:
            graph[(graph.src, src), assign].weight = src_sink_weight
        for sink in sinks:
            graph[(sink, graph.sink), assign].weight = src_sink_weight
        flow = {
            **{edge: 0 for edge in graph}, **{edge.switch(): 0 for edge in graph}
        }
        capacity = {
            **{edge: edge.weight for edge in graph}, **{edge.switch(): edge.weight for edge in graph}
        }
        def residual_capacity(graph, edge):
            return capacity[edge] - flow[edge] if edge in graph and edge.switch() not in graph else flow[edge.switch()] if edge.switch() in graph else 0
        def residual_graph(graph):
            residual = graph.copy().double_directed()
            for edge in list(residual):
                if residual_capacity(graph, edge) == 0:
                    del residual[edge]
            return residual
        def get_augmenting_path(residual):
            path = residual.shortest_path(residual.src, residual.sink)
            if path is None:
                return None
            return [residual[edge] for edge in Graph.vertex_sequence_to_edges(path)]
        residual = residual_graph(graph)
        path = get_augmenting_path(residual)
        while path is not None:
            rc = min(residual_capacity(graph, edge) for edge in path)
            augmenting_flow = {
                **{edge: 0 for edge in flow.keys()}, **{edge: rc for edge in path} 
            }
            for edge in graph:
                flow[edge] += augmenting_flow[edge] - augmenting_flow[edge.switch()]
            residual = residual_graph(graph)
            path = get_augmenting_path(residual)
        return {
            **{edge: f for (edge, f) in flow.items() if edge not in graph.antiparallel and graph.src not in edge and graph.sink not in edge},
            **{graph.antiparallel[edge]: f for (edge, f) in flow.items() if edge in graph.antiparallel}
        }

    def flow_matching(self):
        graph = self.copy().make_bipartite()
        srcs  = [node for node in graph.nodes if node.color == 0]
        sinks = [node for node in graph.nodes if node.color == 1]
        for src in srcs:
            for sink in sinks:
                if (sink, src) in graph:
                    del graph[(sink, src)]
        matching = graph.maximum_flow(srcs, sinks, src_sink_weight=1)
        for edge in list(graph):
            if matching[edge] < 1:
                del graph[edge]
        graph.unmatched = []
        for node in graph.nodes:
            if len(graph.double_directed().neighbors(node)) != 1:
                graph.unmatched.append(node)
        return graph.double_directed()

    def matching_incompatible(self, matching):
        incompatible = EdgeSet()
        self = self.double_directed()
        for a in matching.covered_nodes():
            for b in self.nodes:
                if ((a, b) in self or (b, a) in self):
                    incompatible += [Edge(a, b), Edge(b, a)]
        return incompatible - matching

    def matching_compatible(self, matching):
        return self - self.matching_incompatible(matching)

    def match(self, include=EdgeSet(), exclude=EdgeSet(), force_include=[], force_exclude=[]):
        model = Model(name='one_factor')
        graph = self.double_directed()
        edge_vars = {
            edge: model.integer_var(name=str((edge.s, edge.t))) for edge in graph
        }
        node_vars = {
            node: [edge_vars[edge] for edge in edge_vars if node in edge] for node in graph.nodes
        }
        for edge in edge_vars:
            if edge in force_include:
                model.add_constraint(edge_vars[edge] == 1)
            elif edge in force_exclude:
                model.add_constraint(edge_vars[edge] == 0)
            else:
                model.add_constraint(edge_vars[edge] >= 0)
                model.add_constraint(edge_vars[edge] <= 1)
                model.add_constraint(edge_vars[edge] == edge_vars[edge.switch()])
        for node in node_vars:
            model.add_constraint(sum(node_vars[node]) <= 2)

        exclude_incompatible = graph.matching_incompatible(exclude)
        model.maximize(
            sum(edge_vars.values())
            +
            sum(edge_vars[edge] for edge in edge_vars if (edge in include) or (edge.switch() in include))
            +
            sum(edge_vars[edge] for edge in edge_vars if (edge in exclude_incompatible) or (edge.switch() in exclude_incompatible))
        )
        model.solve()
        return self.__class__(*([] if model.solution is None else (edge for edge in graph if model.solution[str(edge)] == 1.0))).double_directed()

    def forced_edges(self, edge):
        if not hasattr(self, '_forced_edges'):
            self._forced_edges = {}
            # self._forced_edges.copy = lambda _: {}
        if edge not in self._forced_edges:
            excl_matching = self.match(force_exclude=[edge])
            incl_matching = self.match(include=excl_matching, force_include=[edge])
            self._forced_edges[edge] = excl_matching.symmetric_difference(incl_matching)
        return self._forced_edges[edge]

    def forcing_edges(self, edge):
        forcing_set = Graph()
        for other in self:
            if edge in self.forced_edges(other):
                forcing_set += other
        return forcing_set

    def shortest_cycle_at(self, src, nontrivial=True):
        bfs_tree = self.bfs(src).double_directed()
        if len(self-bfs_tree) == 0:
            return Graph()
        def path_length(src, dest):
            p = bfs_tree.shortest_path(src, dest)
            if p is None or len(p) <= 1:
                return float('inf')
            return len(p)
        edges = sorted(self - bfs_tree, key=lambda edge: path_length(src, edge.s) + path_length(src, edge.t))
        if nontrivial:
            if path_length(src, edges[0].s) + path_length(src, edges[1].t) <= 5:
                return Graph()
            # edges = [edge for edge in edges if path_length(src, edge.s) + path_length(src, edge.t) > 5]
        if len(edges) == 0:
            return Graph()
        p1 = bfs_tree.shortest_path(src, edges[0].s)
        if p1 is None:
            return Graph()
        # p2 = (self - p1[1:-1]).shortest_path(src, edges[0].t)
        p2 = bfs_tree.shortest_path(src, edges[0].t)
        if p2 is None:
            return Graph()
        p1, p2 = Graph(*Graph.vertex_sequence_to_edges(p1)), Graph(*Graph.vertex_sequence_to_edges(p2))
        if len(p1.intersect(p2)) > 0:
            return Graph()
        sc = (p1+p2+edges[0]).double_directed()
        if nontrivial and len(sc) == 8:
            return Graph()
        return sc

    def get_subtours(self):
        sbt = []
        for node in self.nodes:
            subtour = self.shortest_cycle_at(node)
            if len(subtour) > 0 and subtour not in sbt:
                sbt.append(subtour)
        return sbt

    def minimum_spanning_tree(self, sort_edges=lambda graph, tree, edge: 1, must_include=EdgeSet()):
        tree = Graph(*must_include.double_directed())
        n = -1
        while n != len(tree):
            n = len(tree)
            try:
                edges = sorted(self, key=lambda edge: sort_edges(self, tree, edge), reverse=True)
                edge = next(edge for edge in edges if tree.find(edge.s) != tree.find(edge.t))
                tree[edge, assign]
                tree[edge.switch(), assign]
            except StopIteration:
                pass
        return tree.double_directed()

    def min_cut(self, src, dest):
        model = Model(name='min_cut')
        paths = {
            (s, t): Graph.vertex_sequence_to_edges(self.shortest_path(s, t)) for s in src for t in dest
        }
        xs = {
            edge: model.integer_var(name=str((edge.s, edge.t))) for edge in self
        }
        for edge in self:
            model.add_constraint(xs[edge] >= 0)
            model.add_constraint(xs[edge] <= 1)
        for path in paths:
            model.add_constraint(sum(xs[edge] for edge in paths[path]) >= 1)
        model.minimize(sum(xs.values()))
        model.solve()
        return self.__class__(*([] if model.solution is None else (edge for edge in self if model.solution[str(edge)] == 1.0)))

    def connect_components(self, components):
        graph = sum(components.values(), start=Graph())
        src = next(iter(components))
        sink = next(component for component in components if component != src)
        graph += self.min_cut(components[src].nodes, components[sink].nodes)
        return graph
    
    def connect_forest(self, forest):
        components = forest.components()
        while len(components) > 1:
            forest = self.connect_components(components)
            components = forest.components()
        return forest

    @property
    def two_factor(self):
        if not hasattr(self, '_two_factor'):
            model = Model(name='two_factor')
            xs = {
                edge: model.binary_var(name=str(edge)) for edge in self
            }
            xvs = {
                node: [xs[edge] for edge in xs if node in edge] for node in self.nodes
            }
            for edge in xs:
                model.add_constraint(xs[edge] + xs[edge.switch()] <= 1)
            for node in xvs:
                model.add_constraint(sum(xvs[node]) == 2)
            model.minimize(sum(xs.values()))
            model.solve()
            if model.solution is not None:
                self._two_factor = self.__class__(*(edge for edge in self if model.solution[str(edge)] == 1.0))
                self._two_factor.dual().interior = self.dual().interior - [node for node in self.dual().interior.nodes if not self._two_factor.test_interior(node)]
            else:
                self._two_factor = self.__class__()
        return self._two_factor

    @two_factor.setter
    def two_factor(self, value):
        self._two_factor = value

    def dijkstra(self, src):
        setattr(src, 'distances', {})
        D = {x: (0, x) if x == src else (float('inf'), None) for x in self.nodes}
        D |= src.distances
        Q = set(self.nodes)
        while len(Q) > 0:
            x = min(Q, key=lambda node: D[node][0])
            Q.remove(x)
            for y in self.neighbors(x):
                alt = D[x][0] + self[(x, y)].weight
                if alt < D[y][0]:
                    D[y] = (alt, x)
        src.distances |= D
        return src

    def shortest_path(self, src, dest):
        src = self.dijkstra(src)
        def has_parent(node):
            return node in src.distances and src.distances[node][1] is not None
        if has_parent(dest):
            p = [dest]
            while has_parent(p[0]) and p[0] != src:
                p.insert(0, src.distances[p[0]][1])
            return p
        else:
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

    # Uses raycasting to determine if a point lies inside a polygon.
    # Assumes self is the polygon. 
    def test_interior(self, point):
        def in_range(edge, axis=0):
            ir = lambda edge: edge.s[axis] <= point[axis] and point[axis] <= edge.t[axis]
            return ir(edge) or ir(edge.switch())
        edges = set([edge.double_directed() for edge in self.single_directed() if in_range(edge)])
        above = [edge for edge in edges if edge.s[1] >= point[1] and edge.t[1] > point[1]]
        return (len(above) % 2) != 0

    def compress(self):
        from pyzstd import compress
        return compress('\n'.join([str(edge) for edge in self]).encode('utf-8'))
        compressor = ZstdCompressor()
        for edge in self:
            compressor.compress(str(edge).encode('utf-8'))
            compressor.compress('\n'.encode('utf-8'))
        return compressor.flush()

    def decompress(self, bytes):
        import re
        from pyzstd import decompress
        r = re.compile(r'\(?\((.*?),\s*(.*?)\)\)?')
        edge_strs = decompress(bytes).decode('utf-8').split('\n')
        for edge_str in edge_strs:
            matches = r.findall(edge_str)
            if len(matches) > 0:
                nodes = [Node(float(match[0]), float(match[1])) for match in matches]
                nodes = [round(node) for node in nodes if node[0].is_integer() and node[1].is_integer()]
                if len(nodes) == 1:
                    self += nodes[0]
                else:
                    self += Edge(nodes[0], nodes[1])
        return self

class SolidGridGraph(Graph):

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def __init__(self, *args, edges=[], double_directed=True):
        super().__init__(*args, *edges)
        for edge in list(self):
            self[edge, assign]
        if double_directed:
            for edge in list(self):
                self[edge.switch(), assign]

    def make_solid(self):
        for node in self.nodes:
            for direction in SolidGridGraph.directions:
                if node + direction in self.nodes:
                    self[(node, node+direction), assign]
        return self

    def face_at_midpoint(self, point):
        points = (point[0]-0.5,point[1]-0.5), (point[0]-0.5,point[1]+0.5), (point[0]+0.5,point[1]+0.5), (point[0]+0.5, point[1]-0.5), (point[0]-0.5,point[1]-0.5)
        points = [(int(x), int(y)) for (x, y) in points]
        return self.intersect(Graph(*Graph.vertex_sequence_to_edges(points)).double_directed())

    def midpoint_at_face(self, face):
        nodes = set([edge.s for edge in face]+[edge.t for edge in face])
        if len(nodes) != 4:
            raise ValueError(f"Attempted to get midpoint at incomplete face {face}.")
        return sum(nodes, start=Node(0, 0)) * 0.25

    def to_primal_edge(edge):
        return Edge(Edge(edge.midpoint(), edge.parallel_right().midpoint()).midpoint(), Edge(edge.midpoint(), edge.parallel_left().midpoint()).midpoint())

    def to_dual_edge(edge):
        return SolidGridGraph.to_primal_edge(edge).switch()

    def dual(self, recompute=False):
        if not hasattr(self, '_dual') or recompute:
            dual = self.__class__()
            dual.interior = self.__class__()
            dual.exterior = self.__class__()
            for node in self.nodes:
                dual += list(SolidGridGraph.box(at=node).nodes)
            for node in dual.nodes:
                if len(self.intersect(SolidGridGraph.box(at=node))) == 4:
                    dual.interior[node, assign]
                else:
                    dual.exterior[node, assign]
            dual.interior.make_solid()
            dual.exterior.make_solid()
            dual.exterior -= [edge for edge in dual.exterior if SolidGridGraph.to_primal_edge(edge) in self]
            dual += dual.interior + dual.exterior
            dual.make_solid()
            setattr(self, '_dual', dual)
            return self._dual
        return self._dual

    def border(self):
        return SolidGridGraph(*[edge for edge in self if (SolidGridGraph.to_dual_edge(edge).s in self.dual().interior.nodes) != (SolidGridGraph.to_dual_edge(edge).t in self.dual().interior.nodes)])

    # check if the exterior nodes in the dual graph are connected
    def is_solid(self):
        dual = self.dual()
        return len(dual.exterior.components()) == 1

    def boundary_from_dual_tree(dual):
        graph = SolidGridGraph()
        for node in dual.nodes:
            graph += SolidGridGraph.box(at=node)
        graph = graph.double_directed()
        return graph - [SolidGridGraph.to_primal_edge(edge) for edge in dual.double_directed()]

    # Left, right, top/front, and bottom/back edges of a box oriented by a direction
    # using midpoint coordinates
    # Oriented counterclockwise
    def left_edge(direction=(0, 1), at=Node(0.5, 0.5)):
        d = Edge((0, 0), direction)
        t = Node(Edge(d.midpoint(), d.parallel_left().midpoint()).midpoint())
        return Edge(t, t - direction)+at

    def right_edge(direction=(0, 1), at=Node(0.5, 0.5)):
        d = Edge((0, 0), direction)
        t = Node(Edge(d.midpoint(), d.parallel_right().midpoint()).midpoint())
        return Edge(t - direction, t)+at
    
    def top_edge(direction=(0, 1), at=Node(0.5, 0.5)):
        return Edge(SolidGridGraph.right_edge(direction, at).t, SolidGridGraph.left_edge(direction, at).s)
    
    def bottom_edge(direction=(0, 1), at=Node(0.5, 0.5)):
        return Edge(SolidGridGraph.left_edge(direction, at).t, SolidGridGraph.right_edge(direction, at).s)

    def box(at=Node(0.5, 0.5)):
        return SolidGridGraph(*[SolidGridGraph.right_edge(direction=d, at=at) for d in SolidGridGraph.directions], double_directed=False)

    def nine_neighborhood(at=Node(0, 0)):
        neighborhood = SolidGridGraph()
        for r in range(-1, 2):
            for c in range(-1, 2):
                neighborhood += (at + (r, c))
        return neighborhood.make_solid()

    def allowed_four_fulls(self):
        allowed = []
        for node in self.nodes:
            dual_square = SolidGridGraph.box(at=node).nodes
            if sum(node in self.dual().interior for node in dual_square) == 4:
                big_component = max(self.dual().interior.components().values(), key=len)
                num_smaller = 0
                for dual_node in dual_square:
                    max_component = max((self.dual().interior - dual_node).copy().components().values(), key=len)
                    if len(max_component.nodes) < len(big_component.nodes) - 1:
                        num_smaller += 1
                if num_smaller == 4:
                    allowed.append(node)
        return allowed

    def four_full_separated_components(self):
        dual = self.dual().interior.copy()
        square_nodes = sum((SolidGridGraph.box(at=node) for node in self.allowed_four_fulls()), start=SolidGridGraph())
        return (dual - list(square_nodes.nodes)).components()

    # go through each node of each subtour in the dual.
    # if removing it decreases the size of the largest component by more than 1, it's a 'chokepoint' and we should remove it as a last resort.  
    # if removing it results in diagonal nodes, we should remove it and its non-diagonal neighbor not part of the cycle as a last resort. 
    def get_subtour_eliminations(dual):
        eliminated = {}
        def is_colinear(node):
            l, r = tour.neighbors(node)
            return l - node == node - r
        def is_chokepoint(tour, node):
            big_component_1 = max((dual-node).components().values(), key=len)
            big_component_2 = max((dual-eliminated[tour]).components().values(), key=len)
            return len(big_component_1) < len(big_component_2)
        def makes_diagonal(node):
            test_dual = dual - node
            primals = SolidGridGraph.box(at=node)
            for primal in primals.nodes:
                a, b, c, d = primal + (-0.5, -0.5), primal + (0.5, -0.5), primal + (0.5, 0.5), primal + (-0.5, 0.5)
                if sum(x in test_dual for x in [a, b, c, d]) == 2:
                    if (a in test_dual and c in test_dual):
                        return (a, c)
                    if (b in test_dual and d in test_dual):
                        return (b, d)
        for tour in dual.get_subtours():
            for node in tour.nodes:
                if is_colinear(node):
                    if tour not in eliminated:
                        eliminated[tour] = [node]
                    else:
                        if not is_chokepoint(tour, node):
                            diagonal = makes_diagonal(node)
                            previous_diagonal = makes_diagonal(eliminated[tour][0])
                            if diagonal is None or previous_diagonal is not None:
                                eliminated[tour] = [node]
                            if diagonal is not None:
                                eliminated[tour] = [eliminated[tour][0], diagonal[0]]
        return eliminated

    def eliminate_subtours(dual):
        while len(dual.get_subtours()) > 0:
            dual -= sum(SolidGridGraph.get_subtour_eliminations(dual).values(), start=[])
        return dual


    def dual_tree(self):
        model = Model(name='dual_tree')
        dual = {
            node: {
                'a': node + (-0.5, -0.5),
                'b': node + (0.5, -0.5), 
                'c': node + (0.5, 0.5), 
                'd': node + (-0.5, 0.5)
            } for node in self.nodes
        }
        xs = {
            node: model.binary_var(name=str(node)) for node in self.dual().nodes
        }
        allowed_four_fulls = self.allowed_four_fulls()
        for node in xs:
            if node not in self.dual().interior:
                model.add_constraint(xs[node] == 0)
        for node in dual.values():
            a, b, c, d = node['a'], node['b'], node['c'], node['d']
            if node not in allowed_four_fulls:
                model.add_constraint(xs[a] + xs[b] + xs[c] + xs[d] <= 3)
            model.add_constraint((xs[a] + xs[c]) - (xs[b] + xs[d]) <= 1)
            model.add_constraint((xs[b] + xs[d]) - (xs[a] + xs[c]) <= 1)
        model.maximize(sum(xs.values()))
        model.solve()
        solution_dual = SolidGridGraph()
        solution_dual += [node for node in xs if not isinstance(xs[node], int) and round(model.solution[str(node)]) == 1]
        solution_dual = SolidGridGraph.eliminate_subtours(solution_dual.make_solid())
        solution_dual = SolidGridGraph(*max(solution_dual.components().values(), key=len).double_directed())
        solution = SolidGridGraph.boundary_from_dual_tree(solution_dual.make_solid()).single_directed()
        solution.dual().interior = solution_dual
        setattr(solution, 'model', model)
        return solution

    def new_ilp(self):
        model = Model(name='new_ilp')

        dual = self.double_directed().dual()

        # Set up indicator variables for primal edges
        xuv = {
            edge: model.binary_var(name=str(edge)) for edge in self.double_directed()
        }

        # Set up indicator variables for dual nodes
        xd = {
            dual_node: model.binary_var(name=str(dual_node)) for dual_node in dual.interior.nodes
        }

        # Associate primal nodes with their incoming/outgoing edges 
        xv = {
            node: {
                'incoming': [xuv[edge] for edge in self if edge.t == node],
                'outgoing': [xuv[edge] for edge in self if edge.s == node]
            } for node in self.nodes
        }

        # Associate face edges to each dual node 
        xf = {
            dual_node: {
                'ab': xuv[round(SolidGridGraph.right_edge(direction=(0, -1), at=dual_node))], 
                'bc': xuv[round(SolidGridGraph.right_edge(direction=(1, 0), at=dual_node))],
                'cd': xuv[round(SolidGridGraph.right_edge(direction=(0, 1), at=dual_node))],
                'da': xuv[round(SolidGridGraph.right_edge(direction=(-1, 0), at=dual_node))]
            } for dual_node in dual.interior.nodes
        }

        # No doubled edges constraint
        for edge in xuv:
            model.add_constraint(xuv[edge] + xuv[edge.switch()] <= 1)

        # 2-factor constraint
        for node in xv:
            model.add_constraint(sum(xv[node]['incoming']) <= 1)
            model.add_constraint(sum(xv[node]['outgoing']) <= 1)
            model.add_constraint(sum(xv[node]['incoming']) == sum(xv[node]['outgoing']))

        # Boundary orientation constraint
        for (u, v) in dual:
            if u in dual.interior and v in dual.exterior:
                # Set clockswise edges to 0
                model.add_constraint(xuv[SolidGridGraph.to_primal_edge(Edge(v, u))] == 0)

        for face in xf:
            # Face implies oriented edge
            model.add_constraint(xd[face] <= sum(xf[face].values()))
            for edge in xf[face]:
                # Oriented edge implies face
                model.add_constraint(xf[face][edge] <= xd[face])

        # Whoops constraint
        for A in xf:
            for B in dual.interior.neighbors(A):
                # A and B share 1 doubled edge.
                # Say it's the rightmost for face A, leftmost for B
                # Going CCW A's oriented edge goes up, B's goes down 
                # The twin to B's oriented edge is A's oriented edge
                # Constraint: if A chooses oriented edge, B can't choose oriented edge s.t. its twin is on A
                # So if A's oriented edge gets chosen, B can't choose its oriented edge
                # Recall: an oriented edge being chosen implies the face orienting that edge is chosen
                # So the constraint is if A's oriented edge gets chosen, B can't be chosen
                A_oriented_edge = SolidGridGraph.to_primal_edge(Edge(A, B))
                model.add_constraint(xuv[A_oriented_edge] + xd[B] <= 1)

        model.maximize(
            sum(xd.values())
            +
            sum(xuv.values())
        )
        model.solve()

        pre_subtour_elimination = SolidGridGraph(*([] if model.solution is None else (edge for edge in xuv if round(model.solution[str(edge)]) == 1)))
        solution_dual = dual.interior - [node for node in dual.interior.nodes if not pre_subtour_elimination.test_interior(node)]
        solution_dual = SolidGridGraph.eliminate_subtours(solution_dual.make_solid())
        solution_dual = SolidGridGraph(*max(solution_dual.components().values(), key=len).double_directed())
        solution = SolidGridGraph.boundary_from_dual_tree(solution_dual.make_solid()).single_directed()
        solution.dual().interior = solution_dual
        setattr(solution, 'model', model)
        return solution


    # combination of the old 2x2 square ILP and new oriented tour ILP from J. Fix
    def combination_ilp(self, subtours=[]):
        model = Model(name='dual_tree')
        # for param in model.parameters.mip.cuts.generate_params():
        #     # print(param.cpx_name)
        #     if not ('FRACCUTS' in param.cpx_name and 'LOCALIMPLBD' not in param.cpx_name):
        #         param.set(-1)
            
        # Set up indicator variables for dual nodes
        xd = {
            dual_node: model.binary_var(name=str(dual_node)) for dual_node in self.dual().nodes
        }

        # Set up indicator variables for type III cells
        xt = {
            dual_node: {
                'x': model.binary_var(name=f'{str(dual_node)}_t3_x'),
                'y': model.binary_var(name=f'{str(dual_node)}_t3_y')
            } for dual_node in self.dual().interior.nodes
        }

        # Set up indicator variables for primal edges
        xuv = {
            edge: model.binary_var(name=str(edge)) for edge in self.double_directed()
        }

        # Associate primal nodes with their incoming/outgoing edges 
        xv = {
            node: {
                'incoming': [xuv[edge] for edge in self if edge.t == node],
                'outgoing': [xuv[edge] for edge in self if edge.s == node]
            } for node in self.nodes
        }

        # Associate each dual node with the edges of its face 
        xf = {
            dual_node: {
                'ab': xuv[round(SolidGridGraph.left_edge(at=dual_node))], 
                'bc': xuv[round(SolidGridGraph.top_edge(at=dual_node))],
                'cd': xuv[round(SolidGridGraph.right_edge(at=dual_node))],
                'da': xuv[round(SolidGridGraph.bottom_edge(at=dual_node))]
            } for dual_node in self.dual().interior.nodes
        }

        # Associate each primal node with the 2x2 square centered by it
        dual = {
            node: {
                'a': node + (-0.5, -0.5),
                'b': node + (0.5, -0.5), 
                'c': node + (0.5, 0.5), 
                'd': node + (-0.5, 0.5)
            } for node in self.nodes
        }

        # Dual nodes on the exterior are not part of the footprint (as J calls it) of the cycle 
        for node in xd:
            if node not in self.dual().interior:
                model.add_constraint(xd[node] == 0)

        # 2x2 square constraints
        for node in dual.values():
            a, b, c, d = node['a'], node['b'], node['c'], node['d']
            model.add_constraint(xd[a] + xd[b] + xd[c] + xd[d] <= 3)
            model.add_constraint((xd[a] + xd[c]) - (xd[b] + xd[d]) <= 1)
            model.add_constraint((xd[b] + xd[d]) - (xd[a] + xd[c]) <= 1)

        # No doubled edge constraint
        for edge in xuv:
            model.add_constraint(xuv[edge] + xuv[edge.switch()] <= 1)

        # 2-factor constraint
        for node in xv:
            model.add_constraint(sum(xv[node]['incoming']) <= 1)
            model.add_constraint(sum(xv[node]['outgoing']) <= 1)
            model.add_constraint(sum(xv[node]['incoming']) == sum(xv[node]['outgoing']))

        # Boundary CCW orientation constraint 
        # for (u, v) in dual:
        #     if u in self.dual().interior and v in self.dual().exterior:
        #         model.add_constraint(xuv[SolidGridGraph.to_primal_edge(Edge(v, u))] == 0)

        for face in xf:
            model.add_constraint(xd[face] <= sum(xf[face].values()))
            # Interior face implies at least one oriented edge
            # model.add_constraint(4 * xd[face] <= 4 * sum(xf[face].values()) + sum(xd[face+d] for d in SolidGridGraph.directions))
            for edge in xf[face]:
                # Oriented edge implies the face that orients it
                model.add_constraint(xf[face][edge] <= xd[face])

        for A in xf:
            for B in self.dual().interior.neighbors(A):
                # Faces can't be neighbors if separated by an oriented edge
                A_oriented_edge = SolidGridGraph.to_primal_edge(Edge(A, B))
                B_oriented_edge = SolidGridGraph.to_primal_edge(Edge(B, A))
                model.add_constraint(xuv[A_oriented_edge] + xd[B] <= 1)
                # Faces must be neighbors if no oriented edge between them exists
                # model.add_constraint(xd[B] >= xd[A] - (xuv[A_oriented_edge] + xuv[B_oriented_edge]))
                model.add_constraint((xd[A] - xd[B]) - (xuv[A_oriented_edge] + xuv[B_oriented_edge]) <= 0)


        # Type III cell constraints
        for node in xt:
            # Only count Type IIIs along one axis 
            model.add_constraint(xt[node]['y'] + xt[node]['x'] <= 1 - xd[node])
            for d in ['x', 'y']:
                # X Y Z <- if Y is 0 and X and Z are 1, Y is a type III cell - count it
                #          if X or Z are 0 then Y is not a type III cell
                c = (0, 1) if d == 'y' else (1, 0)
                model.add_constraint(xt[node][d] <= xd[node + c])
                model.add_constraint(xt[node][d] <= xd[node - c])

        # n = 2k + 2 (n := number of edges, k := number of dual nodes)
        #    This is true for Hamiltonian cycles
        model.add_constraint(sum(xuv.values()) == 2 * sum(xd.values()) + 2)

        for subtour in subtours:
            model.add_constraint(sum(xd[node] for node in subtour.nodes) <= len(subtour.nodes)-1)

        model.maximize(
            # Maximize edges (i.e covered nodes)
            # Minimize footprint; there are cases where adding 1 dual node takes away 1 edge
            #   We always want to add edges over dual nodes
            sum(xuv.values()) - sum(xd.values())
            +
            # Maximize the number of type 3 cells so the resulting graph is a Hamiltonian SGG
            sum(xt[node]['y'] + xt[node]['x'] for node in xt)
        )

        model.solve()
        # print(model.get_cuts())
        # print(model.solve_details)
        # print(model.solve_details.dettime)
        # visualization gobbledygook
        solution_dual = SolidGridGraph() + ([] if model.solution is None else [node for node in xd if model.solution[str(node)] > 0.0])
        solution = SolidGridGraph() + ([] if model.solution is None else [edge for edge in xuv if model.solution[str(edge)] > 0.0])
        # connects all adjacent neighbors in the dual together
        solution.dual().interior = solution_dual.make_solid()
        setattr(solution, 'model', model)
        return solution

    def longest_cycle(self):
        # return self.combination_ilp().dual().interior
        combo = self.combination_ilp()
        subtours = []
        current_subtours = combo.dual().interior.get_subtours()
        while len(current_subtours) > 0:
            subtours += current_subtours
            combo = self.combination_ilp(subtours=subtours)
            current_subtours = combo.dual().interior.get_subtours()
        return combo.dual().interior

    def test(self, k=None):
        model = Model(name='test')
        graph = self.double_directed()
        boundary = SolidGridGraph(*[edge for edge in graph if (SolidGridGraph.to_dual_edge(edge).s in graph.dual().interior.nodes) != (SolidGridGraph.to_dual_edge(edge).t in graph.dual().interior.nodes)])

        xs = {
            edge: model.binary_var(name=str(edge)) for edge in graph
        }

        xvs = {
            v: {
                'incoming': [xs[edge] for edge in xs if v == edge.t],
                'outgoing': [xs[edge] for edge in xs if v == edge.s],
            } for v in graph.nodes
        }

        xv = {
            v: sum(xvs[v]['incoming']+xvs[v]['outgoing']) for v in graph.nodes
        }

        ccw = {
            face: model.binary_var(name=f'{str(face)}_ccw') for face in self.dual().interior.nodes
        }

        cw = {
            face: model.binary_var(name=f'{str(face)}_cw') for face in self.dual().interior.nodes
        }

        interior = {
            face: model.binary_var(name=f'{str(face)}_interior') for face in self.dual().interior.nodes
        }

        xf = {
            edge: model.integer_var(name=f'{str(edge)}_flow') for edge in xs
        }

        for edge in xf:
            model.add_constraint(xs[edge] <= xf[edge])
            model.add_constraint(xf[edge] <= len(graph.nodes) * xs[edge], f'{edge}_capacity')

        distinguished = next(node for node in graph.nodes if node not in boundary.nodes)

        for node in xv:
            x = xv[node]/2
            incoming = [edge for edge in xf if edge.t == node]
            outgoing = [edge for edge in xf if edge.s == node]
            if node == distinguished:
                model.add_constraint(sum(xf[i] for i in incoming) == 1, f'distinguished_{node}_flow')
                model.add_constraint(sum(xf[o] for o in outgoing) == sum(xs.values()), f'distinguished_{node}_flow')
            else:
                model.add_constraint(sum(xf[o] for o in outgoing) == sum(xf[i] for i in incoming) - x, f'{node}_flow')

        for edge in graph:
            x, y = SolidGridGraph.to_dual_edge(edge)
            if x in graph.dual().exterior and y in graph.dual().interior:
                model.add_constraint(xs[edge] == 0)
        for edge in xs:
            model.add_constraint(xs[edge] + xs[edge.switch()] <= 1)
        for node in xvs:
            model.add_constraint(sum(xvs[node]['incoming']) <= 1)
            model.add_constraint(sum(xvs[node]['outgoing']) <= 1)
            model.add_constraint(sum(xvs[node]['incoming']) == sum(xvs[node]['outgoing']))

        if k is not None:
            for face in self.dual().interior.nodes:
                # A cell can't be CCW and CW 
                model.add_constraint(cw[face] + ccw[face] <= 1)
                # CCW cells
                model.add_constraint(ccw[face] <= sum(xs[edge] for edge in SolidGridGraph.box(at=face)))
                model.add_constraint(sum(xs[edge] for edge in SolidGridGraph.box(at=face)) <= 4*ccw[face])
                # CW cells
                model.add_constraint(cw[face] <= sum(xs[edge.switch()] for edge in SolidGridGraph.box(at=face)))
                model.add_constraint(sum(xs[edge.switch()] for edge in SolidGridGraph.box(at=face)) <= 4*cw[face])
                # CCW implies interior unless it's also CW
                model.add_constraint(ccw[face] - cw[face] <= interior[face])
                model.add_constraint(interior[face] <= ccw[face])
                # Face with no edges is sometimes on the interior
                model.add_constraint(sum(interior[n] for n in graph.dual().interior.neighbors(face)) - 3 <= interior[face])

            boundary_nodes = [sum(xvs[node]['incoming'] + xvs[node]['outgoing']) / 2 for node in boundary.nodes]
            model.add_constraint(sum(boundary_nodes) == len(boundary.nodes)-k)
            model.add_constraint(sum(xs.values()) == len(graph.nodes)-k)

        # model.add_constraint(sum(xs.values()) == 2*sum(interior.values())+2)
        model.maximize(sum(xs.values()))
        model.solve()

        solution = SolidGridGraph(*([] if model.solution is None else [edge for edge in xs if model.solution[str(edge)] > 0.0]), double_directed=False)
        solution.dual().interior = SolidGridGraph(*[node for node in graph.dual().interior.nodes if model.solution[f'{str(node)}_interior'] > 0]).make_solid()
        solution.model = model
        solution.distinguished = distinguished
        return solution

    def make_alternating_cycle(start, end):
        if start == end:
            return SolidGridGraph()
        step = round(Edge(start, end).direction())
        nodes = [start]
        boxes = SolidGridGraph()
        while nodes[-1] != end:
            boxes += SolidGridGraph.box(at=nodes[-1])
            nodes.append(nodes[-1] + step.t)
        cycle = boxes.border()
        edge = SolidGridGraph.bottom_edge(direction=step.t, at=start)
        edges = [edge]
        while len(edges) < len(cycle.nodes):
            cycle -= edges[-1].switch()
            edges.append(Edge(edges[-1].t, cycle.neighbors(edges[-1].t)))
        alternating_cycle = SolidGridGraph(*edges[::2])
        alternating_cycle.direction = step.t
        alternating_cycle.ac_nodes = nodes[:-1]
        alternating_cycle.flipped = SolidGridGraph(*edges[1::2])
        alternating_cycle.flipped.direction = step.t
        alternating_cycle.flipped.ac_nodes = nodes[:-1]
        return alternating_cycle

    def make_alternating_strip(start, end):
        alternating_cycle = SolidGridGraph.make_alternating_cycle(start, end)
        alternating_cycle += SolidGridGraph(*[SolidGridGraph.top_edge(alternating_cycle.direction, at=alternating_cycle.direction+node) for node in alternating_cycle.ac_nodes[::2][:-1]])
        return alternating_cycle

    def G_F_minus(self, F):
        dual = self.dual().copy()
        for edge in F.double_directed():
            dual -= SolidGridGraph.to_dual_edge(edge)
        return SolidGridGraph(*dual)
    
    # def get_alternating_strips(self, F):


    def get_alternating_strip(self, edge):
        edge = edge.copy()
        sequence = [edge]
        perimeter = [edge]
        flipped_perimeter = []
        start = edge.__class__(edge.midpoint(), edge.parallel_right().midpoint()).midpoint()
        def isin(box):
            return sum(e in self for e in box) == len(box)
        def n_box(edge):
            return edge.take_right().go_forward(), edge.switch().take_left().go_forward().switch(), edge.parallel_right().parallel_right()
        while edge.parallel_right().parallel_right() in self and isin(n_box(edge)):
            upper, lower, right = n_box(edge)
            perimeter += [upper, lower]
            flipped_perimeter += [upper.switch().go_forward().switch(), lower.go_forward()]
            sequence += [upper, lower, right]
            edge = edge.parallel_right().parallel_right()
        if edge.parallel_right() in self:
            sequence.append(edge.parallel_right())
            perimeter += [edge.parallel_right()]
            flipped_perimeter += [edge.take_right(), edge.switch().take_left().switch()]
            strip = self.__class__(*sequence)
            setattr(strip, 'odd', True)
        else:
            sequence = sequence[:-1]
            strip = self.__class__(*sequence)
            setattr(strip, 'even_edges', sequence[-2:])
            setattr(strip, 'odd', False)
        if len(strip) > 0:
            setattr(strip, 'start', start)
            setattr(strip, 'end', edge.__class__(sequence[-1].midpoint(), sequence[-2].midpoint()).midpoint())
            setattr(strip, 'edge', sequence[0])
            setattr(strip, 'perimeter', self.__class__(*perimeter))
            setattr(strip, 'flipped_perimeter', self.__class__(*flipped_perimeter))
        return strip.double_directed()

    def get_alternating_strips(self):
        # dual = SolidGridGraph.dual(self)
        alternating_strips = []
        alternating_strip_edges = self.__class__()
        for edge in self:
            x1, y1 = edge.midpoint()
            x2, y2 = edge.parallel_right().midpoint()
            if not self.two_factor.test_interior(((x1+x2)/2, (y1+y2)/2)):
                strip = SolidGridGraph.get_alternating_strip(self.two_factor, edge)
                if len(strip) > 1 and strip not in alternating_strips:
                    alternating_strips.append(strip)
                    alternating_strip_edges += strip
        for strip in alternating_strips:
            setattr(strip, 'chain', strip.edge.take_right() in alternating_strip_edges or strip.edge.switch().take_left() in alternating_strip_edges)
        return alternating_strips
    
    def edge_flip(self, strip):
        # graph = self.copy()
        # perimeter = SolidGridGraph.get_perimeter_of_alternating_strip(strip)
        # graph.two_factor = (self.two_factor - perimeter) + SolidGridGraph.flip_perimeter_of_alternating_strip(perimeter)
        # graph.two_factor = (self.two_factor - strip.perimeter) + strip.flipped_perimeter
        self.two_factor -= strip.perimeter
        self.two_factor += strip.flipped_perimeter
        return self
    
    def static_alternating_strip(self):
        alternating_strips = SolidGridGraph.get_alternating_strips(self)
        dual = SolidGridGraph.dual(self)
        H = Graph()
        for strip in alternating_strips:
            H[strip.start, assign]
            H[strip.end, assign]
        for strip in alternating_strips:
            start = H[strip.start]
            end = H[strip.end]
            setattr(start, 'chain', strip.chain)
            setattr(end, 'odd', strip.odd)
            edge = Edge(start, end)
            setattr(edge, 'weight', len(strip) // 2)
            H[edge, assign]
        for strip in alternating_strips:
            if not strip.odd:
                even_edges = strip.even_edges
                c = Edge(even_edges[0].midpoint(), even_edges[1].midpoint()).midpoint()
                c1 = Edge(c, even_edges[0].midpoint()).go_forward().t
                c2 = Edge(c, even_edges[1].midpoint()).go_forward().t
                setattr(strip, 'c', c)
                setattr(strip, 'c1', c1)
                setattr(strip, 'c2', c2)
                for other in alternating_strips:
                    if not other.chain:
                        if other.start in dual.shortest_path(c1, c2):
                            edge = Edge(c, other.start)
                            setattr(edge, 'weight', 0)
                            H[edge, assign]
                    else:
                        if len(dual.shortest_path(c1, c2)) > 0 and other.start in (c1, c2):
                            edge = Edge(c, other.start)
                            setattr(edge, 'weight', 0)
                            H[edge, assign]
        # all_pairs_shortest_path = {
        #     node: H.dijkstra(node) for node in H.nodes
        # }
        begins = [node for node in H.nodes if hasattr(node, 'chain') and not node.chain]
        odds = [node for node in H.nodes if hasattr(node, 'odd') and node.odd]
        paths = sum([[H.shortest_path(begin, odd) for begin in begins] for odd in odds], [])
        paths = [path for path in paths if path is not None]
        def pathlen(path):
            return sum(0 if path[i] == path[i+1] else H[(path[i], path[i+1])].weight for i in range(len(path)-1))
        static_strip = Graph()
        if len(paths) > 0:
            path = min(paths, key=pathlen)
            for i in range(len(path)-1):
                for strip in alternating_strips:
                    if path[i] == strip.start and path[i+1] == strip.end and not self.two_factor.test_interior(strip.start):
                        static_strip += strip
        return static_strip

    def reduce_two_factor(self):
        static_strip = self.static_alternating_strip()
        if len(static_strip) == 0:
            return None
        else:
            return self.edge_flip(static_strip)

    def has_hc(self):
        next = self.reduce_two_factor()
        if len(self.__class__(*self.two_factor.as_cycle_facing_inwards()).nodes) == len(self.nodes):
            return True
        if next is None:
            return False
        return self.__class__.has_hc(next)

    def random_thick_solid_grid_graph(n=10):
        graph = SolidGridGraph() + [round(edge) for edge in SolidGridGraph.box(at=Node(n+0.5, n+0.5)).double_directed()]
        while len(graph.dual(recompute=True).interior.nodes) < n:
            is_solid = False
            nodes = list(graph.dual().interior.nodes)
            random.shuffle(nodes)
            while not is_solid and len(nodes) > 0:
                node = nodes.pop()
                directions = SolidGridGraph.directions.copy()
                random.shuffle(directions)
                while not is_solid and len(directions) > 0:
                    d = directions.pop()
                    face = [round(edge) for edge in SolidGridGraph.box(at=node+d).double_directed()]
                    is_solid = (graph + face).is_solid()
                    if is_solid:
                        graph += face
        anchor = Node(min(list(graph.nodes), key=lambda node: node[0])[0], min(list(graph.nodes), key=lambda node: node[1])[1])
        return SolidGridGraph(*[edge - anchor for edge in graph])
        return graph

# takes every possible combination of possible pairs of nodes
def SGG_iterator(nodes, n_edges=0):
    def possible_edges(nodes):
        for node in nodes.copy():
            for direction in SolidGridGraph.directions:
                if node + direction in nodes:
                    yield (node, node+direction)
    yield from combinations(possible_edges(nodes), n_edges)


