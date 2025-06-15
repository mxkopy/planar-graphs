import random, copy
from typing import Union, TypeVar, Generic
from math import atan2, degrees, fmod, sqrt, pow, sin, cos, radians
from itertools import chain, combinations
from docplex.mp.model import Model

TEST = True

def diff_update_dict(dict_a, dict_b):
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
        node.edges = set()
        node.parent = node
        node.size = 1
        return node

    def copy(self):
        node = self.__class__(*self)
        diff_update_dict(node.__dict__, self.__dict__)
        return node

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
            return Node(self[i]/o for i, o in enumerate(other))
        else:
            return Node(i/other for i in self)

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
    
    def neighbors(self):
        yield from (edge.t for edge in self.edges if edge.s == self)

    def component(self):
        component = set([self])
        n = -1
        while n != len(component):
            n = len(component)
            for node in component.copy():
                for neighbor in node.neighbors():
                    component.add(neighbor)
        return component

class Edge:

    s: Node
    t: Node

    def __init__(self, s, t):
        self.s = s if isinstance(s, Node) else Node(*s)
        self.t = t if isinstance(t, Node) else Node(*t)
        self.s.edges.add(self)
        self.t.edges.add(self)
        self.weight = 1

    def copy(self):
        edge = self.__class__(self.s.copy(), self.t.copy())
        diff_update_dict(edge.__dict__, self.__dict__)
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

    def undirected(self):
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
        return self.__class__(self.s-self.s, self.t-self.s)
    
    def rotate(self, degrees):
        degrees = radians(degrees)
        u = self.direction()
        v = u.t*(cos(degrees), sin(degrees))
        w = u.t*(sin(degrees), cos(degrees))
        d = (v[0]-v[1], w[0]+w[1])
        if isinstance(self.s[0], int):
            d = round(d[0]), round(d[1])
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


#TODO: add nodeset
class EdgeSet(setdict):

    nodes: setdict
    
    def __init__(self, *args):
        self.nodes = setdict()
        super().__init__()
        for arg in args:
            self[arg, assign]
        
    def copy(self):
        graph = super().copy()
        for node in list(graph.nodes):
            graph.nodes[node.copy(), assign]
        for edge in list(graph):
            graph[edge.copy(), assign]
        return graph

    def set_node(self, key):
        return self.nodes[key.copy() if isinstance(key, Node) else Node(*key), assign]

    def get_node(self, key):
        return self.nodes[key]

    def set_edge(self, key):
        edge = key.copy() if isinstance(key, Edge) else Edge(*key)
        edge = super().__getitem__(edge, op=assign)
        edge.s = self.set_node(edge.s) if edge.s not in self else self.nodes[edge.s]
        edge.t = self.set_node(edge.t) if edge.t not in self else self.nodes[edge.t]
        edge.s.edges.add(edge)
        edge.t.edges.add(edge)
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
        for edge in list(self[key].edges):
            self.nodes[key].edges.remove(edge)
            del self[edge]
        del self.nodes[key]

    def del_edge(self, key):
        for node in self.nodes:
            if key in self.nodes[node].edges:
                self.nodes[node].edges.remove(key)
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
            return graph
        else:
            raise ValueError(f"Attempted to add object of type {type(other)} to an EdgeSet.")
        
    def __sub__(self, other):
        if isinstance(other, (EdgeSet, Edge, Node, list)):
            graph = self.copy()
            graph -= other
            return graph
            return self.__class__(*(edge for edge in self if edge not in other))
        else:
            raise ValueError(f"Attempted to subtract object of type {type(other)} to an EdgeSet.")

    def covered_nodes(self):
        covered = set()
        for edge in self:
            covered.add(edge.s)
            covered.add(edge.t)
        return covered

    def undirected(self):
        graph = self.copy()
        for edge in list(graph):
            graph[edge.switch(), assign]
        return graph

    def remove_directed(self):
        graph = self.undirected()
        for edge in list(graph):
            if edge.undirected().switch() in graph:
                del graph[edge.undirected().switch()]
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

class Undirected:

    def neighbors(self):
        pass

class Graph(EdgeSet):

    # def set_node(self, key):
    #     key = EdgeSet.set_node(self, key)
    #     return self.dijkstra(key)

    def neighbors(self, node):
        # return [edge.t for edge in node.edges]
        return [n for n in self.nodes if (node, n) in self]

    def degree(self, node):
        return len(self.neighbors(node))

    def component(self, node):
        component = Graph(node)
        n = -1
        while n != len(component):
            n = len(component)
            for node in list(component.nodes):
                for neighbor in self.neighbors(node):
                    component += self[(node, neighbor)]
        return component

    def components(self):
        graph = self.copy()
        for node in graph.nodes:
            setattr(node, 'parent', node)
            setattr(node, 'size', 0)
        for edge in graph:
            graph.union(edge.s, edge.t)
        components = {}
        for edge in graph:
            parent = graph.find(edge.s)
            if parent not in components:
                components[parent] = Graph(edge)
            else:
                components[parent] += edge
        return components

    def find(self, node):
        # if not hasattr(self[node], 'parent') or not hasattr(self[node], 'size'):
        #     setattr(self[node], 'parent', self[node])
        #     setattr(self[node], 'size', 1)
        if self[node].parent != self[node]:
            self[node].parent = self.find(node.parent)
            return node.parent
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
        setattr(src, 'bfs_time', 0)
        queue = [src]
        discovered = set()
        while len(queue) > 0:
            x = queue.pop(0)
            for neighbor in self.neighbors(x):
                setattr(neighbor, 'bfs_time', x.bfs_time+1)
                if neighbor not in discovered:
                    queue.append(neighbor)
                    discovered.add(neighbor)
        return self

    def make_bipartite(self):
        for node in self.nodes:
            self.bfs(node)
        for node in self.nodes:
            node.color = node.bfs_time % 2
        return self

    def maximum_flow(self, srcs, sinks, src_sink_weight=float('inf')):
        graph = self.copy()
        graph.antiparallel = {}
        for edge in graph.remove_directed():
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
            residual = graph.copy().undirected()
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
            if len(graph.undirected().neighbors(node)) != 1:
                graph.unmatched.append(node)
        return graph.undirected()

    def matching_incompatible(self, matching):
        incompatible = EdgeSet()
        self = self.undirected()
        for a in matching.covered_nodes():
            for b in self.nodes:
                if ((a, b) in self or (b, a) in self):
                    incompatible += [Edge(a, b), Edge(b, a)]
        return incompatible - matching

    def matching_compatible(self, matching):
        return self - self.matching_incompatible(matching)

    def covering_matchings(self, graph):
        matchings = []
        while sum(matchings, start=Graph()) != graph:
            matchings.append((graph-sum(matchings, start=Graph())).flow_matching())
        return matchings

    def match(self, include=EdgeSet(), exclude=EdgeSet(), force_include=[], force_exclude=[]):
        model = Model(name='one_factor')
        edge_vars = {}
        node_vars = {node: [] for node in self.nodes}
        graph = self.remove_directed()
        undirected = graph.undirected()
        for edge in graph:
            id = str((edge.s, edge.t))
            edge_vars[edge] = model.integer_var(name=id)
            node_vars[edge.s].append(edge_vars[edge])
            node_vars[edge.t].append(edge_vars[edge])
        for edge in edge_vars:
            if edge in force_include:
                model.add_constraint(edge_vars[edge] == 1)
            elif edge in force_exclude:
                model.add_constraint(edge_vars[edge] == 0)
            else:
                model.add_constraint(edge_vars[edge] >= 0)
                model.add_constraint(edge_vars[edge] <= 1)
        for node in node_vars:
            model.add_constraint(sum(node_vars[node]) <= 1)
        include_incompatible = undirected.matching_incompatible(include)
        exclude_incompatible = undirected.matching_incompatible(exclude)
        model.maximize(sum(edge_vars.values())+sum(edge_vars[edge] for edge in edge_vars if edge not in include_incompatible)+sum(edge_vars[edge] for edge in edge_vars if edge in exclude_incompatible))
        model.solve()
        return self.__class__(*([] if model.solution is None else (edge for edge in graph if model.solution[str(edge)] == 1.0))).undirected()

    def mfs_matchings(self):
        graph = self.undirected()
        mfs = graph.minimum_feedback_set()
        blue = graph.perfect_match(include=mfs)
        lc = graph.longest_alternating_cycle([blue])
        red = graph.perfect_match(include=lc-lc.intersect(blue), exclude=blue)
        return blue, red

    def experimental_extend(self, blue, red):
        blue = self.flip_matching_with_cycle(blue, self.longest_alternating_cycle([blue, red]))
        blue, lc = max((blue, self.longest_alternating_cycle([blue])), (red, self.longest_alternating_cycle([red])), key=lambda m: len(m[1]))
        red = self.match(include=lc-lc.intersect(blue), exclude=red.intersect(blue))
        return blue, red

    def component_connections(self, graph):
        connections = Graph()
        graph = graph.copy()
        for edge in graph:
            graph.union(edge.s, edge.t)
        for edge in self:
            if edge.s in graph and edge.t in graph:
                if graph[edge.s].parent != graph[edge.t].parent:
                    connections += edge
        return connections.undirected()

    def _longest_alternating_cycle_at(self, start, matchings):
        def alternates(edge_1, edge_2):
            for matching in matchings:
                if (edge_1 in matching) == (edge_2 not in matching):
                    return True
            return False
        def dfs(u, v, cycle, longest):
            for n in self.neighbors(v):
                if n == start and len(cycle)+1>len(longest):
                    longest = cycle + [n]
                if alternates((u, v), (v, n)) and n not in cycle:
                    longest = dfs(v, n, cycle + [n], longest)
            return longest
        return max(*(dfs(start, neighbor, [start, neighbor], []) for neighbor in self.neighbors(start)), key=len)

    def longest_alternating_cycle_at(self, start, matching, other_matching):
        def alternates(edge_1, edge_2):
            if (edge_1 in matching) and (edge_2 in other_matching):
                return True
            if (edge_2 in matching) and (edge_1 in other_matching):
                return True
            return False
        def is_tree(cycle):
            cycle = Graph.right_faces_inwards(cycle)
            edges = [Edge(*edge) for edge in Graph.vertex_sequence_to_edges(cycle)]
            tree = Graph()
            for edge in edges:
                tree += Node(Edge(edge.midpoint(), edge.parallel_right().midpoint()).midpoint())
            tree = SolidGridGraph.make_solid(tree)
            return tree.minimum_spanning_tree() == tree
        def dfs(u, v, cycle, longest):
            for n in self.neighbors(v):
                if is_tree(cycle+[n]):
                    if n == start and len(cycle)+1>len(longest):
                        longest = cycle + [n]
                    if alternates((u, v), (v, n)) and n not in cycle:
                        longest = dfs(v, n, cycle+[n], longest)
            return longest
        longest = []
        for neighbor in self.neighbors(start):
            cycle = dfs(start, neighbor, [start, neighbor], [])
            if len(cycle) > len(longest):
                longest = cycle
        return longest

    def longest_alternating_cycle(self, matching, other_matching=None):
        if other_matching is None:
            other_matching == self - matching
        cycle = max(*(self.longest_alternating_cycle_at(node, matching, other_matching) for node in self.nodes), key=len)
        edges = [self[edge] for edge in Graph.vertex_sequence_to_edges(cycle)]
        cycle = Graph() + edges
        setattr(cycle, 'edges', edges)
        return cycle.undirected()

    def longest_cycle(self):
        mfs = self.minimum_feedback_set()
        mfs_matching_1 = self.match(include=mfs)
        lc = self.longest_alternating_cycle(mfs_matching_1)
        mfs_matching_2 = self.match(include=lc-lc.intersect(mfs_matching_1), exclude=lc.intersect(mfs_matching_1))
        return self.longest_alternating_cycle(mfs_matching_2)

    def flip_matching(self, cycle, matching):
        return self.match(force_include=cycle-cycle.intersect(matching), include=matching)

    def pendant_nodes(self):
        graph = self.undirected()
        mfs = graph.minimum_feedback_set()
        m1 = graph.match(force_include=mfs.flow_matching())
        m2 = graph.match(force_include=(mfs-mfs.flow_matching()).flow_matching())
        if len(graph.nodes) != len(m1.covered_nodes()) and len(m1.covered_nodes()) == len(m2.covered_nodes()):
            return [node for node in graph.nodes if node not in m1.covered_nodes()]
        return []

    def minimum_spanning_tree(self, must_include=EdgeSet()):
        graph = self.copy()
        tree = Graph() + must_include
        n = -1
        for edge in tree:
            graph.union(graph[edge].s, graph[edge].t)
        while n != len(tree):
            n = len(tree)
            edges = list(graph)
            edges.sort(key=lambda edge: tree.degree(edge.s) == 3 or tree.degree(edge.t) == 3)
            for edge in edges:
                if not graph.find(edge.s) == graph.find(edge.t):
                    graph.union(edge.s, edge.t)
                    tree[graph[edge], assign]
        return tree.undirected()

    def minimum_feedback_set(self):
        return self.undirected() - self.minimum_spanning_tree().undirected()

    def get_two_factorable(self):
        graph = self.copy()
        uot = graph.union_of_tours()
        # coerce matching:
        # subtract uot from graph
        # match
        # subtract from graph, take intersection with uot
        # in general: coerce(set) = graph - (graph - set).match()
        potential_intersection = lambda graph, set: (graph - (graph - set).flow_matching()).flow_matching().intersect(set)
        # graph - (uot - graph.flow_matching())
        print(len(potential_intersection(graph, uot)))
        print(len(uot))
        return potential_intersection(graph, uot)
        # matching = self.flow_matching()
        # without_unmatched = self.copy()
        # for node in matching.unmatched:
            # del without_unmatched[node]
        # matching = without_unmatched.flow_matching()
        # for node in matching.unmatched:
            # del without_unmatched[node]
        # return without_unmatched.flow_matching()

    @property
    def two_factor(self):
        if not hasattr(self, '_two_factor'):
            model = Model(name='two_factor')
            xs = {}
            xvs = {node: [] for node in self.nodes}
            graph = self.remove_directed()
            for edge in graph:
                id = str((edge.s, edge.t))
                u, v = edge.s, edge.t
                xs[edge] = model.integer_var(name=id)
                xvs[u].append(xs[edge])
                xvs[v].append(xs[edge])
            for uv in xs:
                model.add_constraint(xs[uv] >= 0)
                model.add_constraint(xs[uv] <= 1)
            for u in xvs:
                uvs = xvs[u]
                model.add_constraint(sum(uvs) == 2)
            model.maximize(sum(xs.values()))
            model.solve()
            if model.solution is not None:
                self._two_factor = self.__class__(*(edge for edge in graph if model.solution[str(edge)] == 1.0))
            else:
                self._two_factor = self.__class__()
        return self._two_factor

    @two_factor.setter
    def two_factor(self, value):
        self._two_factor = value

    def dijkstra(self, src):
        unweighted_weight = lambda edge: 1 if not hasattr(edge, 'weight') else edge.weight
        setattr(src, 'distances', {})
        D = {x: (0, None) if x == src else (float('inf'), None) for x in self.nodes}
        D |= src.distances
        Q = set(self.nodes)
        while len(Q) > 0:
            x = min(Q, key=lambda node: D[node][0])
            Q.remove(x)
            for y in Q:
                if y in self.neighbors(x):
                    alt = D[x][0] + unweighted_weight(self[(x, y)])
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
            while has_parent(p[0]):
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

    # a TST has a net "flow" (i.e. #incoming - #outgoing edges per node) of 0, 
    # and is covering (i.e. all nodes are reached by at least 1 edge)
    # this function solves for these constraints via ILP, so every TST is a solution to this ILP (we call these solutions "union of tours")
    # we hope to recover a TST from a UOT. 
    def union_of_tours(self, exclude_dual=EdgeSet()):
        model = Model(name='uot')
        dual = self.dual()
        mfs_matching = self.match(include=self.minimum_feedback_set())
        dual_mfs_matching = dual.interior.match(include=dual.interior.minimum_feedback_set())
        xs = {
            edge: model.integer_var(name=str((edge.s, edge.t))) for edge in self
        }
        xvs = {
            node: {
                'incoming': [xs[edge] for edge in self if edge.t == node], 
                'outgoing': [xs[edge] for edge in self if edge.s == node], 
            } for node in self.nodes
        }
        for node in self.nodes:
            model.add_constraint((sum(xvs[node]['incoming']) - sum(xvs[node]['outgoing'])) == 0)
            model.add_constraint((sum(xvs[node]['incoming']) + sum(xvs[node]['outgoing'])) <= 2)
        for edge in self:
            model.add_constraint(xs[edge] + xs[edge.switch()] <= 1)

        def edge_dualnodes(edge):
            return dual[Edge(edge.midpoint(), edge.parallel_right().midpoint()).midpoint()], dual[Edge(edge.midpoint(), edge.parallel_left().midpoint()).midpoint()]

        model.maximize(
            sum(sum(xvs[node]['outgoing'])+sum(xvs[node]['incoming']) for node in self.nodes)
            +
            sum(xs[edge] for edge in self.matching_compatible(mfs_matching))
            +
            sum(xs[edge] for edge in self if (edge_dualnodes(edge) in dual_mfs_matching - exclude_dual))
        )
        model.solve()
        uot = self.__class__(*([] if model.solution is None else (edge for edge in self if model.solution[str(edge)] == 1.0)))
        return uot, dual_mfs_matching

    # Uses raycasting to determine if a point lies inside a polygon.
    # Assumes self is the polygon. 
    def test_interior(self, point):
        def in_range(edge, axis=0):
            ir = lambda edge: edge.s[axis] <= point[axis] and point[axis] <= edge.t[axis]
            return ir(edge) or ir(edge.switch())
        edges = set([edge.undirected() for edge in self.remove_directed() if in_range(edge)])
        above = [edge for edge in edges if edge.s[1] >= point[1] and edge.t[1] > point[1]]
        return (len(above) % 2) != 0

class SolidGridGraph(Graph):

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def __init__(self, *args, edges=[]):
        super().__init__(*args, *edges)
        for edge in list(self):
            self[edge, assign]
            self[edge.switch(), assign]

    # def neighbors(self, node: Node):
    #     # THIS IS FUCKING MADDENING
    #     # print(hash((-2, 3)), hash((-1, 3)))
    #     return [self.nodes[node + direction] for direction in SolidGridGraph.directions if (node, node + direction) in self and node+direction in self.nodes]

    def make_solid(self):
        for node in self.nodes:
            for direction in SolidGridGraph.directions:
                if node + direction in self.nodes:
                    self[(node, node+direction), assign]
        return self.undirected()

    def face_at_midpoint(self, point):
        points = (point[0]-0.5,point[1]-0.5), (point[0]-0.5,point[1]+0.5), (point[0]+0.5,point[1]+0.5), (point[0]+0.5, point[1]-0.5), (point[0]-0.5,point[1]-0.5)
        points = [(int(x), int(y)) for (x, y) in points]
        return self.intersect(Graph(*Graph.vertex_sequence_to_edges(points)).undirected())

    def midpoint_at_face(self, face):
        nodes = set([edge.s for edge in face]+[edge.t for edge in face])
        if len(nodes) != 4:
            raise ValueError(f"Attempted to get midpoint at incomplete face {face}.")
        return sum(nodes, start=Node(0, 0)) * 0.25

    @property
    def faces(self):
        if not hasattr(self, '_faces'):
            faces = set()
            def boxes(node):
                def box(axis):
                    return self.__class__(axis, axis.take_right(), axis.take_right().take_right(), axis.take_right().take_right().take_right())
                return [box(Edge(node, node + direction)) for direction in SolidGridGraph.directions]
            for node in self.nodes:
                for face in boxes(node):
                    faces.add(face)
            self._faces = faces
        return self._faces
    
    @faces.setter
    def faces(self, value):
        self._faces = value

    @property
    def interior_faces(self):
        if not hasattr(self, '_interior_faces'):
            faces = set()
            for face in self.faces:
                if face in self:
                    faces.add(face)
            self._interior_faces = faces
        return self._interior_faces
    
    @interior_faces.setter
    def interior_faces(self, value):
        self._interior_faces = value

    def dual(self):
        dual = self.__class__()
        exterior = self.__class__()
        for node in self.nodes:
            for direction in [(0.5, 0.5), (-0.5, 0.5), (0.5, -0.5), (-0.5, -0.5)]:
                dual[node+direction, assign]
        for node in dual.nodes:
            for direction in SolidGridGraph.directions:
                edge = SolidGridGraph.front_edge(direction=Edge((0, 0), direction), at=node)
                if node+direction in dual.nodes and edge in self:
                    dual[(node, node+direction), assign]
                    dual[(node+direction, node), assign]
            setattr(node, 'interior', len(dual.neighbors(node)) == 4)
            if not node.interior:
                exterior[node, assign]
        exterior = exterior.make_solid()
        for edge in dual:
            if edge.s.interior != edge.t.interior:
                edge.weight = 0
        for edge in list(exterior):
            if SolidGridGraph.front_edge(direction=edge-edge.s, at=edge.s) in self:
                del dual[edge]
                del exterior[edge]
        interior = self.__class__(*(edge for edge in dual if edge.s.interior and edge.t.interior))
        setattr(dual, 'interior', interior)
        setattr(dual, 'exterior', exterior)
        dual += exterior
        return dual

    # check if the exterior nodes in the dual graph are connected
    def is_solid(self):
        dual = self.dual()
        exterior = next(node for node in dual.nodes if not node.interior)        
        return len([node for node in dual.copy().bfs(exterior).nodes if hasattr(node, 'bfs_time')]) == len([node for node in dual if not node.interior])
    
    def boundary_cycle(self):
        boundary_edges = []
        for edge in self:
            if edge.parallel_left() not in self or edge.parallel_right() not in self:
                boundary_edges.append(edge)
        return self.__class__(*boundary_edges).as_cycle_facing_inwards()

    # Left, right, top/front, and bottom/back edges of a box oriented by a direction
    # using midpoint coordinates
    def left_edge(direction=Edge((0, 0), (0, 1)), at=Node(0, 0)):
        t = Edge(direction.midpoint(), direction.parallel_left().midpoint()).midpoint()
        return Edge(t - direction.s, t - direction.t)+at
    
    def right_edge(direction=Edge((0, 0), (0, 1)), at=Node(0, 0)):
        t = Edge(direction.midpoint(), direction.parallel_right().midpoint()).midpoint()
        return Edge(t - direction.s, t - direction.t)+at
    
    def front_edge(direction=Edge((0, 0), (0, 1)), at=Node(0, 0)):
        return Edge(SolidGridGraph.left_edge(direction, at).t, SolidGridGraph.right_edge(direction, at).t)
    
    def back_edge(direction=Edge((0, 0), (0, 1)), at=Node(0, 0)):
        return (SolidGridGraph.front_edge(direction, at) - direction.t).switch()

    def box(at=Node(0, 0)):
        return Graph(*(SolidGridGraph.right_edge(direction=Edge((0, 0), d), at=at) for d in SolidGridGraph.directions))

    def make_alternating_cycle(start, end):
        def start_face(step, node):
            return Graph(SolidGridGraph.back_edge(step, node)).undirected()
        def end_face(step, node):
            return Graph(SolidGridGraph.front_edge(step, node)).undirected()
        def side_face(step, node):
            return Graph(SolidGridGraph.left_edge(step, node), SolidGridGraph.right_edge(step, node)).undirected()
        cycle = Graph()
        cycle.alternating_cycle_nodes = [start]
        cycle.flipped = Graph()
        cycle.flipped.alternating_cycle_nodes = cycle.alternating_cycle_nodes
        cycle.flipped.flipped = cycle
        if start == end:
            step = Edge((0, 0), (0, 1))
            cycle += side_face(step, start)
            cycle.flipped += side_face(step.rotate_right(), start)
            return cycle
        step = Edge((0, 0), end - start) / abs(max(*(end-start)))
        i = 0
        node = start
        while node != end:
            if i == 0:
                cycle += start_face(step, node)
                cycle.flipped += side_face(step, node)
            if (i+1) % 2 == 0:
                cycle += side_face(step, node)
            else:
                cycle.flipped += side_face(step, node)
            i += 1
            node += step.t
            cycle.alternating_cycle_nodes.append(node)
        if len(cycle) % 2 == 0:
            cycle += end_face(step, node)
            cycle.flipped += side_face(step, node)
        else:
            cycle += side_face(step, node)
            cycle.flipped += end_face(step, node)
        return cycle.undirected()

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
        return strip.undirected()

    def test(self):
        graph = self.undirected()
        mfs_matching = graph.match(include=graph.minimum_feedback_set().flow_matching())
        dual = self.undirected().dual()
        dual.interior = dual.interior.undirected().make_bipartite()
        r = len([node for node in dual.interior.nodes if node.color % 2 == 0]) < len([node for node in dual.interior.nodes if node.color % 2 == 1])
        model = Model(name='test')
        edges = {
            edge: model.integer_var(name=str((edge.s, edge.t))) for edge in graph
        }
        nodes = {
            node: {
                'outgoing': [edges[edge] for edge in edges if edge.s == node],
                'incoming': [edges[edge] for edge in edges if edge.t == node]
            } for node in self.nodes
        }
        dual_nodes = {
            node: {
                'right': {
                    edge: edges[edge if node.color % 2 == (not r) else edge.switch()] for edge in SolidGridGraph.box(at=node) if edge in graph
                },
                'left': {
                    edge: edges[edge.switch() if node.color % 2 == r else edge] for edge in SolidGridGraph.box(at=node) if edge in graph
                }
            } for node in dual.interior.nodes
        }

        rights = sum([list(dual_nodes[node]['right'].values()) for node in dual_nodes], start=[])
        lefts  = sum([list(dual_nodes[node]['left'].values()) for node in dual_nodes], start=[])

        for edge in edges:
            model.add_constraint(edges[edge] >= 0)
            model.add_constraint(edges[edge] <= 1)
            model.add_constraint(edges[edge] + edges[edge.switch()] <= 1)
            # if edge in mfs_matching:
            #     model.add_constraint(edges[edge] == 1)
        for node in nodes:
            model.add_constraint(sum(nodes[node]['incoming']) == sum(nodes[node]['outgoing']))
            model.add_constraint(sum(nodes[node]['incoming']) <= 1)
            model.add_constraint(sum(nodes[node]['outgoing']) <= 1)

        # model.add_constraint(
        #     sum(right + left)
        # )
        # model.add_constraint(sum(lefts) == len(graph.nodes)-1)
        model.add_constraint(sum(rights) == len(graph.nodes))

        # model.add_constraint(sum(rights) - sum(lefts) <= len(graph.nodes))
        # model.add_constraint(sum(rights) >= len(graph.union_of_tours()))
        # print(len(graph.union_of_tours()))
        # for dual_node in dual_nodes:
            # for right, left in zip(dual_nodes[dual_node]['right'], dual_nodes[dual_node]['left']):
                # model.add_constraint(dual_nodes[dual_node]['right'][right] - dual_nodes[dual_node]['left'][left] >= 0)
                # model.add_constraint(dual_nodes[dual_node]['right'][right] + dual_nodes[dual_node]['left'][left] <= 1)

        # model.minimize(sum(edges[edge] for edge in edges))
        model.maximize(sum(sum(nodes[node]['outgoing']) + sum(nodes[node]['incoming']) for node in nodes) + sum(rights) - sum(lefts) + sum(edges[edge] for edge in mfs_matching))
        # model.maximize(sum(sum(nodes[node]['outgoing']) + sum(nodes[node]['incoming']) for node in nodes))

        # model.maximize(sum(rights) - sum(lefts))

        # model.maximize(sum(sum(dual_nodes[node]['right'].values()) for node in dual_nodes)-sum(sum(dual_nodes[node]['left'].values()) for node in dual_nodes))

        model.solve()
        return self.__class__(*([] if model.solution is None else (edge for edge in self.undirected() if model.solution[str(edge)] == 1.0)))

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

    def make_face(node, d):
        dx, dy = d
        a = node + (dx, 0)
        b = node + (dx, dy)
        c = node + (0, dy)
        return [
            Edge(node, a),
            Edge(node, c),
            Edge(a, b),
            Edge(c, b)
        ]

    def add_random_face(graph, m, n):
        graph.nodes[Node(0, 0), assign]
        num_edges_before = len(graph)
        nodes = list(graph.nodes)
        random.shuffle(nodes)
        while len(graph) == num_edges_before:
            node = nodes.pop()
            directions = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
            random.shuffle(directions)
            while len(directions) > 0:
                d = directions.pop()
                if (0, 0) <= node + d and node + d <= (m, n):
                    for edge in SolidGridGraph.make_face(node, d):
                        graph[edge, assign]
                        graph[edge.switch(), assign]

# takes every possible combination of possible pairs of nodes
def SGG_iterator(nodes, n_edges=0):
    def possible_edges(nodes):
        for node in nodes.copy():
            for direction in SolidGridGraph.directions:
                if node + direction in nodes:
                    yield (node, node+direction)
    yield from combinations(possible_edges(nodes), n_edges)


