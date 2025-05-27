import random, copy
from typing import Union, TypeVar, Generic
from math import atan2, degrees, fmod, sqrt, pow, sin, cos, radians
from itertools import chain, combinations
from docplex.mp.model import Model


def diff_update_dict(dict_a, dict_b):
    for key in dict_b.keys():
        if key not in dict_a:
            dict_a[key] = dict_b[key] if not hasattr(dict_b[key], 'copy') else dict_b[key].copy()

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
            super(setdict, self).__setitem__(key, key)
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
        return node

    def copy(self):
        node = self.__class__(*self)
        diff_update_dict(node.__dict__, self.__dict__)
        return node

    def __add__(self, other):
        if isinstance(other, Node) or isinstance(other, tuple):
            return Node(self[i]+o for i, o in enumerate(other))
        elif isinstance(other, Edge):
            return other+self
        else:
            return Node(i+other for i in self)

    def __sub__(self, other):
        if isinstance(other, Node) or isinstance(other, tuple):
            return Node(self[i]-o for i, o in enumerate(other))
        else:
            return Node(i-other for i in self)

    def __mul__(self, other):
        if isinstance(other, Node) or isinstance(other, tuple):
            return Node(self[i]*o for i, o in enumerate(other))
        else:
            return Node(i*other for i in self)

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
        elif isinstance(other, Node):
            return self.__class__(self.s+other, self.t+other)
        else:
            raise ValueError(f"Attempted to add {type(other)} to an edge")
    
    def undirected(self):
        return self.__class__(min(self.s, self.t, key=hash).copy(), max(self.s, self.t, key=hash).copy())

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
# @opunpack
class EdgeSet(setdict):

    nodes: setdict

    def __init__(self, *args, edges=[]):
        self.nodes = setdict()
        super().__init__()
        for arg in args:
            self[arg, assign]

    def set_node(self, key):
        key = self.nodes[key.copy() if isinstance(key, Node) else Node(*key), assign]
        for node in self.nodes:
            if (key, node) in self:
                key.edges.add(self[(key, node)])
                self[(key, node)].s = key
            if (node, key) in self:
                key.edges.add(self[(node, key)])
                self[(node, key)].t = key
        return key

    def get_node(self, key):
        if key in self.nodes:
            return self.nodes[key]
        raise KeyError(f"{key}")

    def set_edge(self, key):
        edge = key if isinstance(key, Edge) else Edge(*key)
        edge.s = self.nodes[(edge.s.copy(), assign) if edge.s not in self else edge.s]
        edge.t = self.nodes[(edge.t.copy(), assign) if edge.t not in self else edge.t]
        edge.s.edges.add(edge)
        edge.t.edges.add(edge)
        return super().__getitem__(edge, op=assign)

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
            self[key].edges.remove(edge)
            del self[edge]
        del self.nodes[key]

    def del_edge(self, key):
        edge = self[key]
        if edge in edge.s.edges:
            edge.s.edges.remove(edge)
        if edge in edge.t.edges:
            edge.t.edges.remove(edge)
        super().__delitem__(edge)

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
                self.nodes[node, assign]
            for edge in other:
                self[edge, assign]
            return self
        elif isinstance(other, Edge) or isinstance(other, Node):
            self[other, assign]
            return self
        else:
            raise ValueError(f"Attempted to add object of type {type(other)} to an EdgeSet.")

    def __isub__(self, other):
        if isinstance(other, EdgeSet):
            for edge in other:
                if edge in self:
                    del self[edge]
            return self
        elif isinstance(other, Edge) or isinstance(other, Node): 
            del self[other]
            return self
        else:
            raise ValueError(f"Attempted to subtract object of type {type(other)} to an EdgeSet.")

    def __add__(self, other):
        graph = self.copy()
        graph += other
        return graph
        
    def __sub__(self, other):
        if isinstance(other, EdgeSet):
            graph = self.copy()
            graph -= other
            return graph
            return self.__class__(*(edge for edge in self if edge not in other))
        else:
            raise ValueError(f"Attempted to add object of type {type(other)} to an EdgeSet.")
        
    def copy(self):
        graph = super().copy()
        for node in list(graph.nodes):
            graph.nodes[node, assign]
        for edge in list(graph):
            graph[edge, assign]
        return graph

    def undirected(self):
        graph = self.copy()
        for edge in list(graph):
            graph[edge.switch(), assign]
        return graph

    def remove_directed(self):
        graph = self.copy()
        for edge in list(graph):
            if edge.switch() in graph:
                del graph[edge]
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

# @opunpack
class Graph(EdgeSet):

    def set_node(self, key):
        key = EdgeSet.set_node(self, key)
        return self.dijkstra(key)

    def neighbors(self, node):
        return [edge.t for edge in self[node].edges if edge.s == node]

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

    # Naively assumes self is a simple cycle, and returns self reoriented as such (if possible)
    def as_simple_cycle(self):
        if len(self.nodes) == 0:
            return []
        s = list(self.nodes)[0]
        D = set([s])
        cycle = []
        while s in self:
            try:
                edge = next(edge for edge in self if edge.s == s and edge.t not in D)
                cycle.append(edge)
                s = edge.t
                D.add(s)
            except StopIteration:
                D.add(s)
        return cycle

    # Returns a cycle such that the rightwards-perpendicular direction of each edge faces inwards
    def as_cycle_facing_inwards(self):
        cycle = self.as_simple_cycle()
        if cycle[-1].t == cycle[0].s:
            insides, outsides = 0, 0
            for i in range(len(cycle)-1):
                if fmod(cycle[i+1].axis() - cycle[i].axis(), 360) < 180:
                    insides += 1
                if fmod(cycle[i+1].axis() - cycle[i].axis(), 360) > 180:
                    outsides += 1
            if outsides > insides:
                for i in range(len(cycle)):
                    cycle[i] = cycle[i].switch()
            return cycle
        return []

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
    
    def make_bipartite(self):
        # for node in self.nodes:
        #     setattr(node, 'color', 0)
        # nodes = set(self.nodes)
        # discovered = set()
        # while len(discovered) != len(nodes):
        #     q = [(nodes - discovered).pop()]
        #     while len(q) > 0:
        #         x = q.pop()
        #         discovered.add(x)
        #         for neighbor in x.neighbors():
        #             neighbor.color = not x.color
        #             if neighbor not in discovered:
        #                 q.append(neighbor)
        # return self
        src = next(iter(self.nodes))
        src.color = 0
        D = set()
        D.add(src)
        Q = []
        Q.append(src)
        while len(Q) > 0:
            x = Q.pop()
            D.add(x)
            for neighbor in self.neighbors(x):
                neighbor.color = not x.color
                self.nodes[neighbor].color = not x.color
                if neighbor not in D:
                    Q.append(self.nodes[neighbor])
                    D.add(self.nodes[neighbor])
        return self

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
        src = self[src]
        if not hasattr(src, 'distances'):
            setattr(src, 'distances', {})
        D = {x: (0, x) if x == src else (float('inf'), None) for x in self.nodes}
        D |= src.distances
        Q = set(self.nodes)
        while len(Q) > 0:
            x = min(Q, key=lambda node: D[node][0])
            Q.remove(x)
            for y in Q:
                if y in self.neighbors(x):
                    # if (x, y) not in self:
                        # print(x, y, self.neighbors(x), x.edges, Edge(x, y) in self)
                        # exit()
                    alt = D[x][0] + unweighted_weight(self[(x, y)])
                    if alt < D[y][0]:
                        D[y] = (alt, x)
        src.distances |= D
        return src

    def shortest_path(self, src, dest):
        def has_parent(node):
            return node in src.distances and src.distances[node][1] is not None
        if has_parent(dest):
            p = [dest]
            while has_parent(p[0]) and p[0] != src:
                p.insert(0, src.distances[p[0]][1])
            return p
        else:
            return []

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
    def union_of_tours(self):
        self.make_bipartite()
        model = Model(name='tst')
        xvs = {
            'incoming': {node: [] for node in self.nodes},
            'outgoing': {node: [] for node in self.nodes}
        }
        for edge in self:
            var = model.integer_var(name=str((edge.s, edge.t)))
            xvs['outgoing'][edge.s].append(var)
            xvs['incoming'][edge.t].append(var)
        for node in self.nodes:
            model.add_constraint((sum(xvs['incoming'][node]) - sum(xvs['outgoing'][node])) == 0)
            model.add_constraint((sum(xvs['incoming'][node]) + sum(xvs['outgoing'][node])) >= 1) # not clear if >= 2 vs >= 1 changes anything. it should be >= 2
        model.minimize(sum(sum(xvs['outgoing'][node]) + sum(xvs['incoming'][node]) for node in self.nodes))
        model.solve()
        if model.solution is not None:
            graph = self.__class__()
            for edge in self:
                if model.solution[str(edge)] == 1.0:
                    graph[edge, assign]
            return graph
        return []

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
        # edges = [Edge(points[i], points[(i+1)%4]) for i in range(len(points))]
        return self.__class__(*(Edge(points[i], points[i+1]) for i in range(4) if (points[i], points[i+1]) in self)).undirected()

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

    def get_dual(self):
        dual = self.__class__()
        for face in self.faces:
            node = Node(*self.midpoint_at_face(face))
            setattr(node, 'interior', face in self.interior_faces)
            dual.nodes[node, assign]
        dual = dual.make_solid()
        exterior_nodes = set([x for x in dual.nodes if not x.interior])
        for a in exterior_nodes:
            for b in exterior_nodes:
                for direction in SolidGridGraph.directions:
                    if a == b + direction:
                        edge = Edge(a, b)
                        setattr(edge, 'weight', 0)
                        dual[edge, assign]
        return dual

    # check if the exterior nodes in the dual graph are connected
    def is_solid(self):
        dual = self.get_dual()
        interior = [node for node in dual.nodes if node.interior]
        exterior = [node for node in dual.nodes if not node.interior]
        if len(interior) == 0:
            return False
        def bfs(src):
            D = set()
            Q = [src]
            while len(Q) > 0:
                x = Q.pop()
                D.add(x)
                for n in dual.neighbors(x):
                    if n.interior == src.interior and n not in D:
                        Q.append(n)
                        D.add(n)
            return D
        return len(bfs(interior[0])) == len(interior) and len(bfs(exterior[0])) == len(exterior)
    
    def boundary_cycle(self):
        boundary_edges = []
        for edge in self:
            if edge.parallel_left() not in self or edge.parallel_right() not in self:
                boundary_edges.append(edge)
        return self.__class__(*boundary_edges).as_cycle_facing_inwards()
    
    def get_alternating_strip(self, edge):
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

    def get_alternating_strips(self):
        # dual = SolidGridGraph.get_dual(self)
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
        dual = SolidGridGraph.get_dual(self)
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
        paths = [path for path in paths if len(path) > 0]
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

    def components(self):
        components = []
        for node in self.nodes:
            a = node.component()
            add_component = True
            for b in components:
                if len(a.symmetric_difference(b)) == 0:
                    add_component = False
            if add_component:
                components.append(a)
        return components

    def is_tours(self):
        tours = []
        nodes = set(self.nodes)
        while len(nodes) > 0:
            node = nodes.pop()
            discovered = set([node])
            tour = [node]
            len_t = 0
            while len_t != len(tour):
                len_t = len(tour)
                for neighbor in tour[-1].neighbors():
                    if neighbor not in discovered or neighbor == tour[0]:
                        tour.append(neighbor)
                        discovered.add(neighbor)
            if tour[-1] != tour[0]:
                return False
            else:
                tours.append(tour)
            nodes -= discovered
        return True

    def ninesquare(at=Node(0, 0)):
        square = SolidGridGraph()
        for r in range(-1, 2):
            for c in range(-1, 2):
                square.nodes[Node(*at) + Node(r, c), assign]
        return square.make_solid()

    def neighborhood(self, at=Node(0, 0)):
        return self.intersect(SolidGridGraph.ninesquare(at))

    def clear_neighborhood(self, at):
        self -= self.neighborhood(at)
        return self

    # for each 9-neighborhood, find the set of edges that  
    # 1. maintains tour-ness (each node is part of a tour)
    # 2. maintains the number of edges 
    # 3. decreases the number of strongly connected components in the graph
    # the 9-neighborhood to patch is specified by the node passed to the 'at' parameter
    # edges around the node are cleared and then the search proceeds
    def kernel_patch(self, at):
        graph = self.copy()
        n_components = len(graph.components())
        i = 0
        k = 0
        for edge_set in SGG_iterator(self.neighborhood(at).nodes, len(self.neighborhood(at))):
            print(i)
            i+=1
            candidate = self.copy().clear_neighborhood(at) + Graph(*edge_set)
            if len(candidate.components()) < n_components and candidate.is_tours():
                graph = candidate
                n_components = len(graph.components())
                k += 1
            if k == 1:
                print(len(graph.neighborhood(at)))
                for node in graph.neighborhood(at).nodes:

                    print(node.edges)
                return graph
            if n_components == len(self.components()) - 1:
                return graph
        return graph

    # get dual
    def unfurl_uot(self, uot):
        uot = Graph(*uot)
        dual = self.get_dual()
        mst = Graph()

        def face_at(dual_node):
            face = SolidGridGraph.face_at_midpoint(uot, dual_node)
            edges = [edge for edge in face if edge in uot] + [edge.switch() for edge in face if edge.switch() in uot]
            es = Graph(*edges)
            for node in face.nodes:
                if node in uot.nodes:
                    es[node, assign]
            return es
        
        def midpoint_left(edge):
            return Edge(edge.midpoint(), edge.parallel_left().midpoint()).midpoint()

        def midpoint_right(edge):
            return Edge(edge.midpoint(), edge.parallel_right().midpoint()).midpoint()

        def midpoint_forward(edge):
            return edge.go_forward().midpoint()

        return mst

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
        for node in nodes:
            for direction in SolidGridGraph.directions:
                if node + direction in nodes:
                    yield (node, node+direction)
    yield from combinations(possible_edges(nodes), n_edges)


