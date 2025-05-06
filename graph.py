import random, copy
from typing import Union, TypeVar, Generic
from math import atan2, degrees, fmod, sqrt, pow, sin, cos, radians
from docplex.mp.model import Model

class Node(tuple):

    def __new__(cls, *value, edges=[]):
        while isinstance(value, tuple) and len(value) == 1:
            value = value[0]
        node = super(Node, cls).__new__(cls, value)
        node.edges = edges
        return node

    def __add__(self, other):
        if isinstance(other, Node) or isinstance(other, tuple):
            return Node(*(self[i]+o for i, o in enumerate(other)), edges=self.edges)
        elif isinstance(other, Edge):
            return other+self
        else:
            return Node(*(i+other for i in self), edges=self.edges)

    def __sub__(self, other):
        if isinstance(other, Node) or isinstance(other, tuple):
            return Node(*(self[i]-o for i, o in enumerate(other)), edges=self.edges)
        else:
            return Node(*(i-other for i in self), edges=self.edges)

    def __mul__(self, other):
        if isinstance(other, Node) or isinstance(other, tuple):
            return Node(*(self[i]*o for i, o in enumerate(other)))
        else:
            return Node(*(i*other for i in self))

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


class Edge:

    s: Node
    t: Node
    w: int
    d: bool

    def __init__(self, s, t, d=False):
        self.s = s if isinstance(s, Node) else Node(*s)
        self.t = t if isinstance(t, Node) else Node(*t)
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
            if self.d and other.d:
                return hash((self.s, self.t)) == hash((other.s, other.t))
        return hash(self) == hash(other)
    
    def __getitem__(self, idx):
        return [self.s, self.t][idx]
        # if idx != 0 and idx != 1:
        #     raise IndexError(f"Attempted to get item {idx} in edge.") 
        # else:
        #     if idx == 0:
        #         return self.s
        #     else: 
        #         return self.t
    
    def __contains__(self, node):
        return self.s == node or self.t == node
            
    def __str__(self):
        return f"({self.s}, {self.t})"
    
    def __repr__(self):
        return (self.s, self.t).__repr__()

    def __add__(self, other):
        if isinstance(other, Edge):
            return Edge(self.s+other.s, self.t+other.t)
        elif isinstance(other, Node):
            return Edge(self.s+other, self.t+other)
        else:
            raise ValueError(f"Attempted to add {type(other)} to an edge")
        # if isinstance(other, Node):
        #     return Edge(self.s+other, self.t+other)
        # elif isinstance(other, Edge):
        #     return Edge(self.s+other.s, self.t+other.t)
        # else:
        #     raise ValueError(f"Attempted to add {type(other)} to an edge")

    def meets(self, other):
        if not self.d or not other.self.d:
            return self.s if self.s in other else self.t
        else:
            return self.s if self.s == other.t else self.t if self.t == other.t else None

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
        return self.__class__(self.t, self.s, *self[2:])

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
        return Edge(self.s-self.s, self.t-self.s)
    
    def rotate(self, degrees):
        degrees = radians(degrees)
        u = self.direction()
        v = u.t*(cos(degrees), sin(degrees))
        w = u.t*(sin(degrees), cos(degrees))
        d = (v[0]-v[1], w[0]+w[1])
        if isinstance(self.s[0], int):
            d = round(d[0]), round(d[1])
        return Edge(self.s, self.s+d)

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
    

class assign:
    pass

def opunpack(cls):
    getitem = cls.__getitem__
    def getter(self, key, op=None):
        if isinstance(key, tuple) and len(key) > 1 and key[1] is assign:
            return getitem(self, key[0], op=key[1])
        else:
            return getitem(self, key, op=op)
    cls.__getitem__ = getter
    return cls

@opunpack
class setdict(dict):

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        super(setdict, self).__setitem__(key, value)

    # def keys(self):
    #     return self.values()

    def __getitem__(self, key, op=None):
        if op is assign and key not in self:
            super(setdict, self).__setitem__(key, key)
        return super().__getitem__(key)

    def __init__(self, *args):
        super().__init__()
        for arg in args:
            self[arg, assign]

    def __str__(self):
        return str(list(self.keys()))

#TODO: add nodeset
@opunpack
class EdgeSet(setdict):

    nodes: setdict

    def __init__(self, *args, edges=[]):
        self.nodes = setdict()
        super().__init__(*args, *edges)
        for edge in self:
            self.nodes[edge.s, assign]
            self.nodes[edge.t, assign]

    def __getitem__(self, key, op=None):
        if op is assign:
            if isinstance(key, Node):
                newnode = self.nodes[key, assign]
                for edge in self:
                    if edge.s == newnode:
                        edge.s = newnode
                    if edge.t == newnode:
                        edge.t = newnode
                return newnode
            else:
                edge = key if isinstance(key, Edge) else Edge(*key)
                if edge.s in self.nodes:
                    edge.s = self.nodes[edge.s]
                else:
                    self.nodes[edge.s, assign]
                if edge.t in self.nodes:
                    edge.t = self.nodes[edge.t]
                else:
                    self.nodes[edge.t, assign]
                return super().__getitem__(edge, op=op)
        else:
            if isinstance(key, Node):
                for edge in self:
                    if key in edge:
                        return edge
                raise KeyError
            else:
                return super().__getitem__(key, op=op)

    # TODO: see if you can use the fact that connected edges are kept track of in nodes to make this more efficient
    def __delitem__(self, edge):
        super().__delitem__(edge)
        for node in (edge.s, edge.t):
            if sum([node in edge for edge in self]) == 0:
                del self.nodes[node]

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
            for edge in other:
                self[edge, assign]
            return self
        elif isinstance(other, Edge):
            self[other, assign]
            return self
        else:
            raise ValueError(f"Attempted to add object of type {type(other)} to an EdgeSet.")

    def __add__(self, other):
        graph = self.copy()
        graph += other
        return graph
        
    def __sub__(self, other):
        if isinstance(other, EdgeSet):
            return EdgeSet(*[edge for edge in self if edge not in other])
            graph = self.copy()
            for edge in self:
                if edge in other:
                    del graph[edge]
            return graph
        else:
            raise ValueError(f"Attempted to add object of type {type(other)} to an EdgeSet.")
        
    def intersect(self, other):
        union = self + other
        return EdgeSet(*[edge for edge in union if edge in self and edge in other])
    
    def symmetric_difference(self, other):
        union = self + other
        return EdgeSet(*[edge for edge in union if not (edge in self and edge in other)])        

    def copy(self):
        newself = copy.deepcopy(self)
        newself.nodes = copy.deepcopy(self.nodes)
        return newself

    # Returns a cycle such that the direction of each edge faces inwards
    def get_cycle(self):
        graph = self.copy()
        t = list(graph.nodes)[0]
        cycle = []
        while t in graph:
            edge = graph[t]
            if t != edge.s and not edge.d:
                cycle.append(edge.switch())
            elif t == edge.s:
                cycle.append(edge)
            t = cycle[-1].other(t)
            del graph[edge]
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
        return None
                    
    # Uses raycasting to determine if a point lies inside a polygon.
    # Assumes self is the polygon. 
    def test_interior(self, point):
        # cycle = self.get_cycle()
        # if len(cycle) == 0:
        #     raise ValueError("Attempted to test the interior of a non-cyclical set of edges.")
        # Sees if a point lies in the range of the edge along some axis.
        def in_range(edge, axis=0):
            return (edge.s[axis] <= point[axis] and point[axis] <= edge.t[axis]) or (edge.t[axis] <= point[axis] and point[axis] <= edge.s[axis])
        edges = [edge for edge in self if in_range(edge)]
        above = [edge for edge in edges if edge.s[1] >= point[1] and edge.t[1] >= point[1]]
        return (len(above) % 2) != 0
    

@opunpack
class Graph(EdgeSet):

    def __getitem__(self, edge, op=None):
        if self.is_admissible_edge(edge):
            return super().__getitem__(edge, op=op)
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
        src = self.nodes[list(self.nodes.keys())[0]]
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
        for edge in self:
            if not self.nodes[edge.s].color:
                self[edge].switch()

    @property
    def two_factor(self):
        if not hasattr(self, '_two_factor'):
            model = Model(name='two_factor')
            xs = {}
            xvs = {node: [] for node in self.nodes}
            for edge in self:
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
                self._two_factor = EdgeSet(*[edge for edge in self if model.solution[str(edge)] == 1.0])
            else:
                self._two_factor = EdgeSet()
        return self._two_factor

    @two_factor.setter
    def two_factor(self, value):
        self._two_factor = value

    def dijkstra(self, src):
        unweighted_weight = lambda edge: 1 if not hasattr(edge, 'weight') else edge.weight
        D = {x: 0 if x == src else float('inf') for x in self.nodes}
        P = {src: src}
        Q = set(self.nodes)
        while len(Q) > 0:
            x = min(Q, key=lambda node: D[node])
            Q.remove(x)
            for y in Q:
                if y in self.neighbors(x) and Edge(x, y) in self:
                    alt = D[x] + unweighted_weight(self[Edge(x, y)])
                    if alt < D[y]:
                        D[y] = alt
                        P[y] = x
        def path(x):
            p = [x]
            while p[-1] != src and p[-1] in P:
                p.append(P[p[-1]])
            if x in P:
                return p + [src]
            else: 
                return []
        D['path'] = path
        return D
        
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
            
    def tst(self):
        self.make_bipartite()
        model = Model(name='tst')
        xvs = {
            'incoming': {node: [] for node in self.nodes},
            'outgoing': {node: [] for node in self.nodes}
        }

        for edge in self:
            for e in [edge, edge.switch()]:
                var = model.integer_var(name=str((e.s, e.t)))
                xvs['outgoing'][e.s].append(var)
                xvs['incoming'][e.t].append(var)

        for node in self.nodes:
            model.add_constraint((sum(xvs['incoming'][node]) - sum(xvs['outgoing'][node])) == 0)
            model.add_constraint((sum(xvs['incoming'][node]) + sum(xvs['outgoing'][node])) >= 1)

        model.minimize(sum(sum(xvs['outgoing'][node]) + sum(xvs['incoming'][node]) for node in self.nodes))
        model.solve()

        if model.solution is not None:
            return [*[Edge(edge.s, edge.t, d=True) for edge in self if model.solution[str(edge)] == 1.0], *[Edge(edge.t, edge.s, d=True) for edge in self if model.solution[str(edge.switch())] == 1.0]]
        return []

class SolidGridGraph(Graph):

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def is_admissible_edge(self, edge):
        edge = Edge(*edge)
        for d in SolidGridGraph.directions:
            if edge.s + d == edge.t:
                return True
        return False

    def neighbors(self, node: Node):
        return [node + direction for direction in SolidGridGraph.directions if node + direction in self.nodes]

    def make_solid(self):
        for node in self.nodes:
            for neighbor in self.neighbors(node):
                self[Edge(node, neighbor), assign]
        return self

    def face_at_midpoint(self, point):
        points = (point[0]-0.5,point[1]-0.5), (point[0]-0.5,point[1]+0.5), (point[0]+0.5,point[1]-0.5), (point[0]+0.5, point[1]+0.5)
        edges = [(points[i], points[(i+1)%4]) for i in range(len(points))]
        return [self[edge] for edge in edges if edge in self]

    def midpoint_at_face(self, face):
        nodes = set([edge.s for edge in face]+[edge.t for edge in face])
        if len(nodes) != 4:
            raise ValueError(f"Attempted to get midpoint at incomplete face {face}.")
        return sum(nodes, start=Node(0, 0)) * 0.25

    @property
    def faces(self):
        if not hasattr(self, '_faces'):
            faces = set()
            def r_box(edge):
                return EdgeSet(edge, edge.take_right(), edge.take_right().take_right(), edge.take_right().take_right().take_right())
            def l_box(edge):
                return EdgeSet(edge, edge.take_left(), edge.take_left().take_left(), edge.take_left().take_left().take_left())
            for edge in self:
                for e in [edge, edge.switch()]:
                    r, l = r_box(e), l_box(e)
                    faces.add(r)
                    faces.add(l)
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
    
    def boundary_cycle(self):
        boundary_edges = []
        for edge in self:
            if edge.parallel_left() not in self or edge.parallel_right() not in self:
                boundary_edges.append(edge)
        return EdgeSet(*boundary_edges).get_cycle()
    
    
    def get_alternating_strip(self, edge):
        sequence = [edge]
        start = Edge(edge.midpoint(), edge.parallel_right().midpoint()).midpoint()
        def isin(box):
            return sum(e in self for e in box) == len(box)
        def n_box(edge):
            return edge.take_right().go_forward(), edge.switch().take_left().go_forward().switch(), edge.parallel_right().parallel_right()
        while edge.parallel_right().parallel_right() in self and isin(n_box(edge)):
            for e in n_box(edge):
                sequence.append(e)
            edge = edge.parallel_right().parallel_right()
        if edge.parallel_right() in self:
            sequence.append(edge.parallel_right())
            strip = EdgeSet(*sequence)
            setattr(strip, 'odd', True)
        else:
            sequence = sequence[:-1]
            strip = EdgeSet(*sequence)
            setattr(strip, 'even_edges', sequence[-2:])
            setattr(strip, 'odd', False)
        if len(strip) > 0:
            setattr(strip, 'start', start)
            setattr(strip, 'end', Edge(sequence[-1].midpoint(), sequence[-2].midpoint()).midpoint())
            setattr(strip, 'edge', sequence[0])
        return strip

    def get_perimeter_of_alternating_strip(strip):
        graph = strip.copy()
        nodes = []
        for node in strip.nodes:
            if sum([node in edge for edge in strip]) > 1:
                nodes.append(node)
        for n1 in nodes:
            for n2 in nodes:
                if (n1, n2) in strip:
                    del graph[Edge(n1, n2)]
        return graph
    
    def flip_perimeter_of_alternating_strip(perimeter):
        flipped = EdgeSet()
        ff = lambda edge: edge.go_forward().go_forward()
        fr = lambda edge: edge.go_forward().take_right()
        rf = lambda edge: edge.take_right().go_forward()
        rr = lambda edge: edge.take_right().take_right()
        edge = next(iter(perimeter))
        i = edge
        # exit()
        for _ in perimeter:
            if ff(i) in perimeter:
                i = ff(i)
            elif fr(i) in perimeter:
                i = fr(i)
            elif rf(i) in perimeter:
                i = rf(i)
            elif rr(i) in perimeter:
                i = rr(i)
            else:
                edge = edge.switch()
                break
        for _ in perimeter:
            if ff(edge) in perimeter:
                flipped[edge.go_forward(), assign]
                edge = ff(edge)
            elif fr(edge) in perimeter:
                flipped[edge.go_forward(), assign]
                edge = fr(edge)
            elif rf(edge) in perimeter:
                flipped[edge.take_right(), assign]
                edge = rf(edge)
            elif rr(edge) in perimeter:
                flipped[edge.take_right(), assign]
                edge = rr(edge)
        return flipped

    def get_alternating_strips(self):
        two_factor = self.two_factor
        alternating_strips = []
        alternating_strip_edges = EdgeSet()
        for edge in self:
            for edge in [edge, edge.switch()]:
                x1, y1 = edge.midpoint()
                x2, y2 = edge.parallel_right().midpoint()
                if not two_factor.test_interior(((x1+x2)/2, (y1+y2)/2)):
                    strip = SolidGridGraph.get_alternating_strip(two_factor, edge)
                    if len(strip) > 1 and strip not in alternating_strips:
                        alternating_strips.append(strip)
                        alternating_strip_edges += strip
        for strip in alternating_strips:
            setattr(strip, 'chain', strip.edge.take_right() in alternating_strip_edges or strip.edge.switch().take_left() in alternating_strip_edges)
        return alternating_strips
    
    def edge_flip(self, strip):
        graph = self.copy()
        perimeter = SolidGridGraph.get_perimeter_of_alternating_strip(strip)
        graph.two_factor = (self.two_factor - perimeter) + SolidGridGraph.flip_perimeter_of_alternating_strip(perimeter)
        return graph

    def get_dual(self):
        dual = SolidGridGraph()
        for face in self.faces:
            node = Node(*self.midpoint_at_face(face))
            setattr(node, 'interior', face in self.interior_faces)
            dual.nodes[node, assign]
        dual = dual.make_solid()
        dual.is_admissible_edge = lambda *args: True
        exterior_nodes = set([x for x in dual.nodes if not x.interior])
        for a in exterior_nodes:
            for b in exterior_nodes:
                edge = Edge(a, b)
                setattr(edge, 'weight', 0)
                dual[edge, assign]
        return dual
    
    def static_alternating_strip(self):
        alternating_strips = SolidGridGraph.get_alternating_strips(self)
        dual = SolidGridGraph.get_dual(self)
        all_pairs_shortest_path = {
            node: dual.dijkstra(node) for node in dual.nodes
        }
        H = Graph()
        for strip in alternating_strips:
            start = Node(*strip.start)
            end = Node(*strip.end)
            H.nodes[start, assign]
            H.nodes[end, assign]
        for strip in alternating_strips:
            start = H.nodes[Node(*strip.start)]
            end = H.nodes[Node(*strip.end)]
            setattr(start, 'chain', strip.chain)
            setattr(end, 'odd', strip.odd)
            edge = Edge(start, end, d=True)
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
                        if other.start in all_pairs_shortest_path[c1]['path'](c2):
                            edge = Edge(c, other.start, d=True)
                            setattr(edge, 'weight', 0)
                            H[edge, assign]
                    else:
                        if len(all_pairs_shortest_path[c1]['path'](c2)) > 0 and other.start in (c1, c2):
                            edge = Edge(c, other.start, d=True)
                            setattr(edge, 'weight', 0)
                            H[edge, assign]
        all_pairs_shortest_path = {
            node: H.dijkstra(node) for node in H.nodes
        }
        begins = [node for node in H.nodes if hasattr(node, 'chain') and not node.chain]
        odds = [node for node in H.nodes if hasattr(node, 'odd') and node.odd]
        paths = sum([[all_pairs_shortest_path[begin]['path'](odd) for begin in begins] for odd in odds], [])
        paths = [path for path in paths if len(path) > 0]
        static_strip = EdgeSet()
        if len(paths) > 0:
            path = min(paths, key=len)
            for i in range(len(path)-1):
                for strip in alternating_strips:
                    if strip.start == path[i] and strip.end == path[i+1] and not self.two_factor.test_interior(strip.start):
                        static_strip += strip
        return static_strip
        
    def reduce_two_factor(self):
        static_strip = self.static_alternating_strip()
        if len(static_strip) == 0:
            return None
        else:
            return self.edge_flip(static_strip)

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

# sgg = SolidGridGraph(((1, 0), (2, 0)), ((1, 1), (2, 1)), ((1, 2), (1, 1)), ((2, 2), (2, 1)), ((1, 4), (1, 3)), ((1, 3), (2, 3)), ((2, 4), (2, 3)), ((1, 6), (1, 5)), ((1, 5), (2, 5)), ((2, 6), (2, 5)), ((1, 7), (2, 7)), ((0, 0), (1, 0)), ((2, 0), (3, 0)), ((3, 0), (3, 1)), ((3, 1), (3, 2)), ((3, 2), (2, 2)), ((1, 2), (0, 2)), ((0, 2), (0, 1)), ((0, 1), (0, 0)), ((1, 4), (0, 4)), ((0, 4), (0, 5)), ((0, 5), (0, 6)), ((0, 6), (1, 6)), ((2, 6), (3, 6)), ((3, 6), (3, 5)), ((3, 5), (3, 4)), ((3, 4), (2, 4)), ((1, 7), (1, 8)), ((1, 8), (0, 8)), ((0, 8), (0, 9)), ((0, 9), (1, 9)), ((1, 9), (2, 9)), ((2, 9), (3, 9)), ((3, 9), (3, 8)), ((3, 8), (2, 8)), ((2, 8), (2, 7)), ((2, 7), (2, 6)), ((2, 6), (1, 6)), ((1, 6), (1, 7)), ((0, 5), (1, 5)), ((1, 5), (1, 4)), ((1, 4), (2, 4)), ((2, 5), (2, 4)), ((2, 5), (3, 5)), ((1, 8), (2, 8)), ((1, 9), (1, 8)), ((2, 9), (2, 8)), ((1, 3), (1, 2)), ((1, 2), (2, 2)), ((2, 2), (2, 3)), ((0, 1), (1, 1)), ((1, 1), (1, 0)), ((2, 0), (2, 1)), ((2, 1), (3, 1)))
# sgg.boundary_cycles()

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
 

