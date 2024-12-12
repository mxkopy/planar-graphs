import random

# nodes are defined as 2-int tuples associated with an integer weight
class Weights: 

    graph_type = None
    weights = {}

    def order_edge(self, edge):
        return min(edge[0], edge[1], key=self.graph_type.hash_node), max(edge[0], edge[1], key=self.graph_type.hash_node)

    def __init__(self, graph_type, weights={}):
        self.graph_type = graph_type
        self.weights = weights

    def __getitem__(self, edge):
        return self.weights[self.graph_type.order_edge(edge)]
    
    def __setitem__(self, edge, value):
        if self.graph_type.is_admissible(edge):
            self.weights[self.graph_type.order_edge(edge)] = value
        else:
            raise KeyError(f"Attempted to set an inadmissible edge {edge} to the graph.")

    def __delitem__(self, edge):
        del self.weights[edge]

    def __contains__(self, edge):
        return self.graph_type.order_edge(edge) in self.weights.keys()
    
    def __iter__(self):
        return iter(self.weights)

    def values(self):
        return self.weights.values()
    
    def keys(self):
        return self.weights.keys()
    
    def edges(self):
        return self.keys()

    def nodes(self):
        return set([x for x, _ in self.weights] + [y for _, y in self.weights])
    
    def copy(self):
        return Weights(self.graph_type, weights=self.weights.copy())

class SolidGridGraph:

    weights: Weights
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def hash_node(node):
        return 2**(node[0]+1) + 3**(node[1]+1)
    
    def is_admissible(edge):
        u, v = edge
        for (dx, dy) in SolidGridGraph.directions:
            if u[0] + dx == v[0] and u[1] + dy == v[1]:
                return True
        return False

    def order_edge(edge):
        return min(edge[0], edge[1], key=SolidGridGraph.hash_node), max(edge[0], edge[1], key=SolidGridGraph.hash_node)

    def neighbors(node, weights):
        return [(node[0] + dx, node[1] + dy) for (dx, dy) in SolidGridGraph.directions if SolidGridGraph.contains((node, (node[0] + dx, node[1] + dy)), weights)]

    def contains(edge, weights):
        return SolidGridGraph.order_edge(edge) in weights
    
    def nodes_from_weights(weights):
        return set([x for x, _ in weights] + [y for _, y in weights])

    def paths_iterator(paths, weights, filter=lambda path: len(path) == len(set(path))):
        next = []
        for path in paths:
            for n in SolidGridGraph.neighbors(path[-1], weights):
                new_path = path + [n]
                if filter(new_path):
                    next.append(new_path)
        if len(next) > 0:
            yield next
            yield from SolidGridGraph.paths_iterator(next, weights, filter=filter)
        else:
            return None

    def paths(src, weights, filter=lambda path: len(path) == len(set(path))):
        yield from SolidGridGraph.paths_iterator([[src]], weights, filter=filter)

    def cycles(src, weights):
        for paths in SolidGridGraph.paths(src, weights, filter=lambda path: len(path[:-1]) == len(set(path[:-1]))):
            yield list(filter(lambda path: path[-1] == src, paths))

    def dijkstra(src, weights):
        nodes = SolidGridGraph.nodes_from_weights(weights)
        D = {x: 0 if x == src else float('inf') for x in nodes}
        P = {src: src}
        Q = set(nodes)
        while len(Q) > 0:
            x = min(Q, key=lambda node: D[node])
            Q.remove(x)
            for y in filter(lambda y: y in Q, SolidGridGraph.neighbors(x, weights)):
                alt = D[x] + weights[SolidGridGraph.order_edge((x, y))]
                if alt < D[y]:
                    D[y] = alt
                    P[y] = x
        return D, P
        
    def shortest_path(src, dest, weights):
        _, p = SolidGridGraph.dijkstra(src, weights)
        if dest in p.keys():
            path = [dest]
            x = dest
            while x != src:
                x = p[x]
                path.append(x)
            path.reverse()
            return path
        return None

    def shortest_cycle(src, weights):
        cycle = None
        for n in SolidGridGraph.neighbors(src, weights):
            weights_ = weights.copy()
            del weights_[SolidGridGraph.order_edge((src, n))]
            sp = SolidGridGraph.shortest_path(src, n, weights_)
            if sp is not None:
                if cycle is None or len(sp) < len(cycle):
                    cycle = sp + [src]
        return cycle

    def not_holey(weights):
        nodes = SolidGridGraph.nodes_from_weights(weights)        
        for node in nodes:
            cycle_iterator = SolidGridGraph.cycles(node, weights)
            four_cycles = [[SolidGridGraph.order_edge(cycle[i:i+2]) for i in range(len(cycle)-2)] for cycle in list(zip(cycle_iterator, range(4)))[-1][0]]
            if len(four_cycles) < 4:
                for cycles in cycle_iterator:
                    for cycle in cycles:
                        cycle_edges = [SolidGridGraph.order_edge(cycle[i:i+2]) for i in range(len(cycle)-2)]
                        for four_cycle in four_cycles:
                            if len(set(four_cycle) & set(cycle_edges)) == 0:
                                return False
        return True

    def hamiltonian_cycles(weights):
        nodes = list(SolidGridGraph.nodes_from_weights(weights))
        cycle_iterator = SolidGridGraph.cycles(nodes[0], weights)
        hc_cycles = []
        for cycles in cycle_iterator:
            for cycle in filter(lambda cycle: len(set(cycle)) == len(set(nodes)), cycles):
                hc_cycles.append(cycle)
        return hc_cycles
    
    def make_face(node, dx, dy):
        
        # c - b
        # |   |
        # n - a
    
        a = (node[0] + dx, node[1])
        b = (node[0] + dx, node[1] + dy)
        c = (node[0], node[1] + dy) 

        return [SolidGridGraph.order_edge(edge) for edge in [
            (node, a),
            (node, c),
            (a, b),
            (c, b)
        ]]

    def add_random_face(weights, m, n):
        num_edges_before = len(weights)
        nodes = [(0, 0)] if len(weights) == 0 else list(SolidGridGraph.nodes_from_weights(weights))
        random.shuffle(nodes)
        while len(weights) == num_edges_before:
            node = nodes.pop()
            directions = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
            random.shuffle(directions)
            while len(directions) > 0:
                dx, dy = directions.pop()   
                if 0 <= node[0]+dx and node[0]+dx <= m and 0 <= node[1]+dy and node[1]+dy <= n:
                    face = SolidGridGraph.make_face(node, dx, dy)
                    for edge in face:
                        weights[edge] = 1
    
    def __init__(self, num_edges, m=10, n=10):
        self.weights = Weights(SolidGridGraph)
        self.m = m
        self.n = n
        self.weights = {}
        while len(self.weights) < num_edges:
            for _ in range(num_edges - len(self.weights)):
                SolidGridGraph.add_random_face(self.weights, self.m, self.n)

    def init_display(self):
        import matplotlib.pyplot as plt
        try:
            return self.fig, self.ax, self.plt
        except AttributeError:
            self.plt = plt
            self.fig, self.ax = self.plt.subplots()
            self.ax.set_axis_off()
            self.ax.set_xlim(-1, max(x for (x, _) in SolidGridGraph.nodes_from_weights(self.weights)) + 1 )
            self.ax.set_ylim(-1, max(y for (_, y) in SolidGridGraph.nodes_from_weights(self.weights)) + 1 )
            return self.fig, self.ax, self.plt

    def display_edges(self, edges, **kwargs):
        for edge in edges:
            self.plt.plot( [edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]] , **kwargs)

    def display_nodes(self, nodes, **kwargs):
        for node in nodes:
            circle = self.plt.Circle(node, 0.05, **kwargs)
            self.ax.add_patch(circle)

    def display(self):
        self.init_display()
        self.display_edges(self.weights.keys(), color='black')
        self.display_nodes(SolidGridGraph.nodes_from_weights(self.weights), color='red')
        self.plt.gca().set_aspect('equal')
        self.plt.show()


weights = Weights(SolidGridGraph)

weights[(0, 0), (1, 0)] = 1
weights[(0, 1), (1, 1)] = 1
weights[(1, 1), (2, 2)] = 1


for k in weights:
    print(k)
# print(SolidGridGraph.not_holey(sgg.weights))
# print(len(sgg.weights))
# print(sgg.weights)

# sgg = SolidGridGraph(0)
# SolidGridGraph.add_random_face(sgg.weights, 10, 10)
# SolidGridGraph.add_random_face(sgg.weights, 10, 10)
# print(len(sgg.weights))
# for edge in sgg.weights:
#     print(edge, sgg.weights[edge])

# sgg.display()
