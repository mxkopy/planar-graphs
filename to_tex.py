from graph import *
import integer_linear_model

class TikZStr(str):
    def str(self):
        return self

class TiKZOptions(dict):
    def str(self):
        return ', '.join([f"{key}={str(self[key])}" if self[key] is not None else str(self[key]) for key in self.keys()])

class TikZNode(Node):

    style = TiKZOptions({
        'fill': 'white'
    })

    def __new__(cls, *args, edges=[], label='', style={}):
        node = super().__new__(cls, *args, edges=edges)
        node.style = TiKZOptions({**TikZNode.style, **style})
        node.label = label
        return node

    def str(self):
        return f"{str(self)} node[{self.style.str()}] {{{self.label}}}"

class TikZEdge(Edge):

    drawargs = TiKZOptions({
        'line width': '2pt'
    })

    style = TiKZOptions({
        'color': 'black',
        'shorten >': '3',
        'shorten <': '0.5'
    })

    def __init__(self, s, t, w=True, d=False, drawargs={}, style={}):
        # TODO: make Edge generic so we don't have to do this
        s = TikZNode(*s) if not isinstance(s, TikZNode) else s
        t = TikZNode(*t) if not isinstance(t, TikZNode) else t
        super().__init__(s, t, w, d)
        self.style = TiKZOptions({**TikZEdge.style, **style})
        self.drawargs = TiKZOptions({**TikZEdge.drawargs, **drawargs})

    def str(self):
        return f"\draw[{self.drawargs.str()}] {self.s.str()} edge[{self.style.str()}] {str(self.t)} {self.t.str()};"


class TeXGraph(Graph):

    ARROW_ANGLE = 45

    UNDIRECTED_BLACK = lambda edge: TikZEdge(*edge)

    DIRECTED = lambda edge: TikZEdge(
        edge.s,
        edge.t,
        w=edge.w,
        d=True,
    )

    GRAY = lambda edge: TikZEdge(
        *edge,
        stle={
            'color': 'gray'
        }
    )

    SHORT = lambda edge: TikZEdge(
        *edge,
        style=edge.style.update({
            'shorten <': '8'
        })
    )

    DOTTED = lambda edge: TikZEdge(
        *edge,
        style=edge.style.update({
            'dotted': None
        })
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edgesets = []

    def add_edges(self, *args):
        for edges in args:
            self.edgesets.append(edges)

    def tikz_paths(self, exclude_previous=True):
        paths = []
        previous_edges = set([])
        for edges in self.edgesets:
            for edge in edges:
                if exclude_previous and edge not in previous_edges:
                    paths.append(edge.str())
                previous_edges.add(edge)
        return paths

    def str(self):
        return f"{chr(92)}begin{{tikzpicture}}{chr(10)}{chr(10).join(self.tikz_paths())}{chr(10)}{chr(92)}end{{tikzpicture}}"
    
    def write(self, name):
        file = open(f'../draft/examples/{name}.tex', 'w')
        file.write(self.str())
        file.close()


# pass in list of callback functions to determine the styles for each set of edges
# i.e. [(['red, dashed'], getHamiltonianPath), (['blue', getShortestPath(a, b)])]

def getEdges(graph):
    return graph

def getTwoFactor(graph):
    return integer_linear_model.find_two_factor(graph)

def getHamiltonianCycle(graph):
    hc_cycles = graph.hamiltonian_cycles()
    return None if len(hc_cycles) == 0 else hc_cycles[0]

def getHamiltonianPath(src):
    def getPath(graph):
        hc_paths = graph.hamiltonian_paths(src)
        return None if len(hc_paths) == 0 else hc_paths[0]
    return getPath

def getTravellingSalesmanTour(graph):
    return graph.travelling_salesman_tours()[0]

def getShortestPath(src, dst):
    def getPath(graph):
        pass
    return getPath

def make_tex(graph, styles=[], line_width='2pt'):

    ARROW_ANGLE = 45

    solid, empty, _ = graph.make_bipartite()
    solid, empty = set(solid), set(empty)

    def edge_str(edge, style=[], line_width=line_width):
        return f"\draw[line width={line_width}] {str(edge.s)} node[fill={'white' if edge.s in empty else 'black'}]{{}} edge[shorten <= 0.5, shorten >= 3, {', '.join(style)}] {str(edge.t)} {str(edge.t)} node[fill={'black' if edge.t in solid else 'white'}]{{}};"

    previous_edges = set()
    edge_strings = []

    for style, get_edges in styles:
        edges = set(get_edges(graph)).difference(previous_edges)
        for edge in edges:
            previous_edges.add(Edge(edge.s, edge.t, w=edge.w, d=False))
            if edge.d and edge.switch() in edges:
                edge_strings.append(edge_str(edge,          style=[f'out={edge.axis()+ARROW_ANGLE}', f'in={edge.switch().axis()-ARROW_ANGLE}', *style], line_width='1.75pt'))
                edge_strings.append(edge_str(edge.switch(), style=[f'out={edge.switch().axis()+ARROW_ANGLE}', f'in={edge.axis()-ARROW_ANGLE}', *style], line_width='1.75pt'))
            else:
                edge_strings.append(edge_str(edge, style=style))


    return f'{chr(92)}begin{{tikzpicture}}{chr(10)}{chr(10).join(edge_strings)}{chr(10)}{chr(92)}end{{tikzpicture}}'


def write_tex(name, graph, styles=[(['black'], getEdges), (['red'], getTwoFactor)]):
    file = open(f'../draft/examples/{name}.tex', 'w')
    tex  = make_tex(graph, styles=styles) 
    file.write(tex)
    file.close()


# example_graph_length = 2
# example_graph = TeXGraph(edges=[((0, 0), (example_graph_length, 0)), ((example_graph_length, 0), (example_graph_length / 2, ((example_graph_length ** 2) - ((example_graph_length / 2) ** 2))**0.5 )), ((example_graph_length / 2, ((example_graph_length ** 2) - ((example_graph_length / 2) ** 2))**0.5), (0, 0))])
# example_graph.add_edges(
#     [
#         TeXGraph.UNDIRECTED_BLACK(edge) for edge in example_graph.keys()
#     ]
# )
# example_graph.write('graph')
