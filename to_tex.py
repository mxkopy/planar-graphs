from graph import *
import integer_linear_model


class TikZStr(str):
    def str(self):
        return self

class TiKZOptions(dict):

    def draw(key, value, tikz):
        tikz.draw[key] = value
        return tikz

    def style(key, value, tikz):
        tikz.style[key] = value
        return tikz

    def color(color, tikz):
        return TiKZOptions.style('color', color, tikz)
    
    def intensity(n, tikz):
        return TiKZOptions.color(f"{tikz.style['color']}!{n}", tikz)

    def str(self):
        return ', '.join([f"{key}={str(self[key])}" if self[key] is not None else str(key) for key in self.keys()])

    COLOR = lambda color, tikz: TiKZOptions.color(color, tikz)
    GRAY = lambda tikz: TiKZOptions.color('gray', tikz)
    BLUE = lambda tikz: TiKZOptions.color('blue', tikz)

    INTENSITY = lambda n, tikz: TiKZOptions.intensity(n, tikz)
    LIGHT = lambda tikz: TiKZOptions.INTENSITY(30, tikz)
    VERY_LIGHT = lambda tikz: TiKZOptions.INTENSITY(10, tikz)

class TikZNode(Node):

    draw = TiKZOptions({
        # 'draw': None
    })

    style = TiKZOptions({
        'circle': None,
        'draw': None,
        'black': None,
        'solid': None,
        'fill': 'white',
        'scale': 0.5
    })

    def __new__(cls, *args, edges=[], label='', style={}):
        node = super().__new__(cls, *args, edges=edges)
        node.style = TiKZOptions({**TikZNode.style, **style})
        node.label = label
        return node

    def str(self):
        return f"\draw[{self.draw.str()}] node[{self.style.str()}] at {str(self)} {{{self.label}}};"



class TikZEdge(Edge):

    ARROW_ANGLE = 45

    DIRECTED = lambda edge: TiKZOptions.style('shorten >', '3', TiKZOptions.style('->', None, TikZEdge(*edge)))
    
    GRAY = lambda edge: TiKZOptions.color('gray', TikZEdge(*edge))

    SHORT = lambda edge: TiKZOptions.style('shorten <', '8', TikZEdge(*edge))

    # DOTTED = lambda edge: TiKZOptions.style('dash pattern', f"on 10 off 10", TikZEdge(*edge))
    DOTTED = lambda edge: TiKZOptions.style('dotted', None, TikZEdge(*edge))


    # SHORT = lambda edge: TikZEdge(
    #     *edge,
    #     style={
    #         **edge.style,
    #         'shorten <': '8'
    #     }
    # )

    # DOTTED = lambda edge: TikZEdge(
    #     *edge,
    #     style={
    #         **edge.style,
    #         'dotted': None
    #     }
    # )

    draw = TiKZOptions({
        'line width': '2pt'
    })

    style = TiKZOptions({
        'color': 'black',
    })

    def __getitem__(self, idx):
        return [self.s, self.t, self.d, self.draw, self.style][idx]
    
    # TODO: make Edge generic so this isn't necessary
    def __init__(self, s, t, d=False, draw={}, style={}):
        super().__init__(s if isinstance(s, TikZNode) else TikZNode(*s), t if isinstance(t, TikZNode) else TikZNode(*t), d=d)
        self.style = TiKZOptions({**TikZEdge.style, **style})
        self.draw = TiKZOptions({**TikZEdge.draw, **draw})

    def str(self):
        return f"\draw[{self.draw.str()}] {str(self.s)} edge[{self.style.str()}] {str(self.t)};"


class TikZGraph(Graph):
    
    def make_bipartite(self):
        for node in self.nodes:
            self.nodes[TikZNode(*node), assign]
        for edge in self:
            self[TikZEdge(*edge), assign]
        super().make_bipartite()
        for edge in self:
            if edge.s.color == 1:
                edge.s.style['fill'] = 'black'
            if edge.t.color == 1:
                edge.t.style['fill'] = 'black'

    def __init__(self, *edges):
        super().__init__()
        for edge in edges:
            self[TikZEdge(*edge), assign]
        self.edgesets = []
        self.background = []
        self.foreground = []

    def add_edges(self, *args):
        for edges in [[TikZEdge(*edge) for edge in edges] for edges in args]:
            self.edgesets.append(edges)

    def add_background(self, *args):
        self.background += args

    def add_foreground(self, *args):
        self.foreground += args

    def tikz_paths(self, exclude_previous=True):
        paths = []
        previous_edges = set()
        for edges in self.edgesets:
            for edge in edges:
                if edge.d and edge.switch() in edges:
                    edge.draw['line width'] = '1.75pt'
                    edge.style['out'] = edge.axis() + TikZEdge.ARROW_ANGLE
                    edge.style['in'] = edge.switch().axis() - TikZEdge.ARROW_ANGLE
                if edge not in previous_edges or not exclude_previous:
                    paths.append(edge.str())
            for edge in edges:
                edge.d = False
                previous_edges.add(edge)
        for node in self.nodes:
            paths.append(node.str())
        return paths

    def str(self):
        return f"{chr(92)}begin{{tikzpicture}}{chr(10)}{chr(10).join(self.background+self.tikz_paths()+self.foreground)}{chr(10)}{chr(92)}end{{tikzpicture}}"
    
    def write(self, name):
        file = open(f'../draft/examples/{name}.tex', 'w')
        file.write(self.str())
        file.close()
