from graph import *
import integer_linear_model

#TODO: write a DSL for this stuff 
# needs:
#   quickly style edges and sets of edges/nodes
#   run python at any point, importing/exporting context as needed
#   - inline python: entire line parsed wrt to python context
#   - backing graph is always called 'graph', other graphs can be referred to by name 
#   quickly define arbitrary styles that can be automated with python
# ---------------------------------------------------------------------------------------- 
# syntax:
# ----------------------------------------------------------------------------------------
# [\tex-command(;|%)]
# [python-code]
# [$...$ substitution]
# [DEFN name [arg1, arg2, arg3...]|Optional {...}]
# 
# [EDGESET|GRAPH|SGG name]
    # [EDGES ((edge.s[0],edge.s[1]),(edge.t[0],edge.t[1])[,draw[k=v|k...]|Optional][,style[k=v|k...]|Optional])[python-code|Optional]...[draw[...]|Optional][,style[...]|Optional]|Optional]
    # ...
    # [NODES (node[0],node[1][,draw[...]|Optional][,style[...]|Optional])[python-code|Optional]...[draw[...]|Optional][,style[...]|Optional]|Optional]|Optional]
    # ...
    # [SET [PROPERTY]]
    # [FOREGROUND tex-command...[python-code|Optional]|Optional]
    # [BACKGROUND tex-command...[python-code|Optional]|Optional]
# END]
# ----------------------------------------------------------------------------------------
# example:
# --------------------------------------------------------------------------------------
# EDGES draw[black, line width=1.75pt]
# NODES style[circle, draw, black, solid, fill=white, scale=0.5]
# 
# DEFN SHADE-TWO-FACTOR {BACKGROUND \fill['fill=black!10'] ({x-0.5}cm-1pt, {y-0.5}cm-1pt) rectangle ++(1cm+2pt, 1cm+2pt); for (x, y) in [midpoint_at_face(face) for face in graph.faces if graph.two_factor.test_interior(midpoint_at_face(face))]}
# DEFN MAKE-BIPARTITE {NODES (x, y, style[color=white]) for (x, y) in graph.nodes if graph.nodes[(x, y)].color == 1}
# 
# SGG alternating_strip_before_flip
    # EDGES ((1, 0), (2, 0)), ((1, 1), (2, 1)), ((1, 2), (1, 1)), ((2, 2), (2, 1)), ((1, 4), (1, 3)), ((1, 3), (2, 3)), ((2, 4), (2, 3)), ((1, 6), (1, 5)), ((1, 5), (2, 5)), ((2, 6), (2, 5)), ((1, 7), (2, 7)), ((0, 0), (1, 0)), ((2, 0), (3, 0)), ((3, 0), (3, 1)), ((3, 1), (3, 2)), ((3, 2), (2, 2)), ((1, 2), (0, 2)), ((0, 2), (0, 1)), ((0, 1), (0, 0)), ((1, 4), (0, 4)), ((0, 4), (0, 5)), ((0, 5), (0, 6)), ((0, 6), (1, 6)), ((2, 6), (3, 6)), ((3, 6), (3, 5)), ((3, 5), (3, 4)), ((3, 4), (2, 4)), ((1, 7), (1, 8)), ((1, 8), (0, 8)), ((0, 8), (0, 9)), ((0, 9), (1, 9)), ((1, 9), (2, 9)), ((2, 9), (3, 9)), ((3, 9), (3, 8)), ((3, 8), (2, 8)), ((2, 8), (2, 7)), ((2, 7), (2, 6)), ((2, 6), (1, 6)), ((1, 6), (1, 7)), ((0, 5), (1, 5)), ((1, 5), (1, 4)), ((1, 4), (2, 4)), ((2, 5), (2, 4)), ((2, 5), (3, 5)), ((1, 8), (2, 8)), ((1, 9), (1, 8)), ((2, 9), (2, 8)), ((1, 3), (1, 2)), ((1, 2), (2, 2)), ((2, 2), (2, 3)), ((0, 1), (1, 1)), ((1, 1), (1, 0)), ((2, 0), (2, 1)), ((2, 1), (3, 1)) style[draw=none]
    # SET TWO-FACTOR ((0, 0), (1, 0)), ((1, 0), (2, 0)), ((2, 0), (3, 0)), ((3, 0), (3, 1)), ((3, 1), (3, 2)), ((3, 2), (2, 2)), ((2, 2), (2, 1)), ((2, 1), (1, 1)), ((1, 1), (1, 2)), ((1, 2), (0, 2)), ((0, 2), (0, 1)), ((0, 1), (0, 0)), ((1, 3), (2, 3)), ((2, 3), (2, 4)), ((2, 4), (3, 4)), ((3, 4), (3, 5)), ((3, 5), (3, 6)), ((3, 6), (2, 6)), ((2, 6), (2, 5)), ((2, 5), (1, 5)), ((1, 5), (1, 6)), ((0, 6), (1, 6)), ((0, 6), (0, 5)), ((0, 5), (0, 4)), ((0, 4), (1, 4)), ((1, 4), (1, 3)), ((1, 7), (2, 7)), ((2, 7), (2, 8)), ((2, 8), (3, 8)), ((3, 8), (3, 9)), ((3, 9), (2, 9)), ((2, 9), (1, 9)), ((1, 9), (0, 9)), ((0, 9), (0, 8)), ((0, 8), (1, 8)), ((1, 8), (1, 7)) style[color=gray, dotted]
    # SET LONGEST-STRIP max(get_alternating_strips(graph), key=len)
    # SET LONGEST-STRIP-FLIPPED
    # EDGES edge for edge in graph.longest_strip
    # EDGES edge for edge in graph.two_factor style[color=gray, dotted]
    # NODES (x, y, style[draw=none, fill=none]) for (x, y) in graph.nodes if Node(x, y) not in graph.longest_strip
    # SHADE-TWO-FACTOR
# END
# SGG alternating_strip_after_flip
    # EDGES edge for edge in alternating_strip_before_flip style[draw=none]
    # SET TWO-FACTOR edge_flip(alternating_strip_before_flip, alternating_strip_before_flip.longest_strip).two_factor
    # EDGES edge for edge in graph.two_factor style[color=gray, dotted]
    # EDGES edge for edge in alternating_strip_before_flip.longest_strip.flipped_perimeter + alternating_strip_before_flip.longest_strip - alternating_strip_before_flip.longest_strip.perimeter
    # NODES (x, y, style[draw=none, fill=none]) for (x, y) in graph.nodes if Node(x, y) not in alternating_strip_before_flip.longest_strip_flipped
    # SHADE-TWO-FACTOR
# END
# ----------------------------------------------------------------------------------------
# how it will work:
#   works essentially as a python preprocessor 
#   keywords (all caps, style[], draw[], etc.) are expanded into python
#   - python will keep track of variables, so the context is always python's
#   -- `graph` is a special variable that will always refer to the graph defined in the current scope
#   -- TZSL functions are expanded lazily, so variable names resolve at the last possible moment
#   - DEFN defines new macros which are added dynamically at compile-time to the list of keywords to be matched
# the ast will be lispy. 
#   - nodes have the form SYMBOL(DATA...), where DATA is a field that contains arguments to functions in the front or otherwise other AST nodes. 
#   -- ARGS cannot contain keywords, we rely on keywords to delineate ARGS from DATA. 
#   - to enable delineation of tex commands, backslashes \ must be escaped (\\ will result in \ instead of being read as a tex command)
#   -- since ending tex can be ambiguous, either a percent % or semicolon ; signify the end of a tex command (again, escape with \). 
#   - dollar signs $ are treated as brackets in f-strings, substituting strings for their value after being evaluated in python
#   - data lower in the AST has lower scope. 
#   -- this means EDGE-STYLE and NODE-STYLE parameters at a higher scope provide default values for style[] draw[] expansions at lower scopes
#   -- there will be a special toplevel scope called ROOT which is globally available at compile-time
#
# the compilation process will be as follows:
#   1. pattern match against TZSL keywords via regex (e.g. GRAPH (name)?, (EdgeSet|Graph|SolidGridGraph)?, .*) to build AST
#      - split text by TZSL keywords, then build AST by applying scopes
#   2. lower AST to python 
#   3. repeat 1-3 until no keywords are left.
import re
from functools import partial

GRAPH_TYPES = {
    'EDGESET': EdgeSet,
    'GRAPH': Graph,
    'SGG': SolidGridGraph
}

KEYWORDS = [
    "DEFN",
    "\\\\",
    "{",
    "}",
    "EDGESET",
    "GRAPH",
    "SGG",    
    "EDGES",
    "NODES",
    "FOREGROUND",
    "BACKGROUND",
    "SET",
    # "\t",
    "\n",
    "END",
    "TEX",
    "\;",
    "\%",
    # "\\$(\.\*\?)\\$"
]

class ASTNode:

    def __init__(self, **args):
        self.args = args
        self.data = []
        # self.consume = self.__consume__()

    def __eq__(self, other):
        if not isinstance(other, ASTNode):
            return False
        return self.data == other.data and self.args == other.args
    
    # consume text, getting output symbol and unconsumed/scoped text
    # add output symbol to the root node's data, then recurse on the next set of tokens
    # 
    def __consume__(self):
        def consume(root, tokens):
            if len(tokens) == 0:
                return root, []
            F = self.__class__.consume(root, tokens)
            if F is not None:
                return F
            root.data.append(tokens[0])
            return self.consume(root, tokens[1:])
        return consume

class SCOPE(ASTNode):

    def consume(root, tokens):
        if len(tokens) == 0:
            return root, []
        match tokens[0]:
            case '{':
                node, rest_tokens = SCOPE.consume(SCOPE(), tokens[1:])
                root.data.append(node)
                return SCOPE.consume(root, rest_tokens)
            case '}':
                return root, tokens[1:]
            case _:
                root.data.append(tokens[0])
                return SCOPE.consume(root, tokens[1:])

class DEFN(ASTNode):

    def get_args(tokens):
        args = re.match(r'(\w+)\s*(.*)', tokens)
        return {
            'name': args.group(1), 
            'args': re.split(r'\s+', args.group(2) if args.group(2) is not None else [])
        }

    def consume(root, tokens):
        if len(tokens) == 0:
            return root, []
        match tokens[0]:
            case 'DEFN':
                node = DEFN()
                node.data += tokens[1:3]
                root.data.append(node)
                return DEFN.consume(root, tokens[3:])
            case _:
                root.data.append(tokens[0])
                return DEFN.consume(root, tokens[1:])

class GRAPH(ASTNode):
    
    def get_args(tokens):
        return {
            'type': GRAPH_TYPES[tokens[0]], 
            'name': tokens[1]
        }

    def consume(root, tokens):
        if len(tokens) == 0:
            return root, []
        match tokens[0]: 
            case 'EDGESET' | 'GRAPH' | 'SGG':
                node = GRAPH()
                node.data.append(tokens[1])
                node, rest_tokens = GRAPH.consume(node, tokens[2:])
                root.data.append(node)
                return GRAPH.consume(root, rest_tokens)
            case 'END':
                return root, tokens[1:]
            case _:
                root.data.append(tokens[0])
                return GRAPH.consume(root, tokens[1:])

class EDGES(ASTNode):
    def consume(root, tokens):
        if len(tokens) == 0:
            return root, []
        match tokens[0]:
            case 'EDGES':
                node = EDGES()
                node.data.append(tokens[1])
                root.data.append(node)
                return EDGES.consume(root, tokens[2:])
            case _:
                root.data.append(tokens[0])
                return EDGES.consume(root, tokens[1:])

class NODES(ASTNode):
    def consume(root, tokens):
        if len(tokens) == 0:
            return root, []
        match tokens[0]:
            case 'NODES':
                node = NODES()
                node.data.append(tokens[1])
                root.data.append(node)
                return NODES.consume(root, tokens[2:])
            case _:
                root.data.append(tokens[0])
                return NODES.consume(root, tokens[1:])

class SET(ASTNode):
    def consume(root, tokens):
        if len(tokens) == 0:
            return root, []
        match tokens[0]:
            case 'SET':
                node = SET()
                node.data.append(tokens[1])
                root.data.append(node)
                return SET.consume(root, tokens[2:])
            case _:
                root.data.append(tokens[0])
                return SET.consume(root, tokens[1:])

class FOREGROUND(ASTNode):
    def consume(root, tokens):
        if len(tokens) == 0:
            return root, []
        match tokens[0]:
            case 'FOREGROUND':
                node = FOREGROUND()
                node.data.append(tokens[1])
                root.data.append(node)
                return FOREGROUND.consume(root, tokens[2:])
            case _:
                root.data.append(tokens[0])
                return FOREGROUND.consume(root, tokens[1:])

class BACKGROUND(ASTNode):
    def consume(root, tokens):
        if len(tokens) == 0:
            return root, []
        match tokens[0]:
            case 'BACKGROUND':
                node = BACKGROUND()
                node.data.append(tokens[1])
                root.data.append(node)
                return BACKGROUND.consume(root, tokens[2:])
            case _:
                root.data.append(tokens[0])
                return BACKGROUND.consume(root, tokens[1:])

class TEX(ASTNode):
    def consume(root, tokens):
        if len(tokens) == 0:
            return root, []
        match tokens[0]:
            case '\\' | 'TEX':
                node = TEX()
                if tokens[0] == '\\':
                    node.data.append(True)
                node, rest_tokens = SCOPE.consume(node, tokens[1:])
                root.data.append(node)
                return TEX.consume(root, rest_tokens)
            case ';' | '%':
                if tokens[0] == ';':
                    root.data.append(True)
                return root, tokens[1:]
            case _:
                root.data.append(tokens[0])
                return TEX.consume(root, tokens[1:])

class Parser:

    def __init__(self):
        self.KEYWORDS = KEYWORDS
        self.AST_NODES = [TEX, NODES, EDGES, FOREGROUND, BACKGROUND, GRAPH, SET, SCOPE]

    def tokenize(self, text):
        is_whitespace = lambda text: re.fullmatch(r"\s*", text) is not None # check if a token is only whitespace
        remove_trailing_whitespace = lambda text: text if re.match(r"\s*(.*?)\s*$", text) is None else re.match(r"\s*(.*?)\s*$", text).group(1) # get rid of leading and trailing whitespace
        keyword_regexes = [f"(?!{chr(92)}{chr(92)}{keyword}){keyword}" for keyword in self.KEYWORDS] # allow for escaping keywords
        tokens = list(remove_trailing_whitespace(phrase) for phrase in re.split(f"({'|'.join(keyword_regexes)})", text) if not is_whitespace(phrase))
        return tokens

    def retokenize(self, root):
        data = []
        for x in root.data:
            if isinstance(x, ASTNode):
                self.retokenize(x)
                data.append(x)
            if isinstance(x, str):
                tokens = self.tokenize(x)
                data += tokens
        root.data = data

    def parse(self, root, T):
        scope, unparsed = T.consume(ASTNode(), root.data)
        root.data = scope.data + unparsed
        for data in root.data:
            if isinstance(data, ASTNode):
                self.parse(data, T)
        return root

    def add_definitions(self, root):
        for x in root.data:
            if isinstance(x, DEFN):
                name = x.data[0]
                scope = x.data[-1]
                self.KEYWORDS.append(name)
                node_type = type(name, (ASTNode,), {})
                self.AST_NODES.append(node_type)
                def consume(root, tokens, name=name, scope=scope, node_type=node_type):
                    if len(tokens) == 0:
                        return root, []
                    if isinstance(tokens[0], DEFN):
                        return node_type.consume(root, tokens[1:])
                    if tokens[0] == name:
                        root.data.append(scope)
                        return node_type.consume(root, tokens[1:])
                    root.data.append(tokens[0])
                    return node_type.consume(root, tokens[1:])
                node_type.consume = consume
            elif isinstance(x, ASTNode):
                self.add_definitions(x)

    def expand_ast(self, root):
        node = ASTNode()
        while root != node:
            for T in self.AST_NODES:
                root = self.parse(root, T)
            root = self.parse(root, DEFN)
            node.args = root.args
            node.data = root.data
            self.add_definitions(root)
            # self.retokenize(root)
            for T in self.AST_NODES:
                root = self.parse(root, T)
        return root

    def ast(self, text):
        tokens = self.tokenize(text)
        root = SCOPE()
        root.data += tokens
        return self.expand_ast(root)

p = Parser()
import os
print(os.getcwd())
file = open('tzsl.test', 'r')
root = p.ast(str(file.read()))
file.close()
# print(root.data[2].data[1].data)
# print(root.data)
def print_ast(node, s='', lvl=0):
    if isinstance(node, ASTNode):
        print(f"{'-'*lvl}{node.__class__.__name__} {node.args}")
        if isinstance(node.data, list):
            for x in node.data:
                if isinstance(x, ASTNode):
                    print_ast(x, s, lvl+1)
                else:
                    print(f"{'-'*(lvl+1)}{x}")
        else:
            print(f"{'-'*(lvl+1)}{node.data}")
        return s

print_ast(root)

class Interpreter:

    def __init__(self):
        self.ctx = {}
    



class TikZStr(str):
    def str(self):
        return self

class TikZOptions(dict):

    def draw(key, value, tikz):
        tikz.draw[key] = value
        return tikz

    def style(key, value, tikz):
        tikz.style[key] = value
        return tikz

    def color(color, tikz):
        return TikZOptions.style('color', color, tikz)
    
    def intensity(n, tikz):
        return TikZOptions.color(f"{tikz.style['color']}!{n}", tikz)

    def str(self):
        return ', '.join([f"{key}={str(self[key])}" if self[key] is not None else str(key) for key in self.keys()])

    COLOR = lambda color, tikz: TikZOptions.color(color, tikz)
    GRAY = lambda tikz: TikZOptions.color('gray', tikz)
    BLUE = lambda tikz: TikZOptions.color('blue', tikz)

    INTENSITY = lambda n, tikz: TikZOptions.intensity(n, tikz)
    LIGHT = lambda tikz: TikZOptions.INTENSITY(30, tikz)
    VERY_LIGHT = lambda tikz: TikZOptions.INTENSITY(10, tikz)

class TikZNode(Node):

    draw = TikZOptions({
        # 'draw': None
    })

    style = TikZOptions({
        'circle': None,
        'draw': None,
        'black': None,
        'solid': None,
        'fill': 'black!70',
        'scale': 0.5
    })

    def __new__(cls, *args, edges=[], label='', style={}):
        node = super().__new__(cls, *args, edges=edges)
        node.style = TikZOptions({**TikZNode.style, **style})
        node.label = label
        return node

    def str(self):
        return f"\draw[{self.draw.str()}] node[{self.style.str()}] at {str(self)} {{{self.label}}};"

class TikZEdge(Edge):

    ARROW_ANGLE = 45

    DIRECTED = lambda edge: TikZOptions.style('shorten >', '3', TikZOptions.style('->', None, TikZEdge(*edge)))
    
    GRAY = lambda edge: TikZOptions.color('gray', TikZEdge(*edge))

    SHORT = lambda edge: TikZOptions.style('shorten <', '8', TikZEdge(*edge))

    # DOTTED = lambda edge: TikZOptions.style('dash pattern', f"on 10 off 10", TikZEdge(*edge))
    DOTTED = lambda edge: TikZOptions.style('dotted', None, TikZEdge(*edge))


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

    draw = TikZOptions({
        'line width': '2pt'
    })

    style = TikZOptions({
        'color': 'black',
    })

    def __getitem__(self, idx):
        return [self.s, self.t, self.d, self.draw, self.style][idx]
    
    # TODO: make Edge generic so this isn't necessary
    def __init__(self, s, t, d=False, draw={}, style={}):
        super().__init__(s if isinstance(s, TikZNode) else TikZNode(*s), t if isinstance(t, TikZNode) else TikZNode(*t), d=d)
        self.style = TikZOptions({**TikZEdge.style, **style})
        self.draw = TikZOptions({**TikZEdge.draw, **draw})

    def str(self):
        return f"\draw[{self.draw.str()}] {str(self.s)} edge[{self.style.str()}] {str(self.t)};"


class TikZGraph(Graph):
    
    def make_bipartite(self):
        super().make_bipartite()
        for edge in self:
            if edge.s.color == 1:
                edge.s.style['fill'] = 'white'
            if edge.t.color == 1:
                edge.t.style['fill'] = 'white'
        return self

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
        return self

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
        print(f"wrote {name}")
