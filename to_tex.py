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



# class FOREGROUND(ASTNode):

#     keywords = ['FOREGROUND']

#     def consume(data, tokens):
#         match tokens[0]:
#             case 'FOREGROUND':
#                 node = FOREGROUND()
#                 node.data.append(tokens[1])
#                 data.append(node)
#                 return FOREGROUND.consume(data, tokens[2:])

# class BACKGROUND(ASTNode):

#     keywords = ['BACKGROUND']

#     def consume(data, tokens):
#         match tokens[0]:
#             case 'BACKGROUND':
#                 node = BACKGROUND()
#                 node.data.append(tokens[1])
#                 data.append(node)
#                 return BACKGROUND.consume(data, tokens[2:])

# class TEX(ASTNode):

#     keywords = ['\\', 'TEX', ';', '%']

#     def consume(data, tokens):
#         match tokens[0]:
#             case '\\' | 'TEX':
#                 node = TEX()
#                 if tokens[0] == '\\':
#                     node.data.append(True)
#                 _, rest_tokens = TEX.consume(node.data, tokens[1:])
#                 data.append(node)
#                 return TEX.consume(node.data, rest_tokens)
#             case ';' | '%':
#                 if tokens[0] == ';':
#                     data.append(True)
#                 return data, tokens[1:]


import re
import ast

class MetaASTNode(type):

    def __consume__(cls, cls_consume):
        def consume(data, tokens):
            if len(tokens) == 0:
                return data, []
            F = cls_consume(data, tokens)
            if F is not None:
                return F
            data.append(tokens[0])
            return cls.consume(data, tokens[1:])
        return consume

    def __new__(meta_cls, name, parents, attrs):
        new_cls = super().__new__(meta_cls, name, parents, attrs)
        new_cls.consume = MetaASTNode.__consume__(new_cls, new_cls.consume)
        return new_cls


class ASTNode(metaclass=MetaASTNode):

    def __init__(self):
        self.data = []
        self.ctx = globals().copy()
        self.tikz_ctx = {}

    def __iter__(self):
        yield self
        for x in self.data:
            if isinstance(x, ASTNode):
                yield from iter(x)
            else:
                yield x

    def __eq__(self, other):
        if other == None:
            return False
        X, Y = iter(self), iter(other)
        x = next(X, None)
        y = next(Y, None)
        while x is not None and y is not None:
            x = next(X, None)
            y = next(Y, None)
            if x != y:
                return False
        return True
    
    # consume text, getting output symbol and unconsumed/scoped text
    # add output symbol to the root node's data, then recurse on the remaining set of tokens
    def consume(data, tokens):
        if len(tokens) == 0:
            return data, []
        else:
            data.append(tokens[0])
            return ASTNode.consume(data, tokens[1:])

    @classmethod
    def parse(cls, root):
        if isinstance(root, list):
            new_data, _ = cls.consume([], root)
            return new_data
        else:
            root.data, _ = cls.consume([], root.data)
            for x in root.data:
                if isinstance(x, ASTNode):
                    cls.parse(x)
            return root
        
    # returns the name of a variable and its RHS
    def exec_str(self):
        return (None, '')

    def eval(self):
        results = []
        for child in self.data:
            if isinstance(child, ASTNode):
                child.ctx = self.ctx
                results.append(child.eval())
        return results

    def __str__(self):
        if not hasattr(self, 'depth'):
            setattr(self, 'depth', 1)
        # arg_str = ' '.join(f"{k}={v}" for k, v in self.args.values())
        s = f"{self.__class__.__name__}\n"
        for i in range(len(self.data)):
            data = self.data[i]
            if isinstance(data, ASTNode):
                setattr(data, 'depth', self.depth+1)
            delim = ''.join([f'|   ']*(self.depth-1)+['|--'])
            s += f"{delim} {str(data)}"
            if isinstance(data, ASTNode):
                delattr(data, 'depth')
            s += '\n' if s[-1] != '\n' else ''
        return s

    @property
    def draw_ctx(self):
        if not hasattr(self, '_draw_ctx'):
            setattr(self, '_draw_ctx', None)
            for child in self.data:
                if isinstance(child, TIKZ_CTX) and child.type == 'draw':
                    setattr(self, '_draw_ctx', child)
        return getattr(self, '_draw_ctx')
    
    @draw_ctx.setter
    def draw_ctx(self, value):
        self._draw_ctx = value
    
    @property
    def style_ctx(self):
        if not hasattr(self, '_style_ctx'):
            setattr(self, '_style_ctx', None)
            for child in self.data:
                if isinstance(child, TIKZ_CTX) and child.type == 'style':
                    setattr(self, '_style_ctx', child)
        return getattr(self, '_style_ctx')
    
    @style_ctx.setter
    def style_ctx(self, value):
        self._style_ctx = value


class TIKZ_CTX(ASTNode):

    keywords = []

    re = re.compile(r'.*?(draw|style)\[(.*?)\]')

    @property
    def type(self):
        return self.data[0]

    @property
    def args(self):
        return self.data[1]

    def extract_args(string):
        args = string.split(',')
        return tuple({
            Parser.strip_whitespace(arg.split('=')[0]): Parser.strip_whitespace(arg.split('=')[1]) 
        } if '=' in arg else Parser.strip_whitespace(arg)
        for arg in args)

    def consume(data, tokens):
        if not isinstance(tokens[0], str) or TIKZ_CTX.re.match(tokens[0]) is None:
            data.append(tokens[0])
            return TIKZ_CTX.consume(data, tokens[1:])
        else:
            node = TIKZ_CTX()
            match = TIKZ_CTX.re.match(tokens[0])
            node.data += [match.group(1), TIKZ_CTX.extract_args(match.group(2))]
            new_string = re.sub(r'\s*(draw|style)\[(.*?)\]\s*', '', tokens[0])
            if not Parser.is_whitespace(new_string):
                data.append(new_string)
            data.append(node)
            return TIKZ_CTX.consume(data, tokens[1:])

class NODES(ASTNode):

    keywords = ['NODES']

    def eval(self):
        if isinstance(self.data[0], str):
            node_set = setdict()
            nodes = eval(f"({self.data[0]})", self.ctx)
            for node in nodes:
                node_set[node, assign]
            return node_set

    def consume(data, tokens):
        match tokens[0]:
            case 'NODES':
                node = NODES()
                node.data.append(tokens[1])
                data.append(node)
                return NODES.consume(data, tokens[2:])

class EDGES(ASTNode):

    keywords = ['EDGES']

    def eval(self):
        if isinstance(self.data[0], str):
            edge_set = EdgeSet()
            edges = eval(f"({self.data[0]})", self.ctx)
            for edge in edges:
                edge_set[edge, assign]
            return edge_set

    def consume(data, tokens):
        match tokens[0]:
            case 'EDGES':
                node = EDGES()
                node.data.append(tokens[1])
                data.append(node)
                return EDGES.consume(data, tokens[2:])


class SET(ASTNode):

    keywords = ['SET']

    key_regex = re.compile(r'\s*([\w\-]+)\s*')

    @property
    def key(self):
        return self.data[0]
    
    @property
    def value(self):
        return self.data[1]

    def exec_str(self):
        if isinstance(self.value, ASTNode):
            return (self.key, f"{self.value.eval_str()})")
        else:
            return (self.key, f"{self.value})")

    def eval(self):
        if isinstance(self.value, ASTNode):
            return self.value.eval()
        if isinstance(self.value, str):
            return eval(self.value, self.ctx)

    def consume(data, tokens):
        match tokens[0]:
            case 'SET':
                node = SET()
                # if the key is the entire string, set the value to the next token
                if SET.key_regex.fullmatch(tokens[1]) is not None:
                    key, value = SET.key_regex.match(tokens[1]), tokens[2]
                    key = key.group(1).lower().replace('-', '_')
                    node.data += [key, value]
                    data.append(node)
                    return SET.consume(data, tokens[3:])
                # otherwise set the value to the rest of the string
                else:
                    key = SET.key_regex.match(tokens[1])
                    value = tokens[1][key.end():]
                    key = key.group(1).lower().replace('-', '_')
                    node.data += [key, value]
                    data.append(node)
                    return SET.consume(data, tokens[2:])


class GRAPH(ASTNode):
    
    keywords = ['GRAPH', 'SGG', 'END']

    TYPES = {
        'GRAPH': 'Graph',
        'SGG': 'SolidGridGraph'
    }

    @property
    def type(self):
        return self.data[0]

    @property
    def name(self):
        return self.data[1]
    
    @property
    def children(self):
        return self.data[2:]
    
    def eval_str(self):
        S = [
            f"{self.name} = {self.type}",
            f"graph = {self.name}"
        ]
        for child in self.data:
            if isinstance(child, SET):
                S.append('graph.' + child.eval_str())

        S += [child.eval_str() for child in self.data if isinstance(child, ASTNode)]

    
    def eval(self):
        self.ctx['graph'] = eval(f"{self.type}()", self.ctx)
        self.ctx[f"{self.name}"] = self.ctx['graph']
        setattr(self.ctx['graph'], 'edge_sets', [])
        setattr(self.ctx['graph'], 'node_sets', [])
        for child in self.children:
            if isinstance(child, ASTNode):
                child.ctx = self.ctx
                if isinstance(child, EDGES):
                    edge_sets = getattr(self.ctx['graph'], 'edge_sets')
                    edge_set = child.eval()
                    edge_sets.append(edge_set)
                    for edge in edge_set:
                        self.ctx['graph'][edge, assign]
                if isinstance(child, NODES):
                    node_sets = getattr(self.ctx['graph'], 'node_sets')
                    node_set = child.eval()
                    node_sets.append(node_set)
                    for node in node_set:
                        self.ctx['graph'][node, assign]
                if isinstance(child, SET):
                    setattr(self.ctx['graph'], child.key, child.eval())
        return self.ctx['graph']

    def consume(data, tokens):
        match tokens[0]: 
            case 'GRAPH' | 'SGG':
                node = GRAPH()
                node.data += [GRAPH.TYPES[tokens[0]], tokens[1]]
                _, rest_tokens = GRAPH.consume(node.data, tokens[2:])
                data.append(node)
                return GRAPH.consume(data, rest_tokens)
            case 'END':
                return data, tokens[1:]


class SCOPE(ASTNode):

    keywords = ['{', '}', '\n']

    def consume(data, tokens):
        match tokens[0]:
            case '{':
                node = SCOPE()
                _, rest_tokens = SCOPE.consume(node.data, tokens[1:])
                data.append(node)
                return SCOPE.consume(data, rest_tokens)
            case '}':
                return data, tokens[1:]

class DEFN(ASTNode):
    
    keywords = ['DEFN']

    @property
    def name(self):
        return self.data[0]
    
    @property
    def scope(self):
        return self.data[1]

    def consume(data, tokens):
        match tokens[0]:
            case 'DEFN':
                node = DEFN()
                node.data += tokens[1:3]
                data.append(node)
                return DEFN.consume(data, tokens[3:])

    def register(self, ast_nodes):
        @classmethod
        def consume(cls, data, tokens, name=self.name, scope=self.scope):
            if isinstance(tokens[0], DEFN):
                return cls.consume(data, tokens[1:])
            if tokens[0] == name:
                data += scope.data
                return cls.consume(data, tokens[1:])
        node_type = MetaASTNode.__new__(MetaASTNode, self.name, (ASTNode,), {'consume': consume, 'keywords': [self.name]})
        ast_nodes.append(node_type)


class Parser:

    AST_NODES = [NODES, EDGES, GRAPH, SET, TIKZ_CTX, SCOPE, DEFN]

    @property
    def KEYWORDS(self):
        if not hasattr(self, '_ast_nodes_len') or self._ast_nodes_len != len(self.AST_NODES):
            setattr(self, '_ast_nodes_len', len(self.AST_NODES))
            setattr(self, '_keywords', sum((node_type.keywords for node_type in self.AST_NODES), []))
        return self._keywords

    def __init__(self):
        self.AST_NODES = Parser.AST_NODES.copy()

    def escape_keyword(keyword):
        match keyword:
            case '\\':
                return '\\\\'
            case _:
                return keyword

    def is_whitespace(text):
        return re.fullmatch(r"\s*", text) is not None
    
    # removes leading and trailing whitespace
    def strip_whitespace(text):
        return text if re.match(r"\s*(.*?)\s*$", text) is None else re.match(r"\s*(.*?)\s*$", text).group(1)        

    def tokenize(self, text):
        keyword_regexes = [f"(?!{chr(92)}{chr(92)}{Parser.escape_keyword(keyword)}){Parser.escape_keyword(keyword)}" for keyword in self.KEYWORDS] # allow for escaping keywords
        tokens = list(Parser.strip_whitespace(phrase) for phrase in re.split(f"({'|'.join(keyword_regexes)})", text) if not Parser.is_whitespace(phrase))
        return tokens

    def expand_ast(self, root):
        for T in self.AST_NODES:
            T.parse(root)
    
    def add_definitions(self, root):
        for node in root:
            if isinstance(node, DEFN):
                node.register(self.AST_NODES)
    
    def next_unresolved_keyword(self, root):
        for node in root:
            if isinstance(node, str) and node in self.KEYWORDS:
                return node
        return None

    def ast(self, text):
        root = ASTNode()
        root.data = self.tokenize(text)
        while self.next_unresolved_keyword(root) is not None:
            self.expand_ast(root)
            self.add_definitions(root)
        return root


# IDEALLY
# step 1. compile the graph into one python file
# step 2. compile tikz strings from ast+python file
# - run python file, get each variable by ID, store it into AST
# - walk through the AST, this time creating TikZ objects

# CURRENTLY
# step 1. compile 

# p = Parser()
# import os
# print(os.getcwd())
# file = open('tzsl.test', 'r')
# root = p.ast(str(file.read()))
# print(root)
# root.eval()
# file.close()



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

    def copy(self):
        return TikZOptions(**super().copy())

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

    def __new__(cls, *args, label='', style={}):
        if isinstance(args[0], Node):
            node = TikZNode(*args[0])
            diff_update_dict(node.__dict__, args[0].__dict__)
            node.style = TikZOptions({**TikZNode.style, **style})
            node.label = label
            return node
        else:
            node = super().__new__(cls, *args)
            node.style = TikZOptions({**TikZNode.style, **style})
            node.label = label
            return node

    def str(self):
        return f"\draw[{self.draw.str()}] node[{self.style.str()}] at {str(self)} {{{self.label}}};"
    
    def copy(self):
        node = super().copy()
        node.style = self.style.copy()
        node.draw = self.draw.copy()
        return node
    
class TikZEdge(Edge):

    ARROW_ANGLE = 45

    DIRECTED = lambda edge: TikZOptions.style('shorten >', '3', TikZOptions.style('->', None, TikZEdge(edge)))
    
    GRAY = lambda edge: TikZOptions.color('gray', TikZEdge(edge))

    SHORT = lambda edge: TikZOptions.style('shorten <', '8', TikZEdge(edge))

    DOTTED = lambda edge: TikZOptions.style('dotted', None, TikZEdge(edge))

    draw = TikZOptions({
        'line width': '2pt'
    })

    style = TikZOptions({
        'color': 'black',
    })
    
    def __init__(self, *args, force=False, **kwargs):
        if isinstance(args[0], Edge):
            edge = args[0]
            diff_update_dict(edge.__dict__, args[0].__dict__)
        elif len(args) == 1:
            edge = Edge(*args[0])
        else:
            edge = Edge(*args, **kwargs)
        self.style = TikZEdge.style.copy()
        self.draw = TikZEdge.draw.copy()
        if isinstance(edge, TikZEdge):
            self.style |= edge.style
            self.draw |= edge.draw
        s = TikZNode(edge.s) if isinstance(edge.s, Node) else TikZNode(*edge.s)
        t = TikZNode(edge.t) if isinstance(edge.t, Node) else TikZNode(*edge.t)
        self.force = force
        super().__init__(s, t)

    def str(self):
        return f"\draw[{self.draw.str()}] {str(self.s)} edge[{self.style.str()}] {str(self.t)};"
    
    def copy(self):
        edge = super().copy()
        edge.style = self.style.copy()
        edge.draw = self.draw.copy()
        return edge

# @opunpack
class TikZGraph(Graph):
    
    def make_bipartite(self):
        super().make_bipartite()
        for edge in self:
            if self[edge.s].color == 1:
                self[edge.s].style['fill'] = 'white'
            if self[edge.t].color == 1:
                self[edge.t].style['fill'] = 'white'
        return self

    def __init__(self, *edges):
        super().__init__(*(TikZEdge(edge) for edge in edges))
        # for edge in edges:
            # self[TikZEdge(edge), assign]
        self.edgesets = []
        self.background = []
        self.foreground = []

    def add_edges(self, *args):
        for edges in [[TikZEdge(edge) if not isinstance(edge, TikZEdge) else edge for edge in edges] for edges in args]:
            self.edgesets.append(edges)
        return self

    def add_background(self, *args):
        self.background += args
        return self

    def add_foreground(self, *args):
        self.foreground += args
        return self

    def tikz_paths(self):
        paths = []
        previous_edges = set()
        for edges in self.edgesets:
            for edge in edges:
                if edge.switch() in edges:
                    edge.draw['line width'] = '1.75pt'
                    edge.style['out'] = edge.axis() + TikZEdge.ARROW_ANGLE
                    edge.style['in'] = edge.switch().axis() - TikZEdge.ARROW_ANGLE
                if edge not in previous_edges or edge.force:
                    paths.append(edge.str())
            for edge in edges:
                previous_edges.add(edge)
                previous_edges.add(edge.switch())
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
        return self
