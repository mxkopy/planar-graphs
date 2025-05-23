{-
TODO: write a DSL for this stuff 
needs:
  quickly style edges and sets of edges/nodes
  run python at any point, importing/exporting context as needed
  - inline python: entire line parsed wrt to python context
  - backing graph is always called 'graph', other graphs can be referred to by name 
  quickly define arbitrary styles that can be automated with python
---------------------------------------------------------------------------------------- 
syntax:
----------------------------------------------------------------------------------------
[\tex-command(;|%)]
[python-code]
[[A-Z].* TZSL-command]
[{} substitution]
[IF condition (...) [ELSE condition|Optional (...)]|Optional] 
[DEF name[(arg1, arg2, arg3...)|Optional] -> ...]
[EDGE-STYLE (style[]|draw[]|Optional)]
[NODE-STYLE (style[]|draw[]|Optional)]

[GRAPH [name|Optional,] EdgeSet|Graph|SolidGridGraph|Optional
    [EDGES ((edge.s[0],edge.s[1]),(edge.t[0],edge.t[1])[,draw[k=v|k...]|Optional][,style[k=v|k...]|Optional])[python-code|Optional]...[draw[...]|Optional][,style[...]|Optional]|Optional]
    ...
    [NODES (node[0],node[1][,draw[...]|Optional][,style[...]|Optional])[python-code|Optional]...[draw[...]|Optional][,style[...]|Optional]|Optional]|Optional]
    ...
    [SET [PROPERTY]]
    [FOREGROUND tex-command...[python-code|Optional]|Optional]
    [BACKGROUND tex-command...[python-code|Optional]|Optional]
]
----------------------------------------------------------------------------------------
example:
----------------------------------------------------------------------------------------
EDGE-STYLE draw[black, line width=1.75pt]
NODE-STYLE style[circle, draw, black, solid, fill=white, scale=0.5]

DEF SHADE-TWO-FACTOR -> \fill['fill=black!10'] ({x-0.5}cm-1pt, {y-0.5}cm-1pt) rectangle ++(1cm+2pt, 1cm+2pt); for (x, y) in [midpoint_at_face(face) for face in graph.faces if graph.two_factor.test_interior(midpoint_at_face(face))]
DEF MAKE-BIPARTITE -> NODES (x, y, style[color=white]) for (x, y) in graph.nodes if graph.nodes[(x, y)].color == 1

GRAPH alternating_strip_before_flip, SolidGridGraph
    EDGES ((1, 0), (2, 0)), ((1, 1), (2, 1)), ((1, 2), (1, 1)), ((2, 2), (2, 1)), ((1, 4), (1, 3)), ((1, 3), (2, 3)), ((2, 4), (2, 3)), ((1, 6), (1, 5)), ((1, 5), (2, 5)), ((2, 6), (2, 5)), ((1, 7), (2, 7)), ((0, 0), (1, 0)), ((2, 0), (3, 0)), ((3, 0), (3, 1)), ((3, 1), (3, 2)), ((3, 2), (2, 2)), ((1, 2), (0, 2)), ((0, 2), (0, 1)), ((0, 1), (0, 0)), ((1, 4), (0, 4)), ((0, 4), (0, 5)), ((0, 5), (0, 6)), ((0, 6), (1, 6)), ((2, 6), (3, 6)), ((3, 6), (3, 5)), ((3, 5), (3, 4)), ((3, 4), (2, 4)), ((1, 7), (1, 8)), ((1, 8), (0, 8)), ((0, 8), (0, 9)), ((0, 9), (1, 9)), ((1, 9), (2, 9)), ((2, 9), (3, 9)), ((3, 9), (3, 8)), ((3, 8), (2, 8)), ((2, 8), (2, 7)), ((2, 7), (2, 6)), ((2, 6), (1, 6)), ((1, 6), (1, 7)), ((0, 5), (1, 5)), ((1, 5), (1, 4)), ((1, 4), (2, 4)), ((2, 5), (2, 4)), ((2, 5), (3, 5)), ((1, 8), (2, 8)), ((1, 9), (1, 8)), ((2, 9), (2, 8)), ((1, 3), (1, 2)), ((1, 2), (2, 2)), ((2, 2), (2, 3)), ((0, 1), (1, 1)), ((1, 1), (1, 0)), ((2, 0), (2, 1)), ((2, 1), (3, 1)) style[draw=none]
    SET TWO-FACTOR ((0, 0), (1, 0)), ((1, 0), (2, 0)), ((2, 0), (3, 0)), ((3, 0), (3, 1)), ((3, 1), (3, 2)), ((3, 2), (2, 2)), ((2, 2), (2, 1)), ((2, 1), (1, 1)), ((1, 1), (1, 2)), ((1, 2), (0, 2)), ((0, 2), (0, 1)), ((0, 1), (0, 0)), ((1, 3), (2, 3)), ((2, 3), (2, 4)), ((2, 4), (3, 4)), ((3, 4), (3, 5)), ((3, 5), (3, 6)), ((3, 6), (2, 6)), ((2, 6), (2, 5)), ((2, 5), (1, 5)), ((1, 5), (1, 6)), ((0, 6), (1, 6)), ((0, 6), (0, 5)), ((0, 5), (0, 4)), ((0, 4), (1, 4)), ((1, 4), (1, 3)), ((1, 7), (2, 7)), ((2, 7), (2, 8)), ((2, 8), (3, 8)), ((3, 8), (3, 9)), ((3, 9), (2, 9)), ((2, 9), (1, 9)), ((1, 9), (0, 9)), ((0, 9), (0, 8)), ((0, 8), (1, 8)), ((1, 8), (1, 7)) style[color=gray, dotted]
    SET LONGEST-STRIP max(get_alternating_strips(graph), key=len)
    SET LONGEST-STRIP-FLIPPED 
    EDGES edge for edge in graph.longest_strip
    EDGES edge for edge in graph.two_factor style[color=gray, dotted]
    NODES (x, y, style[draw=none, fill=none]) for (x, y) in graph.nodes if Node(x, y) not in graph.longest_strip
    SHADE-TWO-FACTOR
GRAPH alternating_strip_after_flip, SolidGridGraph
    EDGES edge for edge in alternating_strip_before_flip style[draw=none]
    SET TWO-FACTOR edge_flip(alternating_strip_before_flip, alternating_strip_before_flip.longest_strip).two_factor
    EDGES edge for edge in graph.two_factor style[color=gray, dotted]
    EDGES edge for edge in alternating_strip_before_flip.longest_strip.flipped_perimeter + alternating_strip_before_flip.longest_strip - alternating_strip_before_flip.longest_strip.perimeter
    NODES (x, y, style[draw=none, fill=none]) for (x, y) in graph.nodes if Node(x, y) not in alternating_strip_before_flip.longest_strip_flipped
    SHADE-TWO-FACTOR
----------------------------------------------------------------------------------------
how it will work:
  works essentially as a python preprocessor 
  keywords (all caps, style[], draw[], etc.) are expanded into python
  - python will keep track of variables, so the context is always python's
  -- graph is a special variable that will always refer to the graph defined in the current scope
  -- TZSL functions are expanded lazily, so variable names resolve at the last possible moment
  - DEF defines new macros which are added dynamically at compile-time to the list of keywords to be matched
the ast will be lispy. 
  - nodes have the form (HEAD, ARGS..., DATA...), where DATA is a field that contains other AST nodes. 
  -- ARGS cannot contain keywords, we rely on keywords to delineate ARGS from DATA. 
  - to enable delineation of tex commands, backslashes \ must be escaped (\\ will result in \ instead of being read as a tex command)
  -- since ending tex can be ambiguous, either a percent % or semicolon ; signify the end of a tex command (again, escape with \). 
  - brackets {} are treated as they are in f-strings, with substitutions being made relative to python's scope
  - data lower in the AST has lower scope. 
  -- this means EDGE-STYLE and NODE-STYLE parameters at a higher scope provide default values for style[] draw[] expansions at lower scopes
  -- there will be a special toplevel scope called (HEAD, DATA...), which is globally available at compile-time

the compilation process will be as follows:
  1. pattern match against TZSL keywords via regex (e.g. GRAPH (name)?, (EdgeSet|Graph|SolidGridGraph)?, .*) to build AST
  2. reconcile control flow in current AST (i.e. replace IF statement by true branch) 
  3. lower TZSL keywords to python 
  4. repeat 1-3 until no keywords are left. (infinite recursion is possible)
-}