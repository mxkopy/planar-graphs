from graph import SolidGridGraph, EdgeSet
import docplex
from docplex.mp.model import Model
from docplex.mp.utils import DOcplexException
from math import log

model = Model(name='model')

# x = model.continuous_var(name='corn')
# y = model.continuous_var(name='wheat')
# r = lambda x, y: 10*x + 1.5*y
# f = lambda x, y: 1.5*x + 0.6*y
# p = lambda x, y: 0.2*x + 0.05*y

# model.add_constraint(f(x, y) <= 2500)
# model.add_constraint(p(x, y) <= 200)
# model.maximize(r(x, y))
# model.solve()

# print(model.solution)
# print('corn', x.solution_value)
# print('wheat', y.solution_value)


def extract_graph(model, graph):
    if model.solution is not None:
        return EdgeSet(edges=[edge for edge in graph if model.solution[str(edge)] == 1.0])
    return EdgeSet(edges=[])


# L, R = make_bipartite()
# sum(x_{ij}) for i in L <= 1
# sum(x_{ij}) for j in R <= 1

def find_bipartite_matching(graph):
    model = Model(name='bipartite_matching')
    xs = {}
    xvs = { node: [] for node in graph.nodes }
    L, R, _ = graph.make_bipartite()

    for edge in graph:
        id = str(edge)
        u, v = edge
        xs[edge] = model.integer_var(name=id)
        xvs[u].append()
    

def find_two_factor(graph):
    model = Model(name='two_factor')
    xs = {}
    xvs = {node: [] for node in graph.nodes}
    for edge in graph:
        id = str(edge)
        u, v = edge
        xs[edge] = model.integer_var(name=id)
        xvs[u].append(xs[edge])
        xvs[v].append(xs[edge])
    for uv in xs:
        model.add_constraint(xs[uv] >= 0)
        model.add_constraint(xs[uv] <= 1)
    for u in xvs:
        uvs = xvs[u]
        model.add_constraint(sum(uvs) == 2)
    model.minimize(sum(xs.values()))
    model.solve()
    return extract_graph(model, graph)

def display_blue_solid(graph, edgeset):
    graph.display_edges(list(edgeset), color='blue', linewidth=2.0, alpha=1.0, zorder=0)

def display_red_dotted(graph, edgeset):
    graph.display_edges(list(edgeset), color='red', linewidth=2.0, linestyle=(0, (5, 5)), alpha=1.0)

def make_example(fname, n=10):
    graph = SolidGridGraph()
    for _ in range(n):
        graph.add_random_face(n, n)
    two_factor = find_two_factor(graph)


    # graph.init_plt()
    # graph.display_nodes(graph.nodes, color='black')

    # if len(two_factor) != 0:
    #     display_red_dotted(graph, two_factor)
    #     graph.plt.savefig(f'{fname}_TF.png')

    # graph.plt.edges = []
    # for edge in graph.plt.edges:
    #     edge.remove()
    # graph.display_edges(graph.keys(), color='black')
    # graph.plt.savefig(f'{fname}.png')



# graph = SolidGridGraph()

# graph.add_random_face(10, 10)
# graph.add_random_face(10, 10)
# graph.add_random_face(10, 10)
# graph.add_random_face(10, 10)
# graph.add_random_face(10, 10)
# graph.add_random_face(10, 10)
# graph.add_random_face(10, 10)
# graph.add_random_face(10, 10)



# print(graph.hamiltonian_cycles())

# for i in range(5):
#     make_example(f'examples/{i}')

# two_factor = find_two_factor(graph)

# graph.init_plt()
# graph.display_nodes(graph.nodes, color='black')
# display_red_dotted(graph, two_factor)
# graph.show()


