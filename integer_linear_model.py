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
    if model.solution is not None:
        return EdgeSet(*[edge for edge in graph if model.solution(str(edge)) == 1.0])
    return EdgeSet()
