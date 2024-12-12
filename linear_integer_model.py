from graph import SolidGridGraph
from docplex.mp.model import Model
from docplex.mp.utils import DOcplexException

def run_example(n=4):

    model = Model(name='model')

    sgg = SolidGridGraph(n)

    xs = {}
    xvs = {node: [] for node in SolidGridGraph.nodes_from_weights(sgg.weights)}

    for edge in sgg.weights:
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
    return model, sgg

def try_example(n=4, failing=False):
    cont = True
    while cont:
        try:
            model, sgg = run_example(n)
            model.print_information()
            model.report()
            model.print_solution()
            cont = False           
        except DOcplexException:
            if failing:
                cont = False
    return model, sgg


def display_example(model, sgg):
    sgg.init_display()
    if model.solution is not None:        
        solution = [edge for edge in sgg.weights if model.solution[str(edge)] == 1.0]
        sgg.display_edges(solution, color='blue', linewidth=2.0, alpha=1.0, zorder=0)

        HC = SolidGridGraph.hamiltonian_cycles(sgg.weights)

        if len(HC) > 0:
            print(HC[0])
            hc_edges = [HC[0][i:i+2] for i in range(len(HC[0])-1)]
            sgg.display_edges(hc_edges, color='red', linewidth=2.0, linestyle=(0, (5, 5)), alpha=1.0)

    sgg.display_edges(sgg.weights.keys(), color='black', alpha=0.1)
    sgg.display_nodes(SolidGridGraph.nodes_from_weights(sgg.weights), color='purple')
    sgg.plt.gca().set_aspect('equal')
    sgg.plt.show()

display_example(*try_example(n=10, failing=False))