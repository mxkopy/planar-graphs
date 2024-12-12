import graph
from docplex.mp.model import Model

def e(i,j):
    return str(i)+","+str(j)

model = Model(name='2x4')

#  0--2--4--6
#  |  |  |  |
#  1--3--5--7

# BUILD THE EDGE INDICATOR VARIABLES
#
xs = {}
xvs = {}
# vertices
for u in range(8):
    xvs[u] = []

# verticals
for i in range(4):
    u = 2*i
    v = 2*i+1
    id = e(u,v)
    xs[id] = model.integer_var(name='edge '+id)
    xvs[u].append(xs[id])
    xvs[v].append(xs[id])

# top horizontals
for i in range(3):
    u = 2*i
    v = 2*(i+1)
    id = e(u,v)
    xs[id] = model.integer_var(name='edge '+id)
    xvs[u].append(xs[id])
    xvs[v].append(xs[id])
    
# bottom horizontals
for i in range(3):
    u = 2*i+1
    v = 2*(i+1)+1
    id = e(u,v)
    xs[id] = model.integer_var(name='edge '+id)
    print(xs[id])
    xvs[u].append(xs[id])
    xvs[v].append(xs[id])

print(xs)
print(xvs)

# ADD THE 0,1 CONSTRAINTS
#
for uv in xs:
    model.add_constraint(xs[uv] >= 0)
    model.add_constraint(xs[uv] <= 1)

# WONKILY ADD THE DEGREE 2 CONSTRAINTS
print("degree constraints")
for u in xvs:
    uvs = xvs[u]
    # print("vars for",u,"are",uvs)
    if len(uvs) == 2:
        print(model.add_constraint(uvs[0] + uvs[1] == 2))
    if len(uvs) == 3:
        print(model.add_constraint(uvs[0] + uvs[1] + uvs[2] == 2))

# WONKILY ADD THE OBJECTIVE FUNCTION
model.maximize(        xs[e(0,2)] + xs[e(2,4)] + xs[e(4,6)]
               + xs[e(0,1)] + xs[e(2,3)] + xs[e(4,5)] +  xs[e(6,7)]
                     + xs[e(1,3)] + xs[e(3,5)] + xs[e(5,7)]         )
        
# SOLVE AND INSPECT THE RESULT
model.print_information()
soln = model.solve()
model.report()
model.print_solution()

