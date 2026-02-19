# Planar Graphs Engine

The repo contains a set of Python scripts that do various things related to my [senior thesis](thesis.pdf). To run them, you'll need to install the [CPLEX solver from IBM](https://www.ibm.com/products/ilog-cplex-optimization-studio) and `docplex` [for Python 3.10.11](https://ibmdecisionoptimization.github.io/docplex-doc/mp/getting_started_python.html). You will also need `matplotlib` and `sqlite3`, but these and any other dependencies can be installed as usual with the package manager. 

Running `python gen.py` or `python temp.py` will generate TikZ figures as `.tex` files in the `../draft/examples` directory, which you can import in a LaTeX document (e.g. `\input{example.tex}`). This is a good starting point for interacting with the code. 

`to_tex.py` contains the TikZ "compiler", and is responsible for the logic of rendering graphs into TikZ. 
`graph.py` implements the algorithms used in the thesis, and its code is closely related to the mathematical formalisms. 

This code was not written with other users in mind. It exists essentially as supplementary to the thesis document. I might someday refactor it to be more readable.

