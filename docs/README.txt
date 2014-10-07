#### Setup

Create docs directory:

$ mkdir docs

Initialize sphinx (say yes to the “autodoc” extension):

$ cd docs
$ sphinx-quickstart

Edit config.py and specify sys.path to top level:

sys.path.insert(0, os.path.abspath('..'))

#### Update

Automatically discover modules:

$ cd docs
$ sphinx-apidoc -o . ..

#### Build

$ cd docs
$ make html

#### View

$ cd docs
$ open _build/html/index.html


using bootstrap theme:

$ pip install sphinx_bootstrap_theme