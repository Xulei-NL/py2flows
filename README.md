# py2flows

A control flow generator for Python that is able to generate control flow graphs and flows. The motivation behind this
project is to generate flows suitable for data flow analysis for Python.

## Supported language features

### Python version

- [x] Python 3.7
- [] Python 3.8(Untested)
- [] Python 3.9(Untested)
- [] Python 3.10(Untested)

### Abstract Syntax Tree

- [x] ast.Assign
- [x] ast.If
- [x] ast.While
- [x] ast.For
- [x] ast.IfExp
- [x] ast.ListComp
- [x] ast.SetComp
- [x] ast.DictComp
- [x] ast.GeneratorExp
- [x] ast.Lambda
- [x] ast.Try
- [x] ast.Break
- [x] ast.Continue
- [x] ast.Pass
- [x] ast.Return
- [x] ast.Expr
- [x] ast.Call
- [x] ast.Import
- [x] ast.ImportFrom
- [x] ast.Module
- [x] ast.Yield

Support for other statements and expressions will be added gradually.

### Additional features

- [x] Removal of comments and docstrings
- [x] Decomposition of complex statements
- []  Modules and packages
- [x] Isolated entries and exits

## How to use it

### Install

1. Install all dependencies listed in requirements.
2. Open a terminal and run `python setup.py install`
3. If step 2 succeeds, an executable file *py2flows* will be available.
4. `py2flows filename.py`

### Example 1

```python
# 12_listcomp.py
z = [[1, 2, 3], [4, 5, 6]]
a = [x for x in [y for y in z]]
b = [2 * x for y in z if len(y) > 1 for x in y if x > 2 if x < 4]
```

![Example](images/12_listcomp.svg)

### Example 2

```python
def test():
    while i <= n:
        sum = sum + i
        i = i + 1


print("no isolated entries and exits")
```

> No isolated entries and exits

![No isolated entries and exits](images/noiso.svg)

> Isolated entries and exits

![Isolated entries and exits](images/iso.svg)