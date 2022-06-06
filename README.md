# py2flows

A control flow generator for Python that is able to generate control flow graphs and corresponding flows. The motivation
behind this project is to generate flows suitable for data flow analysis for Python. In my plan, an instance of
a [dynamic monotone framework](https://github.com/LayneInNL/dmf) is going to be implemented upon it.

## PLEASE NOTE

> I'm doing my master's thesis now. In order to save some time, now the latest py2flows code lies in [dmf](https://github.com/LayneInNL/dmf). You can find it in module dmf.flows.

> After I finish my thesis, I will update this project.

## Supported language versions

### Python version

- [x] Python 3.7
- [] Python 3.8(Untested)
- [] Python 3.9(Untested)
- [] Python 3.10(Untested)

## Abstract Syntax Tree

### Modules

- [x] ast.Module

### Statements

- [x] ast.FunctionDef
- [x] ast.AsyncFunctionDef(Poor support for now)
- [x] ast.ClassDef
- [x] ast.Return
- [x] ast.Delete
- [x] ast.Assign
- [x] ast.AugAssign
- [x] ast.AnnAssign
- [x] ast.For
- [x] ast.AsyncFor(Poor support for now)
- [x] ast.While
- [x] ast.If
- [x] ast.With(Poor support for now)
- [x] ast.AsyncWith(Poor support for now)
- [x] ast.Raise(Poor support for now)
- [x] ast.Try(Relatively poor support for now)
- [x] ast.Assert
- [x] ast.Import
- [x] ast.ImportFrom
- [x] ast.Global
- [x] ast.Nonlocal
- [x] ast.Expr
- [x] ast.Pass
- [x] ast.Break
- [x] ast.Continue

### Expressions

- [x] ast.BoolOp
- [x] ast.BinOp
- [x] ast.UnaryOp
- [x] ast.Lambda
- [x] ast.IfExp
- [x] ast.Dict
- [x] ast.Set
- [x] ast.ListComp
- [x] ast.SetComp
- [x] ast.DictComp
- [x] ast.GeneratorExp
- [x] ast.Await
- [x] ast.Yield
- [x] ast.YieldFrom
- [x] ast.Compare
- [x] ast.Call
- [x] ast.Num
- [x] ast.Str
- [x] ast.FormattedValue
- [x] ast.JoinedStr
- [x] ast.Bytes
- [x] ast.NameConstant
- [x] ast.Ellipsis
- [x] ast.Constant
- [x] ast.Attribute
- [x] ast.Subscript
- [x] ast.Starred
- [x] ast.Name
- [x] ast.List
- [x] ast.Tuple

Support for other statements and expressions will be added gradually.

### Additional features

- [x] Removal of comments and docstrings
- [x] Decomposition of complex statements
- [x] Isolated entries and exits
- [] Support for Modules and packages. Motivation: a project may contain multiple packages and modules. So we would like
  to maintain a data structure to represent this relationship. It would be helpful for static analysis.
- [x] Refactor exception handling
- [] Count characteristics of each file. Motivation: Emm. It is related to my thesis topic. I wanna count the tendency
  of each source file. I bet this feature would be implemented in 2 weeks.
- [x] Prune temporal variables
- [x] Use two labels to denote call and return of each function call
- [x] Understandable CFGs generated
- [x] Use one return to denote all returns in functions

### What can be better

- [] The data structure representing classes. Motivation: As we know, classes consist of fields and methods. So it's
  better to have a good data structure to store and retrieve fields and methods elegantly. I think it also holds for
  functions(In modern languages, functions can be shadowed).

## How to use it

### Install

1. Install all dependencies listed in *requirements.txt*
2. Open a terminal and run `python setup.py install`
3. If step 2 succeeds, an executable file *py2flows* will be available.
4. `py2flows --help`

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
