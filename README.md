# py2flows

A control flow generator for Python that is able to generate control flow graphs and flows.

## Python version

- Tested on Python 3.7

## Supported language features

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
- [x] ast.Try(partially)
- [x] ast.Break
- [x] ast.Continue
- [x] ast.Pass
- [x] ast.Return
- [x] ast.Expr
- [x] ast.Call
- [x] ast.Import
- [x] ast.ImportFrom
- [x] ast.Module
- [x] Remove comments and docstrings
- [x] ast.Yield

## How to use it

1. Install all dependencies in requirements.
2. Open a terminal and run `python cfg.py file-name`