# py2flows

A control flow generator for Python that is able to generate control flow graphs and corresponding flows. The motivation
behind this project is to generate flows suitable for data flow analysis for Python.
## PLEASE NOTE

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
- [] ast.AsyncFunctionDef(Poor support for now)
- [x] ast.ClassDef
- [x] ast.Return
- [x] ast.Delete
- [x] ast.Assign
- [x] ast.AugAssign
- [x] ast.AnnAssign
- [x] ast.For
- [] ast.AsyncFor(Poor support for now)
- [x] ast.While
- [x] ast.If
- [x] ast.With(Poor support for now)
- [] ast.AsyncWith(Poor support for now)
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
- [] ast.Await
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
