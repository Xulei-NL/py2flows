import sys
import ast
from typing import Any
import astpretty
import astor


class Transformer(ast.NodeTransformer):
    def visit_Assign(self, node: ast.Assign) -> Any:
        if type(node.value) == ast.ListComp:
            self.visit(node.value)

    def visit_ListComp(self, node: ast.ListComp) -> Any:
        new_expr = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="a", ctx=ast.Load()),
                    attr="append",
                    ctx=ast.Load(),
                ),
                args=[node.elt],
                keywords=[],
            )
        )

        last_generator: ast.comprehension = node.generators[-1]
        exprs = last_generator.ifs
        if len(exprs) == 0:
            body = new_expr
        else:
            iff = ast.BoolOp(ast.And(), exprs)
            body = ast.If(test=iff, body=[new_expr], orelse=[])

        inner_for: ast.For = ast.For(
            target=last_generator.target,
            iter=last_generator.iter,
            body=[body],
            orelse=[],
        )
        current_for = inner_for
        rest_generators = node.generators[:-1]
        for generator in reversed(rest_generators):
            current_for = ast.For(
                target=generator.target,
                iter=generator.iter,
                body=[inner_for],
                orelse=[],
            )
            inner_for = current_for

        # self.generic_visit(inner_for)
        return current_for


if __name__ == "__main__":
    filename = sys.argv[1]
    file = open(filename, "r")
    source = file.read()
    node = ast.parse(source)
    new_node = ast.fix_missing_locations(Transformer().visit(node))
    astpretty.pprint(new_node)
    print(astor.to_source(new_node))
