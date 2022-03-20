from __future__ import annotations
from . import randoms
import astor
import graphviz as gv
import os
import ast
import logging
from typing import Dict, List, Tuple, Set, Optional, Type, Any

BASIC_TYPES = (ast.Num, ast.Str, ast.FormattedValue, ast.JoinedStr,
               ast.Bytes, ast.NameConstant, ast.Ellipsis, ast.Constant, ast.Name)


class BlockId:
    counter: int = 0

    @classmethod
    def gen_block_id(cls) -> int:
        cls.counter += 1
        return cls.counter


class BasicFlow:
    def __init__(self, first_label: int, second_label: int):
        self.first_label: int = first_label
        self.second_label: int = second_label


class FuncFlow:
    def __init__(
            self, call_label: int, entry_label: int, return_label: int, exit_label: int
    ):
        self.call_label: int = call_label
        self.entry_label: int = entry_label
        self.return_label: int = return_label
        self.exit_label: int = exit_label


class BasicBlock(object):
    def __init__(self, bid: int):
        self.bid: int = bid
        self.stmt = []
        self.calls: List[str] = []
        self.prev: List[int] = []
        self.next: List[int] = []

    def is_empty(self) -> bool:
        return len(self.stmt) == 0

    def has_next(self) -> bool:
        return len(self.next) != 0

    def has_previous(self) -> bool:
        return len(self.prev) != 0

    def remove_from_prev(self, prev_bid: int) -> None:
        if prev_bid in self.prev:
            self.prev.remove(prev_bid)

    def remove_from_next(self, next_bid: int) -> None:
        if next_bid in self.next:
            self.next.remove(next_bid)

    def stmt_to_code(self) -> str:
        code = str(self.bid) + "\n"
        if self.stmt and type(self.stmt[0]) == ast.Module:
            code += "Module"
            return code
        for stmt in self.stmt:
            line = astor.to_source(stmt)
            code += (
                line.split("\n")[0] + "\n"
                if type(stmt) in [
                    ast.If,
                    ast.For,
                    ast.Try,
                    ast.While,
                    ast.With,
                    ast.AsyncWith,
                    ast.AsyncFor,
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                    ast.excepthandler,
                ]
                else line
            )
        return code

    def calls_to_code(self) -> str:
        return "\n".join(self.calls)

    def __str__(self):
        return "Block ID: {}".format(self.bid)


class FuncBlock(BasicBlock):
    def __init__(self, bid: int):
        super().__init__(bid)
        self.name: Optional[str] = None
        self.paras: List[str] = []


class CallBlock(BasicBlock):
    def __init__(self, bid: int):
        super().__init__(bid)
        self.args: List[str] = []
        self.call = bid
        self.exit = BlockId.gen_block_id()

    def __str__(self):
        return "Block ID: {}".format(self.bid)


class TryBlock(BasicBlock):
    def __init__(self, bid: int):
        super().__init__(bid)


def add_stmt(block: BasicBlock, stmt) -> None:
    block.stmt.append(stmt)


class CFG:
    def __init__(self, name: str):
        self.name: str = name

        self.start: Optional[BasicBlock] = None
        self.final_blocks: List[BasicBlock] = []
        # Function name to (Args, CFG)
        self.func_cfgs: Dict[str, (List[str, ast.AST], CFG)] = {}
        self.async_func_cfgs: Dict[str, (List[str, ast.AST], CFG)] = {}
        self.class_cfgs: Dict[str, CFG] = {}
        self.blocks: Dict[int, BasicBlock] = {}
        self.edges: Dict[Tuple[int, int], Optional[ast.AST]] = {}
        self.graph: Optional[gv.dot.Digraph] = None

        self.flows: Set[Tuple[int, int]] = set()

    def _traverse(
            self, block: BasicBlock, visited: Set[int] = set(), calls: bool = True
    ) -> None:
        if block.bid not in visited:
            visited.add(block.bid)
            self.graph.node(str(block.bid), label=block.stmt_to_code())
            if calls and block.calls:
                self.graph.node(
                    str(block.bid) + "_call",
                    label=block.calls_to_code(),
                    _attributes={"shape": "box"},
                )
                self.graph.edge(
                    str(block.bid),
                    str(block.bid) + "_call",
                    label="calls",
                    _attributes={"style": "dashed"},
                )

            for next_bid in block.next:
                self._traverse(self.blocks[next_bid], visited, calls=calls)
                self.graph.edge(
                    str(block.bid),
                    str(next_bid),
                    label=astor.to_source(self.edges[(block.bid, next_bid)])
                    if self.edges[(block.bid, next_bid)]
                    else "",
                )

    def _show(self, fmt: str = "svg", calls: bool = True) -> gv.dot.Digraph:
        # self.graph = gv.Digraph(name='cluster_'+self.name, format=fmt, graph_attr={'label': self.name})
        self.graph = gv.Digraph(name="cluster_" + self.name, format=fmt)
        self._traverse(self.start, calls=calls)
        for k, v in self.func_cfgs.items():
            self.graph.subgraph(v[1]._show(fmt, calls))
        for k, v in self.async_func_cfgs.items():
            self.graph.subgraph(v[1]._show(fmt, calls))
        for class_name, classCFG in self.class_cfgs.items():
            self.graph.subgraph(classCFG._show(fmt, calls))
        return self.graph

    def show(
            self,
            filepath: str = "../output",
            fmt: str = "svg",
            calls: bool = True,
            show: bool = True,
    ) -> None:
        self._show(fmt, calls)
        path = os.path.normpath(filepath)
        self.graph.render(path, view=show, cleanup=True)


class CFGVisitor(ast.NodeVisitor):

    def __init__(self, isolation: int):
        super().__init__()
        self.ifExp = False
        self.cfg: Optional[CFG] = None
        self.curr_block: Optional[BasicBlock] = None

        self.isolation = isolation
        self.after_loop_stack: List[BasicBlock] = []
        self.loop_guard_stack: List[BasicBlock] = []

    def build(self, name: str, tree: ast.Module) -> CFG:
        self.cfg = CFG(name)
        self.curr_block = self.new_block()
        self.cfg.start = self.curr_block

        self.visit(tree)
        # self.remove_empty_blocks(self.cfg.start)
        logging.debug('Start id: %d', self.cfg.start.bid)
        logging.debug('End id: %d', self.curr_block.bid)
        return self.cfg

    def new_block(self) -> BasicBlock:
        bid: int = BlockId.gen_block_id()
        self.cfg.blocks[bid] = BasicBlock(bid)
        return self.cfg.blocks[bid]

    def new_func_block(self, bid: int) -> FuncBlock:
        self.cfg.blocks[bid] = FuncBlock(bid)
        return self.cfg.blocks[bid]

    def new_call_block(self, bid: int) -> CallBlock:
        self.cfg.blocks[bid] = CallBlock(bid)
        return self.cfg.blocks[bid]

    def add_edge(self, frm_id: int, to_id: int, condition=None) -> BasicBlock:
        self.cfg.blocks[frm_id].next.append(to_id)
        self.cfg.blocks[to_id].prev.append(frm_id)
        self.cfg.edges[(frm_id, to_id)] = condition
        self.cfg.flows.add((frm_id, to_id))
        return self.cfg.blocks[to_id]

    def add_loop_block(self) -> BasicBlock:
        if self.curr_block.is_empty() and not self.curr_block.has_next():
            return self.curr_block
        else:
            loop_block = self.new_block()
            self.add_edge(self.curr_block.bid, loop_block.bid)
            return loop_block

    def add_FuncCFG(self, tree: ast.FunctionDef) -> None:
        arg_list: List[(str, Optional[ast.AST])] = []

        tmp_arg_list: List[str] = []
        for arg in tree.args.args:
            tmp_arg_list.append(arg.arg)
        len_arg_list = len(tmp_arg_list)
        len_defaults = len(tree.args.defaults)
        diff = len_arg_list - len_defaults
        index = 0
        while diff > 0:
            arg_list.append((tmp_arg_list[index], None))
            diff -= 1
            index += 1

        for default in tree.args.defaults:
            arg_list.append((tmp_arg_list[index], default))
            index += 1

        visitor: CFGVisitor = CFGVisitor(self.isolation)
        func_cfg: CFG = visitor.build(tree.name, ast.Module(body=tree.body))
        if self.isolation:
            if not func_cfg.final_blocks:
                visitor.add_stmt(visitor.curr_block, ast.Pass())
                func_cfg.final_blocks.append(visitor.curr_block)
        visitor.remove_empty_blocks(func_cfg.start)
        logging.debug([elt.__str__() for elt in func_cfg.final_blocks])
        self.cfg.func_cfgs[tree.name] = (arg_list, func_cfg)

    def add_AsyncFuncCFG(self, tree: ast.FunctionDef) -> None:
        arg_list: List[(str, Optional[ast.AST])] = []

        tmp_arg_list: List[str] = []
        for arg in tree.args.args:
            tmp_arg_list.append(arg.arg)
        len_arg_list = len(tmp_arg_list)
        len_defaults = len(tree.args.defaults)
        diff = len_arg_list - len_defaults
        index = 0
        while diff > 0:
            arg_list.append((tmp_arg_list[index], None))
            diff -= 1
            index += 1

        for default in tree.args.defaults:
            arg_list.append((tmp_arg_list[index], default))
            index += 1

        visitor: CFGVisitor = CFGVisitor(self.isolation)
        func_cfg: CFG = visitor.build(tree.name, ast.Module(body=tree.body))
        if self.isolation:
            if not func_cfg.final_blocks:
                visitor.add_stmt(visitor.curr_block, ast.Pass())
                func_cfg.final_blocks.append(visitor.curr_block)
        visitor.remove_empty_blocks(func_cfg.start)
        logging.debug([elt.__str__() for elt in func_cfg.final_blocks])
        self.cfg.async_func_cfgs[tree.name] = (arg_list, func_cfg)

    def add_ClassCFG(self, node: ast.ClassDef):
        class_body: ast.Module = ast.Module(body=node.body)
        visitor: CFGVisitor = CFGVisitor(self.isolation)
        class_cfg: CFG = visitor.build(node.name, class_body)
        self.cfg.class_cfgs[node.name] = class_cfg

    def remove_empty_blocks(self, block: BasicBlock, visited: Set[int] = set()) -> None:
        if block.bid not in visited:
            visited.add(block.bid)
            if block.is_empty():
                for prev_bid in list(block.prev):
                    prev_block = self.cfg.blocks[prev_bid]
                    for next_bid in list(block.next):
                        next_block = self.cfg.blocks[next_bid]
                        self.add_edge(prev_bid, next_bid)
                        # self.cfg.flows.add((prev_bid, next_bid))
                        self.cfg.edges.pop((block.bid, next_bid), None)
                        next_block.remove_from_prev(block.bid)
                        # if (block.bid, next_block.bid) in self.cfg.flows:
                        #     self.cfg.flows.remove((block.bid, next_block.bid))
                    self.cfg.edges.pop((prev_bid, block.bid), None)
                    prev_block.remove_from_next(block.bid)
                    # if (prev_block.bid, block.bid) in self.cfg.flows:
                    #     self.cfg.flows.remove((prev_block.bid, block.bid))

                block.prev.clear()
                for next_bid in list(block.next):
                    self.remove_empty_blocks(self.cfg.blocks[next_bid], visited)
                block.next.clear()

            else:
                for next_bid in list(block.next):
                    self.remove_empty_blocks(self.cfg.blocks[next_bid], visited)

    def combine_conditions(self, node_list: List[ast.expr]) -> ast.expr:
        return (
            node_list[0]
            if len(node_list) == 1
            else ast.BoolOp(op=ast.And(), values=node_list)
        )

    def populate_body(self, body_list):
        for child in body_list:
            self.visit(child)

    def populate_body_to_next_bid(self, body_list, to_bid: int) -> None:
        for child in body_list:
            self.visit(child)
        if not self.curr_block.next:
            self.add_edge(self.curr_block.bid, to_bid)

    def visit_Module(self, node: ast.Module) -> None:
        if self.isolation:
            add_stmt(self.curr_block, ast.Pass())
            self.curr_block = self.add_edge(self.curr_block.bid, self.new_block().bid)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        add_stmt(self.curr_block, node)
        self.add_FuncCFG(node)
        self.curr_block = self.add_edge(self.curr_block.bid, self.new_block().bid)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        add_stmt(self.curr_block, node)
        self.add_AsyncFuncCFG(node)
        self.curr_block = self.add_edge(self.curr_block.bid, self.new_block().bid)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        add_stmt(self.curr_block, node)
        self.add_ClassCFG(node)
        self.curr_block = self.add_edge(self.curr_block.bid, self.new_block().bid)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is None:
            add_stmt(self.curr_block, node)
            self.cfg.final_blocks.append(node)
            self.curr_block = self.add_edge(self.curr_block.bid, self.new_block().bid)
        else:
            new_expr_sequence = self.visit(node.value)
            if len(new_expr_sequence) == 1:
                add_stmt(self.curr_block, node)
                self.cfg.final_blocks.append(node)
                self.curr_block = self.add_edge(self.curr_block.bid, self.new_block().bid)
            else:
                return_stmt = ast.Return(value=new_expr_sequence[-1])
                generated_return_sequence = new_expr_sequence[:-1] + [return_stmt]
                self.populate_body(generated_return_sequence)

    def visit_Delete(self, node: ast.Delete) -> None:
        add_stmt(self.curr_block, node)
        self.curr_block = self.add_edge(self.curr_block.bid, self.new_block().bid)

    def visit_Assign(self, node: ast.Assign) -> None:
        # logging.debug('Current assignment: ', astor.to_source(node))
        new_expr_sequence = self.visit(node.value)

        if len(new_expr_sequence) == 1:
            add_stmt(self.curr_block, node)
            self.curr_block = self.add_edge(self.curr_block.bid, self.new_block().bid)
            return

        new_assign = ast.Assign(
            targets=node.targets,
            value=new_expr_sequence[-1]
        )
        new_sequence = new_expr_sequence[:-1] + [new_assign]
        source = ''
        for expr in new_sequence:
            source += astor.to_source(expr)
        logging.debug('Decomposed expr: ' + source)
        self.populate_body(new_sequence)
        return

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        add_stmt(self.curr_block, node)
        self.curr_block = self.add_edge(self.curr_block.bid, self.new_block().bid)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value:
            new_assign: ast.Assign = ast.Assign(
                targets=[node.target],
                value=node.value
            )
            self.visit(new_assign)
        else:
            logging.debug('AnnAssign without value. Just ignore it')

    def visit_For(self, node: ast.For) -> None:
        iter_sequence = self.visit(node.iter)
        node.iter = iter_sequence[-1]
        self.populate_body(iter_sequence[:-1])

        loop_guard = self.add_loop_block()
        self.curr_block = loop_guard
        add_stmt(self.curr_block, node)
        self.loop_guard_stack.append(loop_guard)

        # New block for the body of the for-loop.
        for_block: BasicBlock = self.new_block()
        self.add_edge(self.curr_block.bid, for_block.bid)
        after_for_block: BasicBlock = self.new_block()
        self.add_edge(self.curr_block.bid, after_for_block.bid)
        self.after_loop_stack.append(after_for_block)
        if not node.orelse:
            # Block of code after the for loop.
            # self.add_edge(self.curr_block.bid, after_for_block.bid)

            # self.loop_stack.append(after_for_block)
            self.curr_block = for_block
            self.populate_body_to_next_bid(node.body, loop_guard.bid)
        else:
            # Block of code after the for loop.
            or_else_block: BasicBlock = self.new_block()
            self.add_edge(self.curr_block.bid, or_else_block.bid)

            # self.loop_stack.append(after_for_block)
            self.curr_block = for_block
            self.populate_body_to_next_bid(node.body, loop_guard.bid)

            self.curr_block = or_else_block
            self.populate_body_to_next_bid(node.orelse, after_for_block.bid)

        # Continue building the CFG in the after-for block.
        self.curr_block = after_for_block
        self.after_loop_stack.pop()
        self.loop_guard_stack.pop()

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        iter_sequence = self.visit(node.iter)
        node.iter = iter_sequence[-1]
        self.populate_body(iter_sequence[:-1])

        loop_guard = self.add_loop_block()
        self.curr_block = loop_guard
        add_stmt(self.curr_block, node)
        self.loop_guard_stack.append(loop_guard)

        # New block for the body of the for-loop.
        for_block: BasicBlock = self.new_block()
        self.add_edge(self.curr_block.bid, for_block.bid)
        after_for_block: BasicBlock = self.new_block()
        self.add_edge(self.curr_block.bid, after_for_block.bid)
        self.after_loop_stack.append(after_for_block)
        if not node.orelse:
            # Block of code after the for loop.
            # self.add_edge(self.curr_block.bid, after_for_block.bid)

            # self.loop_stack.append(after_for_block)
            self.curr_block = for_block
            self.populate_body_to_next_bid(node.body, loop_guard.bid)
        else:
            # Block of code after the for loop.
            or_else_block: BasicBlock = self.new_block()
            self.add_edge(self.curr_block.bid, or_else_block.bid)

            # self.loop_stack.append(after_for_block)
            self.curr_block = for_block
            self.populate_body_to_next_bid(node.body, loop_guard.bid)

            self.curr_block = or_else_block
            self.populate_body_to_next_bid(node.orelse, after_for_block.bid)

        # Continue building the CFG in the after-for block.
        self.curr_block = after_for_block
        self.after_loop_stack.pop()
        self.loop_guard_stack.pop()

    def visit_While(self, node: ast.While) -> None:

        test_sequence = self.visit(node.test)
        node.test = test_sequence[-1]
        self.populate_body(test_sequence[:-1])

        loop_guard: BasicBlock = self.add_loop_block()
        self.curr_block = loop_guard
        add_stmt(loop_guard, node)
        self.loop_guard_stack.append(loop_guard)

        # New block for the case where the test in the while is False.
        after_while_block: BasicBlock = self.new_block()
        self.add_edge(self.curr_block.bid, after_while_block.bid)
        self.after_loop_stack.append(after_while_block)

        if not node.orelse:
            # New block for the case where the test in the while is True.
            # Populate the while block.
            while_body_block: BasicBlock = self.new_block()
            self.add_edge(self.curr_block.bid, while_body_block.bid)
            self.curr_block = while_body_block

            self.populate_body_to_next_bid(node.body, loop_guard.bid)
        else:
            or_else_block: BasicBlock = self.new_block()
            self.add_edge(self.curr_block.bid, or_else_block.bid)
            self.curr_block = self.add_edge(self.curr_block.bid, self.new_block().bid)
            self.populate_body_to_next_bid(node.body, loop_guard.bid)

            self.curr_block = or_else_block
            self.populate_body_to_next_bid(node.orelse, after_while_block.bid)

        # Continue building the CFG in the after-while block.
        self.curr_block = after_while_block
        self.after_loop_stack.pop()
        self.loop_guard_stack.pop()

    def visit_If(self, node: ast.If) -> None:

        test_sequence = self.visit(node.test)
        node.test = test_sequence[-1]
        self.populate_body(test_sequence[:-1])

        # Add the If statement at the end of the current block.
        add_stmt(self.curr_block, node)

        # Create a block for the code after the if-else.
        after_if_block: BasicBlock = self.new_block()

        # Create a new block for the body of the if.
        if_body_block: BasicBlock = self.add_edge(
            self.curr_block.bid, self.new_block().bid
        )

        # New block for the body of the else if there is an else clause.
        if node.orelse:
            self.curr_block = self.add_edge(self.curr_block.bid, self.new_block().bid)

            # Visit the children in the body of the else to populate the block.
            self.populate_body_to_next_bid(node.orelse, after_if_block.bid)
        else:
            self.add_edge(self.curr_block.bid, after_if_block.bid)
        # Visit children to populate the if block.
        self.curr_block: BasicBlock = if_body_block
        self.populate_body_to_next_bid(node.body, after_if_block.bid)

        # Continue building the CFG in the after-if block.
        self.curr_block: BasicBlock = after_if_block

    def visit_With(self, node: ast.With) -> None:
        add_stmt(self.curr_block, node)

        with_body_block = self.new_block()
        after_with_block = self.new_block()

        self.add_edge(self.curr_block.bid, with_body_block.bid)
        self.add_edge(self.curr_block.bid, after_with_block.bid)
        self.curr_block = with_body_block
        self.populate_body_to_next_bid(node.body, after_with_block.bid)

        self.curr_block = after_with_block

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        add_stmt(self.curr_block, node)

        with_body_block = self.new_block()
        after_with_block = self.new_block()

        self.add_edge(self.curr_block.bid, with_body_block.bid)
        self.add_edge(self.curr_block.bid, after_with_block.bid)
        self.curr_block = with_body_block
        self.populate_body_to_next_bid(node.body, after_with_block.bid)

        self.curr_block = after_with_block

    # Need to record exception handling stack
    def visit_Raise(self, node: ast.Raise) -> None:
        add_stmt(self.curr_block, node)
        self.curr_block = self.add_edge(self.curr_block.bid, self.new_block().bid)

    def visit_Try(self, node: ast.Try) -> None:
        loop_guard = self.add_loop_block()
        self.curr_block = loop_guard
        add_stmt(
            loop_guard, ast.Try(body=[], handlers=[], orelse=[], finalbody=[])
        )

        try_body_block = self.new_block()
        self.curr_block = self.add_edge(self.curr_block.bid, try_body_block.bid)
        exception_handling_sentinel = self.new_block()
        self.populate_body_to_next_bid(node.body, exception_handling_sentinel.bid)
        add_stmt(exception_handling_sentinel, ast.Name(id="exception handling", ctx=ast.Load()))
        self.curr_block = exception_handling_sentinel

        fake_after_try_block = self.new_block()
        if node.handlers:
            for handler in node.handlers:
                handler_type_block = self.new_block()
                self.curr_block = handler_type_block
                add_stmt(handler_type_block,
                         handler.type if handler.type else ast.Name(id='BaseException', ctx=ast.Load()))
                self.add_edge(
                    exception_handling_sentinel.bid,
                    handler_type_block.bid,
                    ast.Name(id='Except Type', ctx=ast.Load())
                )
                self.curr_block = self.add_edge(self.curr_block.bid, self.new_block().bid)

                self.populate_body_to_next_bid(handler.body, fake_after_try_block.bid)

        if node.orelse:
            before_else_block = self.new_block()
            self.curr_block = before_else_block
            self.add_edge(
                exception_handling_sentinel.bid,
                before_else_block.bid,
                ast.Name(id="No Error", ctx=ast.Load()),
            )

            self.populate_body_to_next_bid(node.orelse, fake_after_try_block.bid)

        self.curr_block = fake_after_try_block
        finally_block = fake_after_try_block
        if node.finalbody:
            self.add_edge(
                exception_handling_sentinel.bid,
                finally_block.bid,
                ast.Name(id="Finally", ctx=ast.Load()),
            )
            after_finally_block = self.new_block()
            self.populate_body_to_next_bid(node.finalbody, after_finally_block.bid)
            self.curr_block = after_finally_block
        else:
            self.add_edge(exception_handling_sentinel.bid, finally_block.bid)

    # If assert fails, AssertionError will be raised.
    # If assert succeeds, execute normal flow.
    def visit_Assert(self, node):
        add_stmt(self.curr_block, node)
        self.cfg.final_blocks.append(self.curr_block)
        self.curr_block = self.add_edge(self.curr_block.bid, self.new_block().bid)

    def visit_Import(self, node: ast.Import) -> None:
        add_stmt(self.curr_block, node)
        self.curr_block = self.add_edge(self.curr_block.bid, self.new_block().bid)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        add_stmt(self.curr_block, node)
        self.curr_block = self.add_edge(self.curr_block.bid, self.new_block().bid)

    def visit_Global(self, node: ast.Global) -> None:
        add_stmt(self.curr_block, node)
        self.curr_block = self.add_edge(self.curr_block.bid, self.new_block().bid)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        add_stmt(self.curr_block, node)
        self.curr_block = self.add_edge(self.curr_block.bid, self.new_block().bid)

    def visit_Expr(self, node: ast.Expr) -> None:

        tmp_var = randoms.RandomUnusedName.gen_unused_name()
        tmp_assign = ast.Assign(
            targets=[ast.Name(id=tmp_var, ctx=ast.Store())],
            value=node.value
        )
        self.visit(tmp_assign)

    def visit_Pass(self, node: ast.Pass) -> None:
        add_stmt(self.curr_block, node)
        self.curr_block = self.add_edge(self.curr_block.bid, self.new_block().bid)

    def visit_Break(self, node: ast.Break) -> None:
        add_stmt(self.curr_block, node)
        assert len(self.after_loop_stack), "Found break not inside loop"
        self.add_edge(self.curr_block.bid, self.after_loop_stack[-1].bid)

    def visit_Continue(self, node: ast.Continue) -> None:
        add_stmt(self.curr_block, node)
        assert self.loop_guard_stack, "Found continue not inside loop"
        self.add_edge(self.curr_block.bid, self.loop_guard_stack[-1].bid)

    ################################################################
    ################################################################
    # expr
    # For now I think there are several basic types in Python.
    # ast.Num, such as 1
    # ast.Str, such as 'jojo'
    # ast.FormattedValue inside JoinedStr, note that I didn't come across a case that it was alone
    # ast.JoinedStr, such as f'{a}'
    # ast.Bytes, such as b'a'
    # ast.NameConstant, such as True
    # ast.Ellipsis, it's ...
    # ast.Constant, I didn't know how it would be used in python 3.7
    # ast.Name, a name represents a value evaluated.
    #
    # We care about these basic types since we need some criterion to stop recursion of expanding expressions.
    #

    def decompose_expr(self, expr: ast.expr, new_expr_sequence):
        decomposed_expr = self.visit(expr)
        new_expr_sequence += decomposed_expr[:-1]
        tmp_var = randoms.RandomVariableName.gen_random_name()
        new_expr_sequence.append(
            ast.Assign(
                targets=[ast.Name(id=tmp_var, ctx=ast.Store())],
                value=decomposed_expr[-1]
            )
        )

        return ast.Name(id=tmp_var, ctx=ast.Load())

    def decompose_expr_list(self, expr_list, new_expr_sequence):
        new_arg_list = []

        for expr in expr_list:
            if type(expr) not in BASIC_TYPES:
                tmp_name = self.decompose_expr(expr, new_expr_sequence)
                new_arg_list.append(tmp_name)
            else:
                new_arg_list.append(expr)

        return new_arg_list

    ################################################################
    ################################################################
    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        new_expr_sequence = []

        new_arg_list = self.decompose_expr_list(node.values, new_expr_sequence)

        node.values = new_arg_list
        return new_expr_sequence + [node]

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        new_expr_sequence = []

        new_arg_list = self.decompose_expr_list([node.left, node.right], new_expr_sequence)

        node.left, node.right = new_arg_list
        return new_expr_sequence + [node]

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        new_expr_sequence = []
        expr = node.operand
        if type(expr) not in BASIC_TYPES:
            tmp_name = self.decompose_expr(expr, new_expr_sequence)
            node.operand = tmp_name

        return new_expr_sequence + [node]

    def visit_Lambda(self, node: ast.Lambda) -> Any:
        tmp_lambda_name = randoms.RandomLambdaName.gen_lambda_name()
        tmp_function_def = ast.FunctionDef(
            name=tmp_lambda_name,
            args=node.args,
            body=[ast.Return(node.body)],
            decorator_list=[],
            returns=None
        )
        tmp_function_name = ast.Name(
            id=tmp_lambda_name,
            ctx=ast.Load()
        )

        return [tmp_function_def, tmp_function_name]

    # def _visit_IfExp(self, node: ast.IfExp) -> List[ast.If]:
    #     return [
    #         ast.If(
    #             test=node.test,
    #             body=self._visit_IfExp(node.body)
    #             if type(node.body) == ast.IfExp
    #             else [ast.Return(value=node.body)],
    #             orelse=self._visit_IfExp(node.orelse)
    #             if type(node.orelse) == ast.IfExp
    #             else [ast.Return(value=node.orelse)],
    #         )
    #     ]

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        new_expr_sequence = []

        new_arg_list = self.decompose_expr_list([node.test, node.body, node.orelse], new_expr_sequence)

        node.test, node.body, node.orelse = new_arg_list
        return new_expr_sequence + [node]

    def visit_Dict(self, node: ast.Dict) -> Any:
        return [node]

    def visit_Set(self, node: ast.Set) -> Any:
        return [node]

    def visit_ListComp(self, node: ast.ListComp) -> Any:

        new_expr_sequence = []
        listcomp_var = randoms.RandomVariableName.gen_random_name()
        new_expr_sequence.append(
            ast.Assign(
                targets=[ast.Name(id=listcomp_var, ctx=ast.Store())],
                value=ast.List(elts=[], ctx=ast.Load())
            )
        )
        new_expr_sequence += self._visit_ListComp(listcomp_var, node.elt, node.generators)
        new_expr_sequence.append(ast.Name(id=listcomp_var, ctx=ast.Load()))
        return new_expr_sequence

    def _visit_ListComp(self, listcomp_var: str, elt: ast.expr, generators: List[ast.comprehension]) -> Any:
        if not generators:
            new_expr_sequence = []
            tmp_call = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=listcomp_var, ctx=ast.Load()),
                    attr="append",
                    ctx=ast.Load(),
                ),
                args=[elt],
                keywords=[],
            )
            if type(elt) not in BASIC_TYPES:
                tmp_name = self.decompose_expr(elt, new_expr_sequence)
                tmp_call.args = [tmp_name]

            new_expr_sequence.append(ast.Expr(value=tmp_call))
            return new_expr_sequence
        else:
            return [
                ast.For(
                    target=generators[0].target,
                    iter=generators[0].iter,
                    body=[
                        ast.If(
                            test=self.combine_conditions(generators[0].ifs),
                            body=self._visit_ListComp(listcomp_var, elt, generators[1:]),
                            orelse=[],
                        )
                    ]
                    if generators[0].ifs
                    else self._visit_ListComp(listcomp_var, elt, generators[1:]),
                    orelse=[],
                )
            ]

    def visit_SetComp(self, node: ast.SetComp) -> Any:
        new_expr_sequence = []
        setcomp_var = randoms.RandomVariableName.gen_random_name()
        new_expr_sequence.append(
            ast.Assign(
                targets=[ast.Name(id=setcomp_var, ctx=ast.Store())],
                value=ast.Call(args=[], func=ast.Name(id='set', ctx=ast.Load()), keywords=[])
            )
        )
        new_expr_sequence += self._visit_SetComp(setcomp_var, node.elt, node.generators)
        new_expr_sequence.append(ast.Name(id=setcomp_var, ctx=ast.Load()))
        return new_expr_sequence

    def _visit_SetComp(self, setcomp_var: str, elt: ast.expr, generators: List[ast.comprehension]):
        if not generators:
            new_expr_sequence = []
            tmp_call = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=setcomp_var, ctx=ast.Load()),
                    attr="add",
                    ctx=ast.Load(),
                ),
                args=[elt],
                keywords=[]
            )
            if type(elt) not in BASIC_TYPES:
                tmp_name = self.decompose_expr(elt)
                tmp_call.args = [tmp_name]

            new_expr_sequence.append(ast.Expr(value=tmp_call))
            return new_expr_sequence
        else:
            return [
                ast.For(
                    target=generators[0].target,
                    iter=generators[0].iter,
                    body=[
                        ast.If(
                            test=self.combine_conditions(generators[0].ifs),
                            body=self._visit_SetComp(setcomp_var, elt, generators[1:]),
                            orelse=[],
                        )
                    ]
                    if generators[0].ifs
                    else self._visit_SetComp(setcomp_var, elt, generators[1:]),
                    orelse=[],
                )
            ]

    def visit_DictComp(self, node: ast.DictComp) -> Any:
        new_expr_sequence = []
        dictcomp_var = randoms.RandomVariableName.gen_random_name()
        new_expr_sequence.append(
            ast.Assign(
                targets=[ast.Name(id=dictcomp_var, ctx=ast.Store())],
                value=ast.Dict(keys=[], values=[])
            )
        )
        new_expr_sequence += self._visit_DictComp(dictcomp_var, node.key, node.value, node.generators)
        new_expr_sequence.append(ast.Name(id=dictcomp_var, ctx=ast.Load()))
        return new_expr_sequence

    def _visit_DictComp(self, dictcomp_var: str, key: ast.expr, value: ast.expr, generators):
        if not generators:
            new_expr_sequence = []

            tmp_index = ast.Index(value=key)
            if type(key) not in BASIC_TYPES:
                tmp_name = self.decompose_expr(key, new_expr_sequence)
                tmp_index.value = tmp_name
            tmp_subscript = ast.Subscript(
                value=ast.Name(id=dictcomp_var, ctx=ast.Load()),
                slice=tmp_index,
                ctx=ast.Store(),
            )

            tmp_assign = ast.Assign(
                targets=[tmp_subscript],
                value=value,
            )
            if type(value) not in BASIC_TYPES:
                tmp_name = self.decompose_expr(value, new_expr_sequence)
                tmp_assign.value = tmp_name

            new_expr_sequence.append(tmp_assign)
            return new_expr_sequence
        else:
            return [
                ast.For(
                    target=generators[0].target,
                    iter=generators[0].iter,
                    body=[
                        ast.If(
                            test=self.combine_conditions(generators[0].ifs),
                            body=self._visit_DictComp(dictcomp_var, key, value, generators[1:]),
                            orelse=[]
                        )
                    ]
                    if generators[0].ifs
                    else self._visit_DictComp(dictcomp_var, key, value, generators[1:]),
                    orelse=[]
                )
            ]

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> Any:
        new_expr_sequence = []
        generator_var = randoms.RandomGeneratorName.gen_generator_name()
        new_expr_sequence.append(
            ast.FunctionDef(
                name=generator_var,
                args=ast.arguments(args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body=self._visit_GeneratorExp(node.elt, node.generators),
                decorator_list=[],
                returns=None
            )
        )
        new_expr_sequence.append(
            ast.Call(
                func=ast.Name(id=generator_var, ctx=ast.Load()),
                args=[],
                keywords=[]
            )
        )
        # tmp_var = randoms.RandomVariableName.gen_random_name()
        # new_expr_sequence.append(
        #     ast.Assign(
        #         targets=[ast.Name(id=tmp_var, ctx=ast.Store())],
        #         value=ast.Call(
        #             func=ast.Name(id=generator_var, ctx=ast.Load()),
        #             args=[],
        #             keywords=[]
        #         )
        #     )
        # )
        # new_expr_sequence.append(ast.Name(id=tmp_var, ctx=ast.Load()))
        return new_expr_sequence

    def _visit_GeneratorExp(self, elt: ast.expr, generators: List[ast.comprehension]):
        if not generators:
            new_expr_sequence = []
            tmp_yield = ast.Yield(value=elt)
            if type(elt) not in BASIC_TYPES:
                tmp_name = self.decompose_expr(elt, new_expr_sequence)
                tmp_yield.value = tmp_name
            new_expr_sequence.append(ast.Expr(value=tmp_yield))
            return new_expr_sequence
        else:
            return [
                ast.For(
                    target=generators[0].target,
                    iter=generators[0].iter,
                    body=[
                        ast.If(
                            test=self.combine_conditions(generators[0].ifs),
                            body=self._visit_GeneratorExp(elt, generators[1:]),
                            orelse=[],
                        )
                    ]
                    if generators[0].ifs
                    else self._visit_GeneratorExp(elt, generators[1:]),
                    orelse=[],
                )
            ]

    def visit_Await(self, node: ast.Await) -> Any:
        new_expr_sequence = []

        expr = node.value
        if type(expr) not in BASIC_TYPES:
            tmp_name = self.decompose_expr(expr, new_expr_sequence)
            node.value = tmp_name

        return new_expr_sequence + [node]

    def visit_Yield(self, node: ast.Yield) -> Any:
        if node.value is None:
            return [node]

        new_expr_sequence = []

        expr = node.value
        if type(expr) not in BASIC_TYPES:
            tmp_name = self.decompose_expr(expr, new_expr_sequence)
            node.value = tmp_name

        return new_expr_sequence + [node]

    def visit_YieldFrom(self, node: ast.YieldFrom) -> Any:
        new_expr_sequence = []

        expr = node.value
        if type(expr) not in BASIC_TYPES:
            tmp_name = self.decompose_expr(expr, new_expr_sequence)
            node.value = tmp_name

        return new_expr_sequence + [node]

    def visit_Compare(self, node: ast.Compare) -> Any:
        new_expr_sequence = []
        new_arg_list = []
        for expr in [node.left] + node.comparators:
            if type(expr) not in BASIC_TYPES:
                tmp_name = self.decompose_expr(expr, new_expr_sequence)
                new_arg_list.append(tmp_name)
            else:
                new_arg_list.append(expr)

        node.left, node.comparators = new_arg_list[0], new_arg_list[1:]
        return new_expr_sequence + [node]

    def visit_Call(self, node: ast.Call) -> Any:
        new_expr_sequence = []
        if type(node.func) == ast.Lambda:
            tmp_name = self.decompose_expr(node.func, new_expr_sequence)
            tmp_call = ast.Call(
                args=node.args,
                func=tmp_name,
                keywords=[]
            )
            return new_expr_sequence + [tmp_call]

        new_arg_list = []
        for expr in node.args:
            if type(expr) not in BASIC_TYPES:
                tmp_name = self.decompose_expr(expr, new_expr_sequence)
                new_arg_list.append(tmp_name)
            else:
                new_arg_list.append(expr)

        node.args = new_arg_list
        return new_expr_sequence + [node]

    def visit_Num(self, node: ast.Num) -> Any:
        return [node]

    def visit_Str(self, node: ast.Str) -> Any:
        return [node]

    def visit_FormattedValue(self, node: ast.FormattedValue) -> Any:
        return [node]

    def visit_JoinedStr(self, node: ast.JoinedStr) -> Any:
        return [node]

    def visit_Bytes(self, node: ast.Bytes) -> Any:
        return [node]

    def visit_NameConstant(self, node: ast.NameConstant) -> Any:
        return [node]

    def visit_Ellipsis(self, node: ast.Ellipsis) -> Any:
        return [node]

    def visit_Constant(self, node: ast.Constant) -> Any:
        return [node]

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        new_expr_sequence = []
        expr = node.value

        if type(expr) not in BASIC_TYPES:
            tmp_name = self.decompose_expr(expr, new_expr_sequence)
            node.value = tmp_name

        return new_expr_sequence + [node]

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        new_expr_sequence = []
        expr = node.value

        if type(expr) not in BASIC_TYPES:
            tmp_name = self.decompose_expr(expr, new_expr_sequence)
            node.value = tmp_name

        return new_expr_sequence + [node]

    def visit_Starred(self, node: ast.Starred) -> Any:
        new_expr_sequence = []
        expr = node.value

        if type(expr) not in BASIC_TYPES:
            tmp_name = self.decompose_expr(expr, new_expr_sequence)
            node.value = tmp_name

        return new_expr_sequence + [node]

    def visit_Name(self, node: ast.Name) -> Any:
        return [node]

    def visit_List(self, node: ast.List) -> Any:
        new_expr_sequence = []
        new_arg_list = self.decompose_expr_list(node.elts, new_expr_sequence)
        node.elts = new_arg_list
        return new_expr_sequence + [node]

    def visit_Tuple(self, node: ast.Tuple) -> Any:
        new_expr_sequence = []
        new_arg_list = self.decompose_expr_list(node.elts, new_expr_sequence)
        node.elts = new_arg_list
        return new_expr_sequence + [node]
