#  Copyright 2022 Layne Liu
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import annotations

import ast
import logging
import os
from typing import Dict, List, Tuple, Set, Optional, Any

import astor
import graphviz as gv

from . import randoms

BASIC_TYPES = (
    ast.Num,
    ast.Str,
    ast.FormattedValue,
    ast.JoinedStr,
    ast.Bytes,
    ast.NameConstant,
    ast.Ellipsis,
    ast.Constant,
    ast.Name,
)


class BlockId:
    counter: int = 0

    @classmethod
    def gen_block_id(cls) -> int:
        cls.counter += 1
        return cls.counter


class BasicBlock(object):
    def __init__(self, bid: int):
        self.bid: int = bid
        self.stmt = []
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
                if type(stmt)
                in [
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

    def __str__(self):
        return "Block ID: {}".format(self.bid)


class CallAndAssignBlock(BasicBlock):
    def __init__(self, basic_block: BasicBlock):
        super().__init__(basic_block.bid)
        self.prev = basic_block.prev
        self.next = basic_block.prev
        self.stmt = basic_block.stmt

        self.call_block = None
        self.call_id = None
        self.exit_block = None
        self.exit_id = None

    def add_call_return(self, call_block: BasicBlock, exit_block: BasicBlock):
        self.call_block = call_block
        self.exit_block = exit_block
        self.call_id = call_block.bid
        self.exit_id = exit_block.bid

    def __str__(self):
        return "Block ID: {}".format(self.bid)


def add_stmt(block: BasicBlock, stmt) -> None:
    block.stmt.append(stmt)


class CFG:
    def __init__(self, name: str, is_func: bool = False, is_class: bool = False):
        self.name: str = name

        self.start: Optional[BasicBlock] = None
        self.start_block: Optional[BasicBlock] = None
        self.final_block: Optional[BasicBlock] = None

        # is this CFG for function ?
        self.is_func: bool = is_func
        # the assignment variable name for return
        self.return_name = None
        # does the function have return labels?
        self.has_return: bool = False
        # Function name to (Args, CFG)
        self.func_cfgs: Dict[Tuple[str, int], (List[str, ast.AST], CFG)] = {}
        self.async_func_cfgs: Dict[Tuple[str, int], (List[str, ast.AST], CFG)] = {}

        # deal with classes
        self.is_class: bool = is_class
        self.class_cfgs: Dict[Tuple[str, int], CFG] = {}

        self.blocks: Dict[int, BasicBlock] = {}
        self.edges: Dict[Tuple[int, int], Optional[ast.AST]] = {}
        self.graph: Optional[gv.dot.Digraph] = None
        self.flows: Set[Tuple[int, int]] = set()
        self.labels: Set[int] = set()
        self.fake_flows: Set[Tuple[int, int]] = set()
        self.inter_flows: Set[Tuple[int, Optional[int], Optional[int], int]] = set()
        self.vars: Set[str] = set()

    def _traverse(self, block: BasicBlock, visited: Set[int] = set()) -> None:
        if block.bid not in visited:
            visited.add(block.bid)
            additional = ""
            if isinstance(self.blocks[block.bid].stmt[0], ast.Assign) and isinstance(
                self.blocks[block.bid].stmt[0].value, ast.Call
            ):
                for call_id, _, _, exit_return_id in self.inter_flows:
                    if call_id == block.bid:
                        additional += "Call the function"
                    if exit_return_id == block.bid:
                        additional += "Exit and return the function"
            self.graph.node(str(block.bid), label=block.stmt_to_code() + additional)
            for next_bid in block.next:
                self._traverse(self.blocks[next_bid], visited)
                self.graph.edge(
                    str(block.bid),
                    str(next_bid),
                    label=astor.to_source(self.edges[(block.bid, next_bid)])
                    if self.edges[(block.bid, next_bid)]
                    else "",
                )

    def generate(self, fmt: str, name: str) -> gv.dot.Digraph:
        self.graph = gv.Digraph(name="cluster_" + self.name, format=fmt)
        self.graph.attr(label=name)
        self._traverse(self.start)
        for (func_name, func_label), funcCFG in self.func_cfgs.items():
            self.graph.subgraph(
                funcCFG[1].generate(fmt, func_name + " at label {}".format(func_label))
            )
        for (func_name, func_label), funcCFG in self.async_func_cfgs.items():
            self.graph.subgraph(funcCFG[1].generate(fmt, func_name))
        for (class_name, class_label), classCFG in self.class_cfgs.items():
            self.graph.subgraph(
                classCFG.generate(fmt, class_name + " at label {}".format(class_label))
            )
        return self.graph

    def show(
        self,
        filepath: str = "output",
        fmt: str = "png",
        show: bool = True,
        name: str = None,
    ) -> None:
        self.generate(fmt, name)
        path = os.path.normpath(filepath)
        self.graph.render(path, view=show, cleanup=True)


class CFGVisitor(ast.NodeVisitor):
    def __init__(
        self,
        isolation: bool = False,
        trans_assert: bool = False,
        trans_for: bool = False,
        is_func: bool = False,
        is_class: bool = False,
    ):
        super().__init__()
        self.cfg: Optional[CFG] = None
        self.curr_block: Optional[BasicBlock] = None

        self.isolation = isolation
        self.trans_assert = trans_assert
        self.trans_for = trans_for
        self.is_func = is_func
        self.is_class = is_class
        self.after_loop_stack: List[BasicBlock] = []
        self.loop_guard_stack: List[BasicBlock] = []
        self.raise_except_stack: List[BasicBlock] = []
        self.raise_final_stack: List[BasicBlock] = []

    def build(self, name: str, tree: ast.Module) -> CFG:
        self.cfg = CFG(name, self.is_func, self.is_class)
        self.curr_block = self.new_block()
        self.visit(tree)
        self.remove_empty_blocks(self.cfg.start)
        self.refactor_flows_and_labels()
        return self.cfg

    def new_block(self) -> BasicBlock:
        bid: int = BlockId.gen_block_id()
        self.cfg.blocks[bid] = BasicBlock(bid)
        return self.cfg.blocks[bid]

    def add_edge(self, frm_id: int, to_id: int, condition=None) -> BasicBlock:
        self.cfg.blocks[frm_id].next.append(to_id)
        self.cfg.blocks[to_id].prev.append(frm_id)
        self.cfg.edges[(frm_id, to_id)] = condition
        return self.cfg.blocks[to_id]

    def add_loop_block(self) -> BasicBlock:
        if self.curr_block.is_empty() and not self.curr_block.has_next():
            return self.curr_block
        else:
            loop_block = self.new_block()
            self.add_edge(self.curr_block.bid, loop_block.bid)

    def add_FuncCFG(self, tree: ast.FunctionDef) -> None:
        name_id_pair = (tree.name, self.curr_block.bid)

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

        visitor: CFGVisitor = CFGVisitor(self.isolation, is_func=True, is_class=False)
        func_cfg: CFG = visitor.build(tree.name, ast.Module(body=tree.body))
        self.cfg.func_cfgs[name_id_pair] = (arg_list, func_cfg)

    def add_AsyncFuncCFG(self, tree: ast.FunctionDef) -> None:
        name_id_pair = (tree.name, self.curr_block.bid)

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
        self.cfg.async_func_cfgs[name_id_pair] = (arg_list, func_cfg)

    def add_ClassCFG(self, node: ast.ClassDef):
        name_id_pair = (node.name, self.curr_block.bid)
        class_body: ast.Module = ast.Module(body=node.body)
        visitor: CFGVisitor = CFGVisitor(self.isolation, is_func=False, is_class=True)
        class_cfg: CFG = visitor.build(node.name, class_body)
        print(class_cfg.start_block.bid, class_cfg.final_block.bid)
        self.cfg.class_cfgs[name_id_pair] = class_cfg

    def remove_empty_blocks(self, block: BasicBlock, visited: Set[int] = set()) -> None:
        if block.bid not in visited:
            visited.add(block.bid)
            if block.is_empty():
                for prev_bid in list(block.prev):
                    prev_block = self.cfg.blocks[prev_bid]
                    for next_bid in list(block.next):
                        next_block = self.cfg.blocks[next_bid]
                        self.add_edge(prev_bid, next_bid)
                        self.cfg.edges.pop((block.bid, next_bid), None)
                        next_block.remove_from_prev(block.bid)
                    self.cfg.edges.pop((prev_bid, block.bid), None)
                    prev_block.remove_from_next(block.bid)

                block.prev.clear()
                for next_bid in list(block.next):
                    self.remove_empty_blocks(self.cfg.blocks[next_bid], visited)
                block.next.clear()

            else:
                for next_bid in list(block.next):
                    self.remove_empty_blocks(self.cfg.blocks[next_bid], visited)

    def refactor_flows_and_labels(self):
        for fst_id, snd_id in self.cfg.edges:
            self.cfg.flows.add((fst_id, snd_id))

        self.cfg.flows -= self.cfg.fake_flows

        for fst_label, snd_label in self.cfg.flows:
            self.cfg.labels.add(fst_label)
            self.cfg.labels.add(snd_label)

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
        self.cfg.start = self.curr_block

        # pre structure cleaning
        if self.isolation:
            self.cfg.start_block = self.curr_block
            add_stmt(self.cfg.start_block, ast.Pass())
            self.cfg.final_block = self.new_block()

            # if is_func, we have to generate a name for assignments
            if self.cfg.is_func:
                return_name = randoms.RandomReturnName.gen_return_name()
                self.cfg.return_name = return_name

            self.curr_block = self.add_edge(self.curr_block.bid, self.new_block().bid)

        # visit ast
        self.generic_visit(node)

        # post structure cleaning
        if self.isolation:
            self.add_edge(self.curr_block.bid, self.cfg.final_block.bid)
            if self.cfg.is_func:
                if self.cfg.has_return:
                    return_node = ast.Return(
                        value=ast.Name(id=self.cfg.return_name, ctx=ast.Load())
                    )
                    add_stmt(self.cfg.final_block, return_node)
                else:
                    assign_node = ast.Assign(
                        targets=[ast.Name(id=self.cfg.return_name, ctx=ast.Store())],
                        value=ast.NameConstant(value=None),
                    )
                    add_stmt(self.cfg.final_block, assign_node)
                    self.curr_block = self.add_edge(
                        self.cfg.final_block.bid, self.new_block().bid
                    )
                    return_node = ast.Return(
                        value=ast.Name(id=self.cfg.return_name, ctx=ast.Load())
                    )
                    add_stmt(self.curr_block, return_node)
                    self.cfg.final_block = self.curr_block
            else:
                add_stmt(self.cfg.final_block, ast.Pass())

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # We only display fields in classes.
        add_stmt(self.curr_block, node)
        self.add_FuncCFG(node)
        self.curr_block = self.add_edge(self.curr_block.bid, self.new_block().bid)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        add_stmt(self.curr_block, node)
        self.add_AsyncFuncCFG(node)
        self.curr_block = self.add_edge(self.curr_block.bid, self.new_block().bid)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # add_stmt(self.curr_block, node)
        add_stmt(self.curr_block, node)
        self.curr_block = self.add_edge(self.curr_block.bid, self.new_block().bid)
        self.add_ClassCFG(node)

    def visit_Return(self, node: ast.Return) -> None:
        if self.cfg.is_func:
            self.cfg.has_return = True

        if node.value is None:
            new_value = ast.NameConstant(value=None)
            new_assign = ast.Assign(
                targets=[ast.Name(id=self.cfg.return_name, ctx=ast.Store())],
                value=new_value,
            )
            add_stmt(self.curr_block, new_assign)
            self.add_edge(self.curr_block.bid, self.cfg.final_block.bid)
            self.curr_block = self.new_block()
        else:
            new_expr_sequence = self.visit(node.value)
            if len(new_expr_sequence) == 1:
                new_assign = ast.Assign(
                    targets=[ast.Name(id=self.cfg.return_name, ctx=ast.Store())],
                    value=node.value,
                )
                add_stmt(self.curr_block, new_assign)
                self.add_edge(self.curr_block.bid, self.cfg.final_block.bid)
                self.curr_block = self.new_block()
            else:
                return_stmt = ast.Return(value=new_expr_sequence[-1])
                generated_return_sequence = new_expr_sequence[:-1] + [return_stmt]
                self.populate_body(generated_return_sequence)

    def visit_Delete(self, node: ast.Delete) -> None:
        add_stmt(self.curr_block, node)
        self.curr_block = self.add_edge(self.curr_block.bid, self.new_block().bid)

    def extract_var(self, target: ast.expr):
        if isinstance(target, ast.Name):
            self.cfg.vars.add(target.id)
            return
        elif isinstance(target, (ast.Attribute, ast.Subscript, ast.Starred)):
            self.extract_var(target.value)
            return
        else:
            assert False

    def extract_vars(self, targets: List[ast.expr]):
        for target in targets:
            self.extract_var(target)

    def add_inter_flows(self, call_label: int, return_label: int):
        # exit label is return label in each function.
        self.cfg.inter_flows.add((call_label, None, None, return_label))
        self.cfg.fake_flows.add((call_label, return_label))

    def transform_boolop_assign(self, node: ast.Assign) -> List:
        right_value: ast.BoolOp = node.value
        assign_list = [
            ast.Assign(targets=node.targets, value=value)
            for value in right_value.values
        ]
        current_sequence = assign_list[-1:]
        for assign in reversed(assign_list[:-1]):
            tmp_if = ast.If(
                test=assign.value
                if isinstance(right_value.op, ast.And)
                else ast.UnaryOp(op=ast.Not(), operand=assign.value),
                body=current_sequence,
                orelse=[],
            )
            current_sequence = [assign, tmp_if]

        return current_sequence

    def visit_Assign(self, node: ast.Assign) -> None:
        # deal with bool op, https://snarky.ca/unravelling-boolean-operations/
        if isinstance(node.value, ast.BoolOp):
            new_sequence = self.transform_boolop_assign(node)
            self.populate_body(new_sequence)
            return

        new_expr_sequence = self.visit(node.value)

        if len(new_expr_sequence) == 1:
            # extracts variables from assignments to self.cfg.vars
            self.extract_vars(node.targets)

            if isinstance(node.value, ast.Call):

                add_stmt(self.curr_block, node)

                return_block = self.new_block()
                return_block.pass_through_name = (
                    randoms.RandomPassThroughName.gen_pass_through_name()
                )
                return_block.pass_through_value = None
                self.add_edge(self.curr_block.bid, return_block.bid)
                logging.debug(
                    "Call block: {}, return block: {}".format(
                        self.curr_block.bid, return_block.bid
                    )
                )
                self.add_inter_flows(self.curr_block.bid, return_block.bid)

                self.curr_block = return_block

            add_stmt(self.curr_block, node)
            self.curr_block = self.add_edge(self.curr_block.bid, self.new_block().bid)
            return

        new_assign = ast.Assign(targets=node.targets, value=new_expr_sequence[-1])
        new_sequence: List = new_expr_sequence[:-1] + [new_assign]
        if isinstance(node.value, (ast.ListComp, ast.SetComp, ast.DictComp)):
            new_sequence.append(
                ast.Delete(
                    targets=[ast.Name(id=new_expr_sequence[-1].id, ctx=ast.Del())]
                )
            )
        source = ""
        for expr in new_sequence:
            source += astor.to_source(expr)
        logging.debug("Decomposed expr: " + source)
        self.populate_body(new_sequence)
        return

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        add_stmt(self.curr_block, node)
        self.curr_block = self.add_edge(self.curr_block.bid, self.new_block().bid)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value:
            new_assign: ast.Assign = ast.Assign(targets=[node.target], value=node.value)
            self.visit(new_assign)
        else:
            logging.debug("AnnAssign without value. Just ignore it")

    def visit_For(self, node: ast.For) -> None:
        if False:
            new_iter: str = randoms.RandomIterable.gen_iter()
            new_assign: ast.Assign = ast.Assign(
                targets=[ast.Name(id=new_iter, ctx=ast.Store())],
                value=ast.Call(
                    args=[node.iter],
                    func=ast.Name(id="iter", ctx=ast.Load()),
                    keywords=[],
                ),
            )
            new_while: ast.While = ast.While(
                test=ast.NameConstant(value=True),
                body=[],
                orelse=[],
            )

            new_try: ast.Try = ast.Try(
                body=[
                    ast.Assign(
                        targets=[node.target],
                        value=ast.Call(
                            func=ast.Name(id="next", ctx=ast.Load()),
                            args=[ast.Name(id=new_iter, ctx=ast.Load())],
                            keywords=[],
                        ),
                    )
                ],
                handlers=[
                    ast.ExceptHandler(
                        type=ast.Name(
                            id="StopIteration",
                            ctx=ast.Load(),
                        ),
                        name=None,
                        body=[ast.Break()],
                    ),
                    ast.ExceptHandler(
                        type=ast.Name(
                            id="Exception",
                            ctx=ast.Load(),
                        ),
                        name=None,
                        body=[
                            ast.Raise(
                                cause=None,
                                exc=None,
                            )
                        ],
                    ),
                ],
                orelse=[],
                finalbody=[],
            )

            new_while.body.append(new_try)
            new_while.body.extend(node.body)

            new_del: ast.Delete = ast.Delete([ast.Name(id=new_iter, ctx=ast.Del())])

            self.populate_body([new_assign, new_while, new_del])
        elif self.trans_for:
            new_call: ast.Call = ast.Call(
                args=[node.iter], func=ast.Name(id="list", ctx=ast.Load()), keywords=[]
            )
            iter_sequence: List = self.visit(new_call)

            new_var = randoms.RandomVariableName.gen_random_name()
            new_assign = ast.Assign(
                targets=[ast.Name(id=new_var, ctx=ast.Store())],
                value=iter_sequence[-1],
            )
            new_name = ast.Name(id=new_var, ctx=ast.Load())

            iter_sequence = iter_sequence[:-1] + [new_assign]
            # self.populate_body(iter_sequence[:-1])

            new_while: ast.While = ast.While(
                test=new_name,
                body=[
                    ast.Assign(
                        targets=[node.target],
                        value=ast.Subscript(
                            value=new_name,
                            slice=ast.Num(n=0),
                            ctx=ast.Load(),
                        ),
                    )
                ]
                + node.body
                + [
                    ast.Assign(
                        targets=[new_name],
                        value=ast.Subscript(
                            value=new_name,
                            slice=ast.Slice(lower=ast.Num(n=1), upper=None, step=None),
                            ctx=ast.Load(),
                        ),
                    )
                ],
                orelse=node.orelse,
            )
            iter_sequence.append(new_while)
            self.populate_body(iter_sequence)
        else:
            iter_sequence = self.visit(node.iter)
            self.populate_body(iter_sequence[:-1])
            node.iter = iter_sequence[-1]

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
        if self.raise_except_stack and self.raise_except_stack[-1] is not None:
            self.add_edge(self.curr_block.bid, self.raise_except_stack[-1].bid)
        if self.raise_final_stack and self.raise_final_stack[-1] is not None:
            self.add_edge(self.curr_block.bid, self.raise_final_stack[-1].bid)
        self.curr_block = self.new_block()

    def visit_Try(self, node: ast.Try) -> None:
        loop_guard = self.add_loop_block()
        self.curr_block = loop_guard
        add_stmt(loop_guard, ast.Try(body=[], handlers=[], orelse=[], finalbody=[]))

        try_body_block = self.new_block()
        self.curr_block = self.add_edge(self.curr_block.bid, try_body_block.bid)
        exception_handling_sentinel = self.new_block()
        self.raise_except_stack.append(exception_handling_sentinel)
        self.populate_body_to_next_bid(node.body, exception_handling_sentinel.bid)
        self.raise_except_stack.pop()
        add_stmt(
            exception_handling_sentinel,
            ast.Name(id="exception handling", ctx=ast.Load()),
        )
        self.curr_block = exception_handling_sentinel

        fake_after_try_block = self.new_block()
        if node.handlers:
            self.raise_except_stack.append(None)
            self.raise_final_stack.append(
                fake_after_try_block if node.finalbody else None
            )
            for handler in node.handlers:
                handler_type_block = self.new_block()
                self.curr_block = handler_type_block
                add_stmt(
                    handler_type_block,
                    handler.type
                    if handler.type
                    else ast.Name(id="BaseException", ctx=ast.Load()),
                )
                self.add_edge(
                    exception_handling_sentinel.bid,
                    handler_type_block.bid,
                    ast.Name(id="Except Type", ctx=ast.Load()),
                )
                self.curr_block = self.add_edge(
                    self.curr_block.bid, self.new_block().bid
                )

                self.populate_body_to_next_bid(handler.body, fake_after_try_block.bid)
            self.raise_final_stack.pop()
            self.raise_except_stack.pop()

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
        if self.trans_assert:
            new_if: ast.If = ast.If(
                test=ast.UnaryOp(op=ast.Not(), operand=node.test),
                body=[
                    ast.Raise(
                        exc=ast.Name(id="AssertionError", ctx=ast.Load()), cause=None
                    )
                ]
                if node.msg is None
                else [
                    ast.Raise(
                        exc=ast.Call(
                            args=[node.msg],
                            func=ast.Name(id="AssertionError", ctx=ast.Load()),
                            keywords=[],
                        ),
                        cause=None,
                    )
                ],
                orelse=[],
            )
            self.visit(new_if)
        else:
            add_stmt(self.curr_block, node)
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
            targets=[ast.Name(id=tmp_var, ctx=ast.Store())], value=node.value
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

    # decompose a single expression.
    # new_expr_sequence stores a list of temporal statements

    def decompose_expr(self, expr: ast.expr, new_expr_sequence):
        decomposed_expr = self.visit(expr)
        new_expr_sequence.extend(decomposed_expr[:-1])
        tmp_var = randoms.RandomVariableName.gen_random_name()
        new_expr_sequence.append(
            ast.Assign(
                targets=[ast.Name(id=tmp_var, ctx=ast.Store())],
                value=decomposed_expr[-1],
            )
        )

        return ast.Name(id=tmp_var, ctx=ast.Load())

    def decompose_expr_list(self, expr_list, new_expr_sequence):
        new_arg_list = []

        for expr in expr_list:
            if not isinstance(expr, BASIC_TYPES):
                tmp_name = self.decompose_expr(expr, new_expr_sequence)
                new_arg_list.append(tmp_name)
            else:
                new_arg_list.append(expr)

        return new_arg_list

    ################################################################
    ################################################################
    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        new_var: str = randoms.RandomVariableName.gen_random_name()
        new_assign = ast.Assign(
            targets=[ast.Name(id=new_var, ctx=ast.Store())], value=node
        )
        new_name: ast.Name = ast.Name(
            id=new_var,
            ctx=ast.Load(),
        )

        return [new_assign, new_name]

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        new_expr_sequence = []

        new_arg_list = self.decompose_expr_list(
            [node.left, node.right], new_expr_sequence
        )

        node.left, node.right = new_arg_list
        return new_expr_sequence + [node]

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        new_expr_sequence = []
        expr = node.operand
        if not isinstance(expr, BASIC_TYPES):
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
            returns=None,
        )
        tmp_function_name = ast.Name(id=tmp_lambda_name, ctx=ast.Load())

        return [tmp_function_def, tmp_function_name]

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        tmp_var: str = randoms.RandomVariableName.gen_random_name()
        tmp_name: ast.Name = ast.Name(id=tmp_var, ctx=ast.Store())
        new_if: ast.If = ast.If(
            test=node.test,
            body=[ast.Assign(targets=[tmp_name], value=node.body)],
            orelse=[ast.Assign(targets=[tmp_name], value=node.orelse)],
        )

        return [new_if, tmp_name]

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
                value=ast.List(elts=[], ctx=ast.Load()),
            )
        )
        new_expr_sequence += self._visit_ListComp(
            listcomp_var, node.elt, node.generators
        )
        new_expr_sequence.append(ast.Name(id=listcomp_var, ctx=ast.Load()))
        return new_expr_sequence

    def _visit_ListComp(
        self, listcomp_var: str, elt: ast.expr, generators: List[ast.comprehension]
    ) -> Any:
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
                            body=self._visit_ListComp(
                                listcomp_var, elt, generators[1:]
                            ),
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
                value=ast.Call(
                    args=[], func=ast.Name(id="set", ctx=ast.Load()), keywords=[]
                ),
            )
        )
        new_expr_sequence += self._visit_SetComp(setcomp_var, node.elt, node.generators)
        new_expr_sequence.append(ast.Name(id=setcomp_var, ctx=ast.Load()))
        return new_expr_sequence

    def _visit_SetComp(
        self, setcomp_var: str, elt: ast.expr, generators: List[ast.comprehension]
    ):
        if not generators:
            new_expr_sequence = []
            tmp_call = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=setcomp_var, ctx=ast.Load()),
                    attr="add",
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
                value=ast.Dict(keys=[], values=[]),
            )
        )
        new_expr_sequence += self._visit_DictComp(
            dictcomp_var, node.key, node.value, node.generators
        )
        new_expr_sequence.append(ast.Name(id=dictcomp_var, ctx=ast.Load()))
        return new_expr_sequence

    def _visit_DictComp(
        self, dictcomp_var: str, key: ast.expr, value: ast.expr, generators
    ):
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
                            body=self._visit_DictComp(
                                dictcomp_var, key, value, generators[1:]
                            ),
                            orelse=[],
                        )
                    ]
                    if generators[0].ifs
                    else self._visit_DictComp(dictcomp_var, key, value, generators[1:]),
                    orelse=[],
                )
            ]

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> Any:
        new_expr_sequence = []
        generator_var = randoms.RandomGeneratorName.gen_generator_name()
        new_expr_sequence.append(
            ast.FunctionDef(
                name=generator_var,
                args=ast.arguments(
                    args=[],
                    vararg=None,
                    kwonlyargs=[],
                    kw_defaults=[],
                    kwarg=None,
                    defaults=[],
                ),
                body=self._visit_GeneratorExp(node.elt, node.generators),
                decorator_list=[],
                returns=None,
            )
        )
        new_expr_sequence.append(
            ast.Call(
                func=ast.Name(id=generator_var, ctx=ast.Load()), args=[], keywords=[]
            )
        )
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
        if not isinstance(expr, BASIC_TYPES):
            tmp_name = self.decompose_expr(expr, new_expr_sequence)
            node.value = tmp_name

        return new_expr_sequence + [node]

    def visit_Yield(self, node: ast.Yield) -> Any:
        if node.value is None:
            return [node]

        new_expr_sequence = []

        expr = node.value
        if not isinstance(expr, BASIC_TYPES):
            tmp_name = self.decompose_expr(expr, new_expr_sequence)
            node.value = tmp_name

        return new_expr_sequence + [node]

    def visit_YieldFrom(self, node: ast.YieldFrom) -> Any:
        new_expr_sequence = []

        expr = node.value
        if not isinstance(expr, BASIC_TYPES):
            tmp_name = self.decompose_expr(expr, new_expr_sequence)
            node.value = tmp_name

        return new_expr_sequence + [node]

    def visit_Compare(self, node: ast.Compare) -> Any:
        new_expr_sequence = []
        new_arg_list = []
        for expr in [node.left] + node.comparators:
            if not isinstance(expr, BASIC_TYPES):
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
            tmp_call = ast.Call(args=node.args, func=tmp_name, keywords=[])
            return new_expr_sequence + [tmp_call]

        new_arg_list = []
        for expr in node.args:
            if not isinstance(expr, BASIC_TYPES):
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

        if not isinstance(expr, BASIC_TYPES):
            tmp_name = self.decompose_expr(expr, new_expr_sequence)
            node.value = tmp_name

        return new_expr_sequence + [node]

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        new_expr_sequence = []
        expr = node.value

        if not isinstance(expr, BASIC_TYPES):
            tmp_name = self.decompose_expr(expr, new_expr_sequence)
            node.value = tmp_name

        return new_expr_sequence + [node]

    def visit_Starred(self, node: ast.Starred) -> Any:
        new_expr_sequence = []
        expr = node.value

        if not isinstance(expr, BASIC_TYPES):
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
