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

import ast
import os
import sys

import autopep8

from .flows import flows


def construct_CFG(file_path) -> flows.CFG:
    with open(file_path) as handler:
        source = autopep8.fix_code(handler.read())
        visitor = flows.CFGVisitor()
        base_name = os.path.basename(file_path)
        cfg = visitor.build(base_name, ast.parse(source))
        if sys.open_graph:
            left_base_name = base_name.partition(".")[0]
            cfg.show(name=left_base_name)

        return cfg
