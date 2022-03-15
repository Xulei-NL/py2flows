from .cfg import comments, flows
import os.path
import ast
import logging
import argparse

logging.basicConfig(level=logging.DEBUG)


def main():
    parser = argparse.ArgumentParser(description='compute flows of control flow graphs. '
                                                 'But of course you can use it to examine cfgs only')
    parser.add_argument('file_name', help='path to the Python file')
    parser.add_argument('-iso', '--isolation',
                        help='If specified, each function will have isolated entries and exits',
                        action='store_true')
    args = parser.parse_args()
    logging.debug(args.file_name)
    logging.debug(args.isolation)

    file = open(args.file_name, "r")
    source = file.read()
    file.close()

    comments_cleaner = comments.CommentsCleaner(source)
    comments_cleaner.remove_comments_and_docstrings()
    comments_cleaner.format_code()
    logging.debug(comments_cleaner.source)

    visitor = flows.CFGVisitor(args.isolation)
    base_name = os.path.basename(args.file_name)
    cfg = visitor.build(base_name, ast.parse(comments_cleaner.source))
    logging.debug('flows: %s', sorted(cfg.flows))
    logging.debug('edges: %s', sorted(cfg.edges.keys()))
    logging.debug('Current Label: %d', visitor.curr_block.bid)
    if visitor.isolation:
        visitor.add_stmt(visitor.curr_block, ast.Pass())
    visitor.remove_empty_blocks(cfg.start)
    cfg.show()


if __name__ == '__main__':
    main()
