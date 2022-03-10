from .cfg import comments, flows
import sys
import ast
import logging


def main():
    if len(sys.argv) == 1:
        logging.error('You have to provide a file. One example is py2flows ./examples/test.py')
        sys.exit(1)
    file_name = sys.argv[1]
    file = open(file_name, "r")
    source = file.read()
    file.close()

    comments_cleaner = comments.CommentsCleaner(source)
    comments_cleaner.remove_comments_and_docstrings()
    comments_cleaner.format_code()
    logging.debug(comments_cleaner.source)

    cfg = flows.CFGVisitor().build(file_name, ast.parse(comments_cleaner.source))
    logging.debug('flows: %s', sorted(cfg.flows))
    logging.debug('edges: %s', sorted(cfg.edges.keys()))
    cfg.show()


if __name__ == '__main__':
    main()
