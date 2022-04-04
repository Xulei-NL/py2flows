from .cfg import comments, flows
import os.path
import ast
import logging
import argparse


def construct_CFG(file_name) -> flows.CFG:
    with open(file_name) as handler:
        source = handler.read()
        comments_cleaner = comments.CommentsCleaner(source)
        visitor = flows.CFGVisitor(isolation=True)
        base_name = os.path.basename(file_name)
        cfg = visitor.build(base_name, ast.parse(comments_cleaner.source))
        logging.debug(visitor.cfg.flows)
        logging.debug(visitor.cfg.blocks)
        cfg.show()

        return cfg


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(description='compute flows of control flow graphs. '
                                                 'But of course you can use it to examine cfgs only')
    parser.add_argument('file_name', help='path to the Python file')
    parser.add_argument('-iso', '--isolation',
                        help='If specified, each function will have isolated entries and exits',
                        action='store_true')
    parser.add_argument('--asserts', help='Desugar asserts', action='store_true')
    parser.add_argument('--fors', help='Desugar fors', action='store_true')
    parser.add_argument('-f', '--format', default='png',
                        help='Specify the format of output graph. Basically three formats: png(default), svg and pdf')
    parser.add_argument('-p', '--path', default='./',
                        help='Specify the path of output graph. The default is current directory')
    parser.add_argument('-n', '--name', default='output',
                        help='Specify the name of the output file. The default is output')
    args = parser.parse_args()
    logging.debug(args.file_name)
    logging.debug('{} {} {}'.format(args.isolation, args.asserts, args.fors))

    file = open(args.file_name, "r", encoding='utf-8')
    source = file.read()
    file.close()

    comments_cleaner = comments.CommentsCleaner(source)
    logging.debug(comments_cleaner.source)

    visitor = flows.CFGVisitor(args.isolation, args.asserts, args.fors)
    base_name = os.path.basename(args.file_name)
    cfg = visitor.build(base_name, ast.parse(comments_cleaner.source))
    logging.debug('Refactored edges: %s', sorted(cfg.edges.keys()))
    logging.debug('Refactored flows: %s', visitor.cfg.flows)
    logging.debug('Refactored labels: %s', visitor.cfg.labels)
    cfg.show(fmt=args.format, filepath=args.path + '/' + args.name, name=base_name)


if __name__ == '__main__':
    main()
