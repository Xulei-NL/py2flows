a = lambda x: x + 1
a(1)
b = lambda x: 2 * x + 5 if x > 10 else 10 if x == 10 else 3 * x
lambda x: x + 1


def test():
    c = lambda x: x + 1
    c(1)


val = lambda: 1
a = (lambda: 1)()
(lambda: 1)()
