# Test if the program can deal with nested IfExp
def test():
    val = 1
    return (3 if 1 == 1 else 2) if val == 1 else (5 if 1 == 1 else 3)


# Test if the program can find all final blocks
def test1():
    val = 1
    if val == 1:
        return val
    else:
        return 3 if 1 == 1 else 2


b = 1 if 2 == 1 else 3
ret = test()
print(b)
