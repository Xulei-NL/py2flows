l = [[1, 2], [3, 4], [5, 6]]
a = {x for y in l if len(y) == 2 for x in y if x > 1}
print("hello world")
z = {1, 2, 3, 4, 5}


def num2str(num):
    return str(num)


a = {num2str(num) for x in z}
