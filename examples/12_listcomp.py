z = [[1, 2, 3], [4, 5, 6]]
a = [x for x in [y for y in z]]
b = [2 * x for y in z if len(y) > 1 for x in y if x > 2 if x < 4]
