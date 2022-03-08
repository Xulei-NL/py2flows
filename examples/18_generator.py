arr = [1, 2, 3]
arr2 = [[1, 2], [3, 4]]
generator = (num for num in arr)
generator2 = (num for single in arr2 for num in single)
