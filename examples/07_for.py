sum = 0

for val in [x + 1, y - 1, z * 1]:
    sum = sum + val
    if sum == 10:
        continue
    c += 1
else:
    sum = 3
    sum += 1

print("The sum is", sum)
