sum = 0

for val in numbers:
    sum = sum + val
    if sum == 10:
        continue
    c += 1
else:
    sum = 3
    sum += 1

print("The sum is", sum)
