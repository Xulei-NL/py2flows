test = "a fake file path"

with open(test) as fptr:
    print("It's okay")

print("Over1")

async with open(test) as fptr:
    print("It's not okay")

print("Over2")
