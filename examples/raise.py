try:
    if val == 0:
        raise ValueError("Test")
        val = 2
    val = 1
except ValueError:
    print("value error")
    raise
finally:
    print("over")
