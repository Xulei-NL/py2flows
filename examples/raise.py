try:
    if val == 0:
        raise ValueError("Test")
except ValueError:
    print("value error")
finally:
    print("over")
