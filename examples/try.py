try:
    a = 1
except Error1:
    a = 2
except:
    print("Unknown")
    raise
else:
    a = 3
finally:
    a = 4

a = 5
