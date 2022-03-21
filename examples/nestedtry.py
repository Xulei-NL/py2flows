try:
    try:
        print("inner try")
        raise
    except:
        print("inner except")
        raise
    # finally:
    #     print("inner finally")
except:
    print("outer except")
    raise
finally:
    print("outer except")
