class RandomVariableName:
    counter = 0
    prefix = '_tmp_var_'

    @classmethod
    def gen_random_name(cls) -> str:
        cls.counter += 1
        return cls.prefix + str(cls.counter)
