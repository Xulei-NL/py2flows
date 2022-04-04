class RandomVariableName:
    counter = 0
    prefix = '_tmp_var_'

    @classmethod
    def gen_random_name(cls) -> str:
        cls.counter += 1
        return cls.prefix + str(cls.counter)


class RandomGeneratorName:
    counter = 0
    prefix = '_gen_exp_'

    @classmethod
    def gen_generator_name(cls) -> str:
        cls.counter += 1
        return cls.prefix + str(cls.counter)


class RandomLambdaName:
    counter = 0
    prefix = '_lambda_exp_'

    @classmethod
    def gen_lambda_name(cls) -> str:
        cls.counter += 1
        return cls.prefix + str(cls.counter)


class RandomUnusedName:
    prefix = '_tmp_unused_var_'

    @classmethod
    def gen_unused_name(cls) -> str:
        return cls.prefix


class RandomIterable:
    counter = 0
    prefix = '_tmp_iter_'

    @classmethod
    def gen_iter(cls) -> str:
        return cls.prefix + str(cls.counter)
