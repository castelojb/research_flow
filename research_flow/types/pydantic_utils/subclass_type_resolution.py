from pydantic import ValidationError, BaseModel


def all_subclasses(cls):
    return list(cls.__subclasses__()) + [
        s for c in cls.__subclasses__() for s in all_subclasses(c)
    ]


class ModelInstanceMeta(type):
    def __getitem__(cls, item):
        if isinstance(item, tuple):
            raise ValueError("ModelInstance takes only one subfield ")
        return type(
            "ModelInstance[" + item.__name__ + "]", (cls,), {"__BaseClass__": item}
        )


class ModelInstance(metaclass=ModelInstanceMeta):
    __BaseClass__ = BaseModel

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value, p):
        if isinstance(value, cls.__BaseClass__):
            return value

        errors = []

        for SubClass in all_subclasses(cls.__BaseClass__)[::-1]:
            for validator in SubClass.__get_validators__():
                try:
                    return validator(value)
                except (ValidationError, ValueError, AttributeError, KeyError) as err:
                    errors.append(err)
        ####
        if errors:
            raise ValueError("\n".split(errors))

        else:
            raise ValueError("cannot find a valid subclass")
