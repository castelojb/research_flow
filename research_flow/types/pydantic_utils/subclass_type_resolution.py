from pydantic import ValidationError, BaseModel


def all_subclasses(cls):
    """
    Return a list of all subclasses of a given class.

    This function returns a list of all subclasses of the given class, including
    subclasses of subclasses. It does this by recursively calling itself on
    each subclass of the given class.

    Parameters
    ----------
    cls : type
        The class to get all subclasses of.

    Returns
    -------
    list
        A list of all subclasses of the given class.
    """
    return list(cls.__subclasses__()) + [
        s for c in cls.__subclasses__() for s in all_subclasses(c)
    ]


class ModelInstanceMeta(type):
    def __getitem__(cls, item):
        """
        Create a new ModelInstance class with the given base class.

        Parameters
        ----------
        item : type
            The base class to use for the new ModelInstance class.

        Returns
        -------
        type
            The new ModelInstance class.

        Raises
        ------
        ValueError
            If the given item is a tuple.
        """
        if isinstance(item, tuple):
            raise ValueError("ModelInstance takes only one subfield ")
        return type(
            "ModelInstance[" + item.__name__ + "]", (cls,), {"__BaseClass__": item}
        )


class ModelInstance(metaclass=ModelInstanceMeta):
    """
    The ModelInstance class represents a base class for model instances.

    This class provides a way to validate values against a hierarchy of subclasses.
    It uses a custom metaclass `ModelInstanceMeta` to generate the necessary metadata.

    The class has two methods:

    - `__get_validators__`: Returns the validators for the ModelInstance class.
    - `validate`: Validates the given value against the sub-classes of the BaseClass.

    Example usage:

    >>>class Animal(BaseModel):
    >>>    pass
    >>>class Cat(Animal):
    >>>    pass
    >>>class Dog(Animal):
    >>>    pass
    >>>class Zoo(BaseModel):
    >>>    animals: List[ModelInstance[Animal]]
    >>>instance = Zoo(animals = [Cat(), Dog()]) # pydantic validation: ok

    Attributes:

    - `__BaseClass__`: The base class for the ModelInstance.

    """

    __BaseClass__ = BaseModel

    @classmethod
    def __get_validators__(cls):
        """
        Returns the validators for the ModelInstance class.

        This is a class method that returns the validators for the ModelInstance class.
        The validators are used to validate the input data and convert it to the
        appropriate type.

        Yields
        ------
        callable
            A validator for the ModelInstance class.
        """
        yield cls.validate

    @classmethod
    def validate(cls, value, p):
        """
        Validates the given value against the sub-classes of the BaseClass.

        The validation is done by iterating over all the sub-classes of the BaseClass
        and calling their validators. If any of the validators succeed, the value is
        returned. If all of them fail, a ValueError is raised with all the error
        messages.

        Parameters
        ----------
        value : any
            The value to validate.
        p : any
            The parent of the value to validate.

        Returns
        -------
        validated_value : any
            The validated value.

        Raises
        ------
        ValueError
            If the value is not valid.
        """
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
