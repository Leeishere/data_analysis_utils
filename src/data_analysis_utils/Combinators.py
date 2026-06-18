"""Exact factorial, permutation, and combination helpers."""

from math import comb, factorial, perm
from numbers import Integral, Real


def _as_nonnegative_integer(value: int | float, parameter_name: str) -> int:
    """Return *value* as an int when it represents a nonnegative integer."""
    if isinstance(value, bool):
        raise TypeError(f"{parameter_name} must be an integer, not bool.")

    if isinstance(value, Integral):
        integer_value = int(value)
    elif isinstance(value, Real) and float(value).is_integer():
        integer_value = int(value)
    else:
        raise TypeError(f"{parameter_name} must be an integer.")

    if integer_value < 0:
        raise ValueError(f"{parameter_name} must be nonnegative.")

    return integer_value


def calculate_factorial(number: int | float) -> int:
    """Return the exact factorial of a nonnegative integer."""
    number = _as_nonnegative_integer(number, "number")
    return factorial(number)


def calculate_num_permutations(
    total_population: int | float,
    selection_size: int | float,
) -> int:
    """Return the exact number of ordered selections of the requested size."""
    total_population = _as_nonnegative_integer(
        total_population,
        "total_population",
    )
    selection_size = _as_nonnegative_integer(selection_size, "selection_size")

    if selection_size > total_population:
        raise ValueError("Selection size can't be greater than population size.")

    return perm(total_population, selection_size)


def calculate_num_combinations(
    total_population: int | float,
    selection_size: int | float,
) -> int:
    """Return the exact number of unordered selections of the requested size."""
    total_population = _as_nonnegative_integer(
        total_population,
        "total_population",
    )
    selection_size = _as_nonnegative_integer(selection_size, "selection_size")

    if selection_size > total_population:
        raise ValueError("Selection size can't be greater than population size.")

    return comb(total_population, selection_size)
