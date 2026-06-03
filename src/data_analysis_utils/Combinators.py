


from math import gamma, factorial



def calculate_factorial(number:int|float):
    """
    where number can be in >=0 or float >=2
    this returns float(factorial)
    """
    if number<0:
        raise ValueError("Can't compute factorials for negative numbers")
    elif number<=1.000001:
        return 1
    elif number > 171:
        number=int(round(number))
        return int(factorial(number))
    else:
        return int(round(gamma(number-1),0))

def calculate_num_permutations(total_population, selection_size):
    """
    """
    #Permutation(n,r) == Permutation(total_population,selection_size) == total_population!/(total_population-selection_size)!
    factorial_total_population=calculate_factorial(total_population)
    total_population_minus_selection_size=total_population - selection_size
    if total_population_minus_selection_size<0:
        raise ValueError("Selection size can't be greater than population size.")
    factorial_total_population_minus_selection_size=calculate_factorial(total_population_minus_selection_size)
    return int(round(factorial_total_population / factorial_total_population_minus_selection_size,0))

def calculate_num_combinations(total_population, selection_size):
    """
    """
    #Combination(n,r) == Combination(total_population,selection_size) == total_population!/(selection_size!(total_population-selection_size)!)
    factorial_total_population=calculate_factorial(total_population)
    factorial_selection_size=calculate_factorial(selection_size)
    total_population_minus_selection_size=total_population - selection_size
    if total_population_minus_selection_size<0:
        raise ValueError("Selection size can't be greater than population size.")
    factorial_total_population_minus_selection_size=calculate_factorial(total_population_minus_selection_size)
    return int(round(factorial_total_population / ( factorial_selection_size * factorial_total_population_minus_selection_size ),0))