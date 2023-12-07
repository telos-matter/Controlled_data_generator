# HEMMOUDA Aymane, ahemmouda@gmail.com
# DEC 2023

if __name__ == '__main__':
    raise Exception("This file is not expected to be ran as the main file, rather it should be imported")

from typing import Callable

import numpy as np
from numbers import Number
from numpy.random import Generator
from inspect import signature

def generate (f: Callable[..., Number], intervals: list[tuple[Number, Number]], *, n_samples: int=100, noise: float=0.0, generator: Generator | int=None, y_interval: None | tuple[Number | None, Number | None]=None, n_features: None | int=None) -> tuple[np.array, np.array]:
    '''Generates a random yet controlled dataset for your regression
    problems. That is, your dataset will follow a function you define
    
    ### Required parameters:
    - `f`: a callable that represents 
    the function that your dataset should follow or be mapped by, for example
    if you want your data to follow the line `y = 2x + 3` then the callable
    should be something similar to this `lambda x: 2*x +3`. Of course the number
    of parameters that the callable takes should be the same as the number of features.
    - `intervals`: a list of tuples containing the lower and upper bounds of the
    half-open intervals (that is [lower, upper), meaning the lower value is included
    meanwhile the upper one is not) that the features should lay in. The number
    of features of your dataset is drawn from the length of the list.
    
    ### Optional parameters:
    - `n_samples`: the number of data points, a.k.a samples of the dataset
    - `noise`: the standard deviation of the gaussian noise
    that is added to the output after applying the callable f. Must be
    greater than or equal to zero.
    - `generator`: a numpy Generator (the newer version of RandomState)
    to generate the data points with. Or an int representing the seed
    to create the generator with. Or None to create a new random one.
    - `y_interval`: if not None then it should be a tuple 
    that determines the half open interval [lower, upper) (lower
    is included, upper is not) that the output data should lay in. If a y
    value is not within the interval it is removed as well as its
    corresponding x value and they are not substituted, which
    means that the actual number of samples
    can be less than or equal to n_samples. If a tuple is provided
    then either lower or upper can be None but not both.
    - `n_features`: an optional assertion parameter that could
    be passed if the user wants to assert that the number of
    features of the dataset is that he expect. Would
    raise an AssertionError if it's not None and is different
    than the length of the list of intervals
    
    ### Returns:
    A tuple of two numpy array, representing the Xs (features) and the Y (target) respectively
    
    ---
    
    #### [Github repo](https://github.com/telos-matter/Controlled_data_generator)
    '''
    
    def check(condition: bool, exception_type: type, message: str) -> None:
        '''Checks the condition and raises an exception if it's false'''
        if not condition:
            raise exception_type(message)
    
    check(f is not None and intervals is not None, TypeError, f"Neither the callable f nor the intervals can be None\nf: {f}\nintervals: {intervals}")
    check(callable(f), TypeError, f"`f` should be a callable function. {f} was passed instead, which cannot be called")
    check(hasattr(intervals, '__len__'), TypeError, f"Intervals should be a list like object. {intervals} was passed")
    check(len(intervals) == len(signature(f).parameters), ValueError, f"The number of parameters that the callable f should have, should be same as the number of features, which is {len(intervals)}, yet the callable provided has {len(signature(f).parameters)}")
    check(type(n_samples) == int, TypeError, f"n_samples should be of type int")
    check(n_samples > 0, ValueError, f"The number of samples should at least be one. {n_samples} was passed")
    check(isinstance(noise, Number), TypeError, f"The noise should a number")
    check(noise >= 0.0, ValueError, f"The noise should be greater than or equal to zero. {noise} was passed")
    check(generator is None or type(generator) == int or type(generator) == Generator, TypeError, f"The generator must be an int, an instance of the numpy Generator or None. {generator} was passed")
    check(y_interval is None or type(y_interval) == tuple, TypeError, f"The y_interval should be a tuple, not `{type(y_interval)}`")
    check(y_interval is None or len(y_interval) == 2, ValueError, f"The y_interval should contain exactly two elements, this one has {y_interval}")
    check(y_interval is None or not (y_interval[0] is None and y_interval[1] is None), ValueError, f"Either the lower or upper bounds of the y_interval can be None but not both; {y_interval}")
    check(n_features is None or n_features == len(intervals), AssertionError, f"The length of the intervals list (which is the number of features) is different than the given n_features. Intervals entails that there is {len(intervals)} features, yet n_features is {n_features}")
    
    n_features = len(intervals)
    if type(generator) != Generator:
        generator = np.random.default_rng(seed=generator)
    
    xs = np.empty(shape=[n_features, n_samples]) # Later on it gets flipped back
    for i, interval in enumerate(intervals):
        check(hasattr(interval, '__len__'), TypeError, f"This interval {interval}, with index {i}, cannot be used")
        check(len(interval) == 2, ValueError, f"This interval {interval}, with index {i}, does not have exactly 2 values, the lower and upper bound")
        lower, upper = interval
        check(isinstance(lower, Number) and isinstance(upper, Number), TypeError, f"This interval {interval}, with index {i}, cannot be used as it has something other than Numbers as bounds") # Too much checking? I guess
        xs[i] = generator.uniform(lower, upper, n_samples)
    
    xs = xs.T # Flip it to the right shape
    
    y = np.empty(shape=[n_samples])
    for i, x in enumerate(xs):
        y[i] = f(*x)
    
    if noise > 0.0:
        y += generator.normal(loc=0.0, scale=noise, size=n_samples)
    
    if y_interval is not None:
        interval_checker = None
        lower, upper = y_interval
        if lower is None: # Only check < upper
            assert upper is not None, f"Unreachable, checked at entry"
            interval_checker = lambda value: value < upper
        elif upper is None: # Only check >= lower
            interval_checker = lambda value: lower <= value
        else: # Check both
            interval_checker = lambda value: lower <= value < upper
        
        indices = [i for i in range(n_samples) if not interval_checker(y[i])]
        xs = np.delete(xs, indices, 0)
        y = np.delete(y, indices)
    
    return (xs, y)
