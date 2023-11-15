from typing import Callable

def generate (f: Callable, bounds: list[tuple[float, float]], n_features: int, n_samples: int, noise: float=5, ordered: bool=False, random_state: int=None) -> tuple:
    '''Takes the function you want the data to follow, the lower
    and upper bounds of the features, the number
    of features, number of samples, noise, random_state
    anc constructs a dataset that follows that function\n
    `oredered`: whether the (xs, ys) should be ordered or shuffeled'''
    
    assert n_samples > 0, f"Number of samples must be positive"
    assert n_features > 0, f"Number of features must be positive"
    assert len(bounds) == n_features, f"Incoherent number of features and bounds"
    
    import numpy as np
    from math import ceil
    
    rng = np.random.RandomState(random_state)
    n_samples_per_feature = ceil(n_samples ** (1/n_features))
    
    
    # Getting the deltas
    deltas = []
    lowers = []
    for bound in bounds:
        lower, upper = bound
        lowers.append(lower)
        deltas.append((upper - lower)/(n_samples_per_feature +1)) # +1 to have space between
    
    # Creating all the values that each feature will have
    features_values = []
    for lower, delta in zip(lowers, deltas):
        feature_values = []
        for _ in range(n_samples_per_feature):
            lower += delta
            feature_values.append(lower)
        features_values.append(feature_values)
    
    '''I want them to be incremented gradually to be able to returna ctual number of requested samples by removing from edges and thus only small / large values will be removed and not middle ones
    but since no brain power rn i will just do this and return extra number of samples'''
    # Putting the values in xs
    xs = []
    js = [0] * n_features
    while True:
        x = []
        for feature_values, j in zip(features_values, js): # The current value pointed to by j
            x.append(feature_values[j])
        xs.append(x)
        
        for i in range(n_features): # Increment and carry
            js[i] += 1
            if js[i] < n_samples_per_feature:
                break
            else:
                js[i] = 0
        
        if sum(js) == 0: # Can only be zero when all the combinations have been travesed
            break
    
    # Removing excess, hah more like removing exes
    # FIXME make it remove from head and tail in semi-random way where it is more likely to pick from head / tail
    # after you make them ordered that is
    actual_n_samples = len(xs)
    assert actual_n_samples == n_samples_per_feature ** n_features, f"Unreachable" # ** and not * because its combinations
    while len(xs) > n_samples:
        i = rng.randint(0, len(xs) -1)
        xs.pop(i)
        
    if not ordered:
        rng.shuffle(xs)
        
        
    # Getting the ys
    ys = []
    for x in xs:
        ys.append(f(*x))
        
    # FIXME make noise a deviasion or idk, point is should be relative to the values not actual deltas
    half_noise = noise/2
    for x in xs:
        for i, xi in enumerate(x):
            x[i] = rng.uniform(xi -half_noise, xi +half_noise)
    
    for i, y in enumerate(ys):
        ys[i] = rng.uniform(y -half_noise, y +half_noise)

    # FIXME use np arrays
    return (xs, ys)