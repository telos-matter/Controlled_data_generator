# TODO do put this in github
def make_regression (f: Callable, bounds: list[tuple[float, float]], n_features: int, n_samples: int, noise: float=5, ordered: bool=False, random_state: int=42) -> tuple:
    '''Takes the function you want the data to follow, the lower
    and upper bounds of the features, the number
    of features, number of samples, noise, random_state
    anc constructs a dataset that follows that function\n
    `oredered`: whether the (xs, ys) should be ordered or shuffeled'''
    
    assert n_samples > 0, f"Number of samples must be positive"
    assert n_features > 0, f"Number of features must be positive"
    assert len(bounds) == n_features, f"Incoherent number of features and bounds"
    
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
    
#     # Making a combination out of all the samples
#     # One way of doing it would be similar to a digital clock, where one increases the next when it's done
#     # But I want a way that picks them randomly and picks all of them
#     # LFSR would work too
#     # But an easier one is to just generate the indexes and randomize the list lol and do the clock thing
#     features_indexes = []
#     for _ in range(n_features):
#         feature_indexes = list(range(n_samples_per_feature))
#         np.random.shuffle(feature_indexes)
#         features_indexes.append(feature_indexes)
#     xs = []
#     js = [0] * n_features
#     while True:
#         x = []
#         for feature_samples, feature_indexes, j in zip(samples, features_indexes, js): # Pick the sample pointed to by feature index which is pointed to by j
#             x.append(feature_samples[feature_indexes[j]])
#         xs.append(x)
        
#         for i in list(range(n_features))[:: -1]: # Increment and carry
#             js[i] += 1
#             if js[i] < n_samples_per_feature:
#                 break
#             else:
#                 js[i] = 0
        
#         if sum(js) == 0: # Can only be zero when they have travesed them all
#             break
            
    return xs

out = make_regression(None, [(5, 20), (50, 100)], 2, 7)
print(out)
# foo = [[1,2,3], [4, 5, 6]]
# foo = np.reshape(foo, (3, 2))
# print(type(foo))
# print(foo)