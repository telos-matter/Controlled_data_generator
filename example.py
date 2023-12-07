'''
Suppose We want to create data for a regression problem in which
we want to predict students' scores based on their latest score and how many
hours they have studied over the past week before the test. Then
we could do something like so 
'''

from controlledDataGenerator import generate

# The function that describes their scores based on the two features
f = lambda x0, x1: 0.85*x0 + 0.1*x1 +10

# The intervals that the features should lay in
intervals = [
    (0, 100), # Their previous test marks are from 0 to a 100
    (0, 20)]  # And they could have studied from 0 to 20 hours during the previous week

# It does not make sense for the score to be outside the 0 - 100 range so
y_interval = (0, 100) 

x, y = generate(f, intervals, n_samples=20, noise=7.5, y_interval=y_interval)

print(x)
print(y)
