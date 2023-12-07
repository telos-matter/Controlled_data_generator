# Controlled data generator ![DEVELOPMENT STATUS: working version](https://badgen.net/badge/DEVELOPMENT%20STATUS/working%20version/green)

Generates random yet controlled data for your regression problems.\
Unlike the [make_regression](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html) function from the `sklearn` library, this function allows you to define the intervals in which your features should lay as well as how the target should be computed from the features.

Check out the [example.py](example.py) file for a quick example. Or read the documentation of the function using:
```console
$ python3
> from controlledDataGenerator import generate
> help(generate)
```
*Note: the file should of course be in the same directory you run your python instance in*

## Requirements:
- Latest `numpy` version