# GradientBoosting
[![GitHub version](https://badge.fury.io/gh/ezuccarelli%2FGradientBoosting.jl.svg)](https://badge.fury.io/gh/ezuccarelli%2FGradientBoosting.jl)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE.md)
The Julia implementation of <a href=https://en.wikipedia.org/wiki/Gradient_Boosting>Gradient Boosting</a>.
The package is now an <a href="https://github.com/JuliaRegistries/General/tree/master/G/GradientBoosting">official entry</a> in the Julia Registry and can be installed using Julia's default package manager <a href="https://julialang.github.io/Pkg.jl/v1/">Pkg.jl</a>.</p>

## Installation
```julia
Pkg.add("GradientBoosting")
```

## Importing the Library
```julia
using GradientBoosting
```
## Using the Package
```julia
# Train the model
train_predictions, gb_models = GradientBoosting.fit(y_trn, X_trn, lr, max_depth, number_of_trees)

# Predict on test data
test_predictions = GradientBoosting.predict(y_tst, X_tst, lr, gb_models)
```
