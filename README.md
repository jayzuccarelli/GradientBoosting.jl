# GradientBoosting.jl

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE.md)
[![Julia](https://img.shields.io/badge/Julia-1.1.1+-9558B2.svg)](https://julialang.org)

A Julia implementation of [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_Boosting) — an ensemble learning method that builds a strong predictive model by sequentially fitting decision trees to the residuals of the previous iteration.

This package is an [official entry](https://github.com/JuliaRegistries/General/tree/master/G/GradientBoosting) in the Julia General Registry and can be installed via [Pkg.jl](https://julialang.github.io/Pkg.jl/v1/).

## Installation

```julia
Pkg.add("GradientBoosting")
```

## Usage

```julia
using GradientBoosting

# Train the model
predictions, models, history = GradientBoosting.fit(y_train, X_train, lr, max_depth, n_trees)

# Predict on new data
test_predictions = GradientBoosting.predict(y_test, X_test, lr, models)
```

### Parameters

| Parameter    | Description                                          |
|--------------|------------------------------------------------------|
| `y`          | Target variable (vector)                             |
| `X`          | Feature matrix                                       |
| `lr`         | Learning rate — controls the contribution of each tree |
| `max_depth`  | Maximum depth of each decision tree                  |
| `n_trees`    | Number of boosting iterations (trees)                |

## License

[MIT](./LICENSE.md)
