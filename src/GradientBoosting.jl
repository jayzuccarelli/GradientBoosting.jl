module GradientBoosting

using DecisionTree
using StatsBase

function random_subset_selection(X_imp, y_imp, k, index_rm)

    n = size(X_imp,1)
    p = size(X_imp,2)

    # Train arrays
    train_Xs = Array{Float64}(undef, 0, p)
    train_ys = Array{Float64}(undef, 0)

    # Randomize
    shuffleray = hcat(y_imp, X_imp, index_rm)
    shuffleray = shuffleray[shuffle(1:end), :]
    y_shuffled = shuffleray[:,1]
    X_shuffled = shuffleray[:,2:(end-1)]
    index_shuffled = shuffleray[:,end]

    train_ys = y_shuffled[1:k]
    train_Xs = X_shuffled[1:k,:]
    index_selected = index_shuffled[1:k]

    return train_Xs, train_ys, index_selected
end

function fit(y_in, X_in, lr, max_depth, number_of_trees)

    yi = deepcopy(y_in)
    Xi = deepcopy(X_in)
    
    # Parameters
    n_subfeatures = 0
    max_depth = max_depth
    min_samples_leaf = 5
    min_samples_split = 2
    min_purity_increase = 0.0
    pruning_purity = 1.0

    # Initialize Error
    ri = 0
    n = length(yi)
    
    # Initialize Predictions with Average
    predf = reshape(zeros(n), :, 1);
    predf .= mean(yi);
    
    # Number of Iterations
    iters = number_of_trees
    
    # Array for storing all trees in the final model
    finalmodels = Array{Union{Node{Float64,Float64},Leaf{Float64}}}(undef, iters,1)
    
    # Array for convergence plot
    predictions = zeros(iters,n)
    
    for i=1:iters
        # Fit the Decision Tree
        model = DecisionTree.build_tree(yi, Xi,
                       n_subfeatures,
                       max_depth,
                       min_samples_leaf,
                       min_samples_split,
                       min_purity_increase)
        
        #add model to final models
        finalmodels[i] = model

        # Predict yi
        predi = apply_tree(model, Xi)
        
        # Compute the New Prediction and the New Residuals
        # Set the New yi equal to the Residuals
        predf .+= lr*predi
        ri = y_in.-predf
        yi = vec(ri)
        
        #record predictions for convergence plot
        predictions[i,:] = predf

    end

    return predf, finalmodels, predictions

end

function predict(y_in, X_in, lr, arrayofmodels)
    yi = deepcopy(y_in)
    Xi = deepcopy(X_in)

    n = size(Xi, 1)

    # Initialize Predictions with Average
    predf = reshape(zeros(n), :, 1);
    predf .= mean(yi);

    for i = 1:size(arrayofmodels,1)
        model = arrayofmodels[i]
        predi = apply_tree(model, Xi)
        predf .+= lr*predi
    end
    return predf
end

export fit, predict

end