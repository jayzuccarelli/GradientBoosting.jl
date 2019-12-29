function Gradient_Boosting(y_in, X_in, lr, max_depth, number_of_trees)

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

    # Plot the Data
    #Plots.scatter(X,y, label="Data")
    
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
        
        # Plot Model Fit of Some Iterations
        #if i == iters #i == 1 || i == iters/2 || i == iters
        #    plot!(X,predf, label="Model Fit (Iterations="*string(i)*")")
        #end
    end
    #plot!()
    return predf, finalmodels, predictions

end
