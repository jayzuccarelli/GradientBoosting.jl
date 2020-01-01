using CSV
using DataFrames
using DecisionTree
using Distributions
using Gurobi
using JuMP
using MLBase
using Plots
using Random
using StatsBase
using StatsPlots
include("../src/GradientBoosting.jl")


data = CSV.read("test/forestfires.csv", header = false)
ys = convert(Matrix, data)[:,end];
Xs = convert(Matrix, data)[:,1:end-1]

shuffleray = hcat(ys,Xs) #combine into single matrix
shuffleray = shuffleray[shuffle(1:end), :]
y_shuffled = shuffleray[:,1] #seperate shuffled matrix into two again
X_shuffled = shuffleray[:,2:end]

y_trn = y_shuffled[1:388] #about 75%
y_tst = y_shuffled[389:end]

X_trn = X_shuffled[1:388,:]
X_tst = X_shuffled[389:end,:]

gb_is_prediction, gb_models = GradientBoosting.fit(y_trn, X_trn, 0.005, 1, 25) #in-sample
gb_os_prediction = GradientBoosting.predict(y_tst, X_tst, 0.005, gb_models) #out-of-sample

print(1/length(y_tst)*sum((y_tst[i]-gb_os_prediction[i])^2 for i = 1:length(y_tst)),)