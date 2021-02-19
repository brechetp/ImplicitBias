using PyPlot,
using Random, ProgressMeter
#using Plots


Base.@kwdef mutable struct TwoLayer
    d::Int  # input dim
    m::Int  # output dim
    W::Array{Float64, 2}  # the input and output weighst
    TwoLayer(d::Int, m::Int) = new(d, m, randn(m, d+1))
end

"""
Initialise a model
"""
function init!(l::TwoLayer, alpha1::Real=1.0, alpha2::Real=1.0, sym::Bool=false)
    d, m = l.d, l.m
    # input weights initialized uniformly on the unit sphere
    l.W[:,1:d] ./= sqrt.(sum(l.W[:,1:d].^2, dims=2))
    l.W[:, 1:d] .*= alpha1
    # output weights initialized by -1 or 1 with equal proba
    l.W[1:div(m,2),end] .= alpha2
    l.W[div(m,2)+1:end,end] .= -alpha2
    if sym
        @assert iseven(m)
        W[div(m,2)+1:end,1:d]  .=  W[1:div(m,2),1:d]
    end
end



##
#
"""
Train 2-layer relu NN with GD on square loss
INPUT
 m is the number of neurons, 
 alpha1 (resp. alpha2): scale of the initialization of the input (resp. output) weights
 ouly_output: train just the output layer (instead of both)
 sym_init: trick to make the output = 0 exactly at initialization
OUTPUT
 Ws: the whole training path
 loss: evolution of loss during training
"""
function twonet_square(model::Layer, X, Y, m, stepsize, niter;alpha1=1.0, alpha2=1.0, weight_decay=0.0, sym_init=false, only_output = false)
    (n,d) = size(X) # n samples in R^d
    # initialize
    #
    # input weights initialized uniformly on the unit sphere
    init!(model)

    Ws    = zeros(m, d+1, niter) # store optimization path
    loss  = zeros(niter) # loss is -log of the empirical risk

    @showprogress 1 "Training neural network..." for iter = 1:niter
        Ws[:,:,iter] = W
        act  =  max.( W[:,1:end-1] * X', 0.0) # activations
        out  =  (1/m) * sum( W[:,end] .* act , dims=1)[:] # predictions of the network
        gradR   = (out .- Y)/n  # size n
        grad_w1 = (W[:,end] .* float.(act .> 0) * ( X .* gradR  )) + weight_decay * W[:,1:end-1] # gradient for input weights
        grad_w2 = act * gradR + weight_decay * W[:,end]  # gradient for output weights
        if only_output 
            grad = cat(zeros(m,d), grad_w2, dims=2) # size (m × d+1)
        else
            grad = cat(grad_w1, grad_w2, dims=2) # size (m × d+1)
        end
        loss[iter] = (1/2)*sum( ( out - Y).^2 )/n + (weight_decay/2)*sum(W.^2)/m
        W = W - stepsize * grad
    end

    return Ws, loss
end

##

Random.seed!(5)
d=3  # dim(X) + bias
n = 10
m = 1000
X = cat(2*(rand(n,2).-0.5), ones(n), dims=2)
Y = rand([-1,1],n)
stepsize = 0.1
niter = 20000
Ws, loss = twonet_square(X, Y, m, stepsize, niter, weight_decay=5e-5, only_output =false)

loglog(loss,label="both")

stepsize = 8
niter = 40000
Ws2, loss2 = twonet_square(X, Y, m, stepsize, niter, weight_decay=5e-6, only_output =true)

loglog!(loss2)

#legend()

colm = :magma
# cm = colormap(colm)

#figure(figsize=[8,4])
f(x1,x2) = (1/m) * sum( Ws[:,end,end] .* max.( Ws[:,1:3,end] * [x1;x2;1], 0.0)) # prediction function
f2(x1,x2) = (1/m) * sum( Ws2[:,end,end] .* max.( Ws2[:,1:3,end] * [x1;x2;1], 0.0)) # prediction function
reso = 100 
xs = range(-1.0,1.0,length=reso)
tab = [f(xs[i],xs[j]) for i=eachindex(xs), j=eachindex(xs)]
tab2 = [f2(xs[i],xs[j]) for i=eachindex(xs), j=eachindex(xs)]

# contour(xs, xs, f, fill=true, c=:magma)

#figure(figsize=[6.75,3])
#subplot(121)
#title("Training both layers")

#pcolormesh(xs', xs, tab', cmap=colm, shading="gouraud",vmin=-4.5,vmax = 1.5,edgecolor="face")

# cm = plt.get_cmap(colm)

scatter!(X[:,1],X[:,2],50,c=cm.(Y/6 .+0.7),edgecolors="k")

#axis("equal")
xticks([-1,0,1]);xlabel(L"(a)")
yticks([-1,0,1])#;ylabel(L"y")
subplot(122)
pcolormesh(xs', xs, tab2', cmap=colm, shading="gouraud",vmin=-4.5,vmax = 1.5,edgecolor="face")
cm = plt.get_cmap(colm)
scatter(X[:,1],X[:,2],50,c=cm.(Y/6 .+0.7),edgecolors="k")
#axis("equal")
#title("Training output layer")
xticks([-1,0,1]);xlabel(L"(b)")
yticks([-1,0,1])#;ylabel(L"y")
#savefig("regularized.png",dpi=150,bbox_inches="tight")
###
