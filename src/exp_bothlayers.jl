# copyright Lénaïc Chizat
# the code shows the training dynamics in parameter and predictor space for a 2-layer relu NN on a two class classification task
# we optimize the exponential loss (the behavior is sensibly the same with the logistic loss)
# we use a specific step-size schedule motivated by the theory in this ref https://arxiv.org/abs/2002.04486 (sections 4 & 5)

# HOW TO RUN THE CODE
# in a prompt first run:
# include("exp_bothlayers.jl")
# Then for fast prototyping, run for instance the following code (takes 1 min)
# illustration(4, 60, 200, 0.4, 100_000, 10, 0.05);
# These parameters are explained in the header of the function "illustration"
# A good illustration is obtained as follows (takes 20 min):
# Random.seed!(9); # 9 generates a nice random problem
# illustration(4, 60, 400, 0.4, 1000000, 400, 0.005);


##
using LinearAlgebra, Random
using PyPlot, ProgressMeter # these packages need to be installed (via "] add NamePackage" )
using Colors
using3D()

##

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

mutable struct Result
    niter::Int
    m::Int
    d::Int
    Ws::Array{Float64, 3}     # store optimization path
    loss::Vector{Float64}  # loss is -log of the empirical risk
    margins::Vector{Float64} 
    betas::Vector{Float64}

    Result(niter::Int, model::TwoLayer) = new(niter, model.m, model.d,
                                              zeros(model.m, model.d+1, niter),
                                              zeros(niter),
                                              zeros(niter),
                                              zeros(niter)
                                             )

end
##
"""
Gradient descent to train a 2-layers ReLU neural network for the exponential loss
We use the step-size schedule from in https://arxiv.org/abs/2002.04486 (sections 4 & 5)
INPUT: X (training input), Y (training output), m (nb neurons), both: training both layers or just the output layer
OUTPUT: Ws (training trajectory)
"""
function twonet(model::TwoLayer, X, Y, m, stepsize, niter)::Result

    (n,d) = size(X) # n samples in R^d
    # initialize
    result = Result(niter, model)

    @showprogress 1 "Training neural network..." for iter = 1:niter
        result.Ws[:,:,iter] = model.W
        act  =  max.( model.W[:,1:end-1] * X', 0.0) # activations
        out  =  (1/m) * sum( model.W[:,end] .* act , dims=1) # predictions of the network
        perf = Y .* out[:]
        margin = minimum(perf)
        temp = exp.(margin .- perf) # numerical stabilization of exp
        gradR = temp .* Y ./ sum(temp)' # gradient of the loss
        grad_w1 = (model.W[:,end] .* float.(act .> 0) * ( X .* gradR  ))  # gradient for input weights
        grad_w2 = act * gradR  # gradient for output weights

        grad = cat(grad_w1, grad_w2, dims=2) # size (m × d+1)
        result.betas[iter] = sum(model.W.^2)/m
        result.loss[iter] = margin - log(sum(exp.(margin .- perf))/n)
        result.margins[iter] = margin/result.betas[iter] # margin in F_1 norm
        model.W = model.W + stepsize * grad/(sqrt(iter+1))
    end

    return result
end

##

"Coordinates of the 2d cluster centers, p is k^2 the number of clusters"
function cluster_center(p,k, Delta)
    p1 = mod1.(p,k)
    p2 = fld1.(p, k)
    x1 =  Delta*3*(p1 .- 1) .- 1/2
    x2 =  Delta*3*(p2 .- 1) .- 1/2
    return x1, x2
end


mutable struct Data
    d::Int  # input dimension
    p::Int  # output dimension
    n::Int  # number of data points
    X::Array{Float64}
    Y::Array{Float64}
end

function Data(k::Int, n::Int)
    sd = 0 # number of spurious dimensions with random noise
    Delta = 1/(3k-1) # interclass distance
    A = ones(k^2) # cluster affectation
    A[randperm(k^2)[1:div(k^2,2)]] .= -1  # random -1 label

    # sample from this data distribution
    P = rand(1:k^2,n) # cluster label
    T = 2π*rand(n)  # shift angle
    R = Delta*rand(n) # shift magnitude
    X = #[ 2(rand(n, 2).-.5) ones(n) ]
    [ cluster_center(P,k, Delta)[1] + R .* cos.(T) cluster_center(P,k, Delta)[2] + R .* sin.(T) ones(n) ]
        # dims=2)  # 
    # X = cat(
        # ones(n),
        # cluster_center(P,k)[1] + R .* cos.(T),
        # cluster_center(P,k)[2] + R .* sin.(T),
        # rand(n,sd) .- 1/2, 
        # dims=2)  # 
    Y = A[P]
    Data(2, 1, n, X, Y)
end


##
"""
Plot the classifier for a test case, comparing training both  layer
k: number of clusters per dimension in the data set (choose 3, 4 or 5)
n: number of training samples
m: number of neurons
stepsize, niter: parameters for training
"""
function create_test_case(k=4, n=60)
    # data distribution (itself random)
    # plot training set
    data = Data(k, n)
    X, Y = data.X, data.Y
    X1 = X[(Y .== 1),:]
    X2 = X[(Y .== -1),:]
    fig = figure(figsize=[3,3])
    plot(X1[:,1],X1[:,2],"+k")
    plot(X2[:,1],X2[:,2],"_k")
    axis("equal");axis("off");
    display(fig)
    data
end

function train_test_case(data::Data; m=200, stepsize=0.05, niter=100_000)

    X, Y = data.X, data.Y
    (n, d) = size(X)
    model = TwoLayer(d, m)
    init!(model)
    # train the neural network
    twonet(model, X, Y, m, stepsize, niter)
end

"""
nframes, resolution: parameters for the plots
"""
function plot_results(result::Result, nframes::Int=20, resolution::Real=0.05)

    # define the sequence of time steps ts to be plotted, with a power law
    niter = result.niter
    a = (niter-1)/(nframes-1)^4
    ts = setdiff(Int.(floor.(a*(0:nframes-1).^4)) .+ 1)  # in order to get unique frames 
    Ws, margins, betas, loss = result.Ws, result.margins, result.betas, result.loss
    Ws = Ws[:,:,ts]

    m = size(Ws, 1)
    X, Y = data.X, data.Y
    # these are the positions we plot for the particles
    # Wproj of size (n,d,nframes)

    # Wproj = result.Ws[:,1:end-1,:] .* abs.(result.Ws[:,end:end,:])
    Wcut = Ws[:, 1:end-1, :]

    # the norm, size n*1*nframes
    Wout = result.Ws[:, end, :]  # will color the points
    vmin, vmax = minimum(Wout), maximum(Wout)
    Ncolors = 100
    cmapidx = floor.(Int, (Ncolors-1) .* (Wout .- vmin) ./ (vmax-vmin) .+ 1) # between 1 and N
    to_rgb(c::RGB) = [red(c), green(c), blue(c)]
    cmap = colormap("RdBu", Ncolors) 
    cmap = to_rgb.(cmap)
    # WN   = sqrt.(sum(Wproj.^2, dims = 2))
    # Wdir = Wproj ./ WN
    # Wlog = tanh.(0.5*WN) .* Wdir


    @showprogress 1 "Plotting images..." for k = 1:length(ts)
        ioff() # turns off interactive plotting
        fig = figure(figsize=[7,4])
        ax1 = subplot(121, projection="3d")
        ax1.set_position([0,0.1,0.5,0.8])

        indt = k<11 ? (1:k) : (k-10):k  # take the 10 most recent k for the tail

        for i = 1:size(Wcut,1)  #for all the  units
            # x, y, z is 2, 3, 1 ?
            plot3D(Wcut[i,1,indt],Wcut[i,2,indt],Wcut[i,3,indt], color="k", linewidth=0.2) # tail
            # plot3D(Wlog[i,2,indt],Wlog[i,3,indt],Wlog[i,1,indt], color="k", linewidth=0.2) # tail
        end
        # plot the positive in blue and the negative in red
        scatter3D(Wcut[:,1,k],Wcut[:,2,k],Wcut[:,3,k],"o",color=cmap[cmapidx[:, k]])

        # ax1.set_xlim3d(-1, 1)
        # ax1.set_ylim3d(-1, 1)
        # ax1.set_zlim3d(-1, 1)
        # ax1.set_xticks([-1/2, 0, 1/2])
        # ax1.set_yticks([-1/2, 0, 1/2])
        # ax1.set_zticks([-1/2, 0, 1/2])
        # ax1.view_init(25-20*sinpi(k/length(ts)),45-45*cospi(k/length(ts)))

        ax2 = subplot(122)
        # ax2.set_position([0.45,0.25,0.5,0.5])


        f(x1,x2,k) = (1/m) * sum( Ws[:,end,k] .* max.( Ws[:,1:3,k] * [x1;x2;1], 0.0)) # prediction function
        # with relu

        xs = -1:resolution:1
        tab = [f(xs[i],xs[j],k) for i=1:length(xs), j=1:length(xs)]
        pcolormesh(xs', xs, tanh.(tab'), cmap="coolwarm", shading="gouraud", vmin=-1.0, vmax=1.0, edgecolor="face")

        contour(xs', xs, tanh.(tab'), levels =0, colors="k", antialiased = true, linewidths=2)

        # plot training set
        X1 = X[(Y .== 1),:]
        X2 = X[(Y .== -1),:]
        plot(X1[:,1],X1[:,2],"+k")
        plot(X2[:,1],X2[:,2],"_k")
        axis("equal");axis("off");
        ax2.set_xticks([-1/2, 0, 1/2])
        ax2.set_yticks([-1/2, 0, 1/2])

        PyPlot.savefig("dynamics_$(k).png",bbox_inches="tight", dpi=300)
        close(fig)
    end


end
##
