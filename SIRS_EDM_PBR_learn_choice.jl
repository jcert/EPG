using DifferentialEquations
using Plots
using Optim, LinearAlgebra, Distributions
using Flux, ProgressMeter
using MLJ, DataFrames




#=
I had to modify one of the packages used to get it to work:
=#
import MLJLinearModels.sigmoid as mysig
import MLJLinearModels.SIGMOID_THRESH

SIGMOID_THRESH(T::Type{<:Real}) = log(1 / eps(T) - 1)

function mysig(x::T) where T <: Real
    τ = SIGMOID_THRESH(T)
    x > τ  && return one(T)
    x < -τ && return zero(T)
    return 1 / (1 + exp(-x))
end

mysig(x) = mysig(x)

#=
end of modifications
=#




#using Plotly



#plotly()
gr()
include("lib/SIRS_Game.jl")
include("lib/Dynamics.jl")
   
plt_ext = "png"

# EstimateFrom will sample from the:
# 1) state : x(t) - this fares worse, as expected
# 2) current choice function : C(p(t)) - this does ok, especially when learning only with the 15 last samples
# and add it to the data that is used to estimate \hat{C} 
EstimateFrom = 2


# Estimator will use:
# 1) logit choice function and look for η_estimated
# 2)
# 3) some NN with a softmax final layer 
# to approximate \hat{C} 
Estimator = 1




g0 = SIRS_Game(2,fp)

g0.x_star
g0.σ = 0.1
g0.ω = 0.005
g0.γ = g0.σ
g0.υ = 2.0
g0.β   = [0.15;0.19]
g0.c   = [0.2;0.0]
g0.c_star = 0.05
g0.ρ = 0.0

myPBR_η_real = 0.5
myPBR_η_estimated = 0.1

f!(du,u,p,t) = h_logit!(du,u,p,t;η=myPBR_η_real)

fixall_PBR!(g0;PBR_η=myPBR_η_estimated)

# for the bound to be meaningful we need to 
#    have S(x(0),p(0)) = 0  
I = Ib(g0,g0.β[1])
R = Rb(g0,g0.β[1])
S = 1.0-I-R

W = [g0.β[1]*I;g0.β[1]*R;1.0;0.0]

## assertions
# betas are in increasing order
@assert all(diff(g0.β).>0)
# c vector is in decreasing order
@assert all(diff(g0.c).<0)
# σ < β[1]
@assert all(g0.σ.<g0.β)
#c_star > g0.c[end]
@assert g0.c_star>0
@assert g0.c_star+g0.c[end]<g0.c[1]

#g0.r_star = [1.0;-10.0]

N = 300    # number of changes to r_star
Tmax = 60*N
k = 1000   # number of samples used to estimate C(r(current))
T = range(0,Tmax;length=N+1)

ts   = []
sols = []
D = []
DD = []

model = Chain(
    Dense(g0.NS => 5, tanh),   # activation function inside layer
    Dense(5 => g0.NS),
    softmax) |> gpu 
    
optim = Flux.setup(Flux.Adam(0.01), model)  # will store optimiser momentum, etc.

LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels

g0.υ = 3.0
myPBR_η_estimated
fixall_PBR!(g0;PBR_η=4.0)
for i_t in eachindex(T[1:end-1])

    #apply the dynamic payoff
    
    prob = ODEProblem(f!,W ,  [T[i_t], T[i_t+1]],g0)
    sol = solve(prob, AutoTsit5(Rosenbrock23()), save_everystep=true, saveat=0.1)  

    #sample from the current choice function value
    
    #either one of the approaches below
    if EstimateFrom == 1    
        #i) sample from the state
        x = sol[end][xi(g0,1:g0.NS-1)]
        x = [x..., 1-sum(x)]
    elseif EstimateFrom == 2
        #ii) sample from the choice function
        x = [C_logit(g0,sol[end],0.0,i;η=myPBR_η_real)  for i=1:g0.NS ]
    else
        throw("EstimateFrom=$EstimateFrom has not been implemented")
    end
     
    rr = [ fp(g0,sol[end],0.0,j) for j=1:g0.NS]
    p = Categorical(x)
    Dk = rand(p, k) #draw the strategy of 'k' agents
    append!(DD, map(x->(x,rr), Dk))
    C = mean([ d.==(1:g0.NS)  for d in Dk])

    push!(D,(C,rr))
    #while(length(D)>15)
    #    popfirst!(D)
    #end



    #estimate the choice function 
    if Estimator == 1    
        #i)   estimate within logit 
        l(η;r) = begin
            s = exp.(inv(abs(η[1]))*r)        
            s/sum(s)
        end 
        J(η) = sum( norm(l(η;r=d[2])-d[1], 1)  for d in D) + 0.001*norm(η)
    
        myPBR_η_estimated = optimize(J,[2.0], autodiff = :forward).minimizer[1]
        myPBR_η_estimated = abs(myPBR_η_estimated)
        fixall_PBR!(g0;PBR_η=myPBR_η_estimated)

    elseif Estimator == 2
        #ii) estimate using SVM the function C()
        target = Vector{Int32}([])
        noisy  = Vector{Vector{Float32}}([])
        for d in DD
            push!(target, d[1] ) 
            push!(noisy,  convert.(Float32, d[2]) )
        end
        target = categorical(target )
        noisy = MLJ.table(hcat(noisy...)' )

        mach = MLJ.fit!(machine(LogisticClassifier(), noisy, target))

        C_hat(x) =  collect(values(predict(mach, permutedims(x))[1].prob_given_ref))

        fixall_PBR_estimated!(g0;estimated_C=C_hat)
    elseif Estimator == 3
        #iii) estimate using general function C(r_star(current))
        # To train the model, we use batches of 64 samples, and one-hot encoding:
        target = Vector{Vector{Float32}}([])
        noisy  = Vector{Vector{Float32}}([])
        for d in D
            push!(target, convert.(Float32, d[1]) ) 
            push!(noisy,  convert.(Float32, d[2]) )
        end
        #do not do one-hot, target = Flux.onehotbatch(truth, [true, false])                   # 2×1000 OneHotMatrix
        
        loader = Flux.DataLoader((noisy, target) |> gpu, batchsize=64, shuffle=true);
        # 16-element DataLoader with first element: (2×64 Matrix{Float32}, 2×64 OneHotMatrix)

        # Training loop, using the whole data set 1000 times:
        losses = []
        Flux.@withprogress Flux.train!(model,  zip(noisy,target) |> gpu, optim) do m, x, y
            sum(abs.(m(x) .- y)) * 100
          end

        fixall_PBR_estimated!(g0;estimated_C=model)
    else
        throw("Estimator=$Estimator has not been implemented")
    end





    #determine the new r_star
    append!(ts,sol.t)
    sols = vcat(sols,sol.u)
    W = sol[end]
end


plot(ts,hcat(sols...)', label=["I" "R" "x_1" "q"], lw=2 )
savefig("images/EstimateFrom$(EstimateFrom)_Estimator$(Estimator)_zoomout_$(g0.c_star)_nu$(round(g0.υ,digits=1)).$(plt_ext)")
ylims!(-0.1,0.6)
savefig("images/EstimateFrom$(EstimateFrom)_Estimator$(Estimator)_zoomin_$(g0.c_star)_nu$(round(g0.υ,digits=1)).$(plt_ext)")

#ylims!(0.4,0.5)



##

X = -5:0.1:5

Z = [model([x;y])[1]  for x in X, y in X]

g(r) = (exp.(inv(myPBR_η_real)*r)./sum(exp, inv(myPBR_η_real)*r))[1];    
ZZ = [ g([x;y]) for x in X, y in X]

heatmap(X,X,ZZ)
savefig("images/Original_heatmap.$(plt_ext)")

heatmap(X,X,Z)
savefig("images/EstimateFrom$(EstimateFrom)_Estimator$(Estimator)_heatmap.$(plt_ext)")

ZZZ = [C_hat([x;y])[1]  for x in X, y in X]

heatmap(X,X,ZZZ)
savefig("images/EstimateFrom$(EstimateFrom)_Estimator$(Estimator)_heatmap.$(plt_ext)")


