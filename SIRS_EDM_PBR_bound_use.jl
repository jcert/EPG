##
using DifferentialEquations
using Plots
using LaTeXStrings
#using Plotly



#plotly()
gr()
include("lib/SIRS_Game.jl")
include("lib/Dynamics.jl")
   
plt_ext = "png"
plt_size = (400,300)


g0 = SIRS_Game(2,fp)

g0.x_star
g0.σ = 0.1
g0.ω = 0.005
g0.γ = g0.σ
g0.υ = 2.0
g0.β   = [0.15;0.19]
g0.c   = [0.2;0.0]
g0.c_star = 0.28
g0.ρ = 0.0

β_o = 0.155
β_tilde = 0.01

myPBR_η = 0.014
myh_logit!(du,u,p,t) = h_logit!(du,u,p,t; η=myPBR_η)
T = 1500.0

Y = []
X = 0.1:0.01:30.0
for g0.c_star = X
    fixall_PBR!(g0;PBR_η=myPBR_η)
    append!(Y, g0.β_star)
end
#plot(X,Y)
ix = findfirst(x->abs(x-β_o)<0.001,Y)
g0.c_star = X[ix]
g0.β_star = Y[ix]
fixall_PBR!(g0;PBR_η=myPBR_η)
bounded_υ = 0.875


# for the bound to be meaningful we need to 
#    have S(x(0),p(0)) = 0
W = [0.0026;0.0533;0.85;0.0]
prob = ODEProblem(myh_logit!,W ,[0.0,10000.0],g0)
sol = solve(prob, AutoTsit5(Rosenbrock23()), save_everystep=true, saveat=1.0)  
W = sol[end]
plot(sol)


g0.β_star = β_o+β_tilde

## assertions
# betas are in increasing order
@assert all(diff(g0.β).>0)
# c vector is in decreasing order
@assert all(diff(g0.c).<0)
# σ < β[1]
@assert all(g0.σ.<g0.β)
#c_star > g0.c[end]
@assert g0.c_star>0
#@assert g0.c_star+g0.c[end]<g0.c[1]

#g0.r_star = [1.0;-10.0]

list_υ = [ 0.2, bounded_υ, 4.0]

##

plot()
for g0.υ = list_υ
    #fixall_PBR!(g0;PBR_η=myPBR_η)
    prob = ODEProblem(myh_logit!,W ,[0.0,T],g0)
    sol = solve(prob, AutoTsit5(Rosenbrock23()), save_everystep=true, saveat=1.0)  

    X = mapslices(x->[x;1.0-sum(x)], sol[xi(g0,1:g0.NS-1),:], dims=1)
    rr = (sol[qi(g0,1),:]'.*g0.β).+g0.r_star
    plot!(sol.t, sum(X.*rr,dims=1)', label="cost(t),υ=$(g0.υ)")
end
plot!(size=plt_size)
plot!(x->g0.c_star,c=:black,linestyle=:dash, label=nothing)
ylabel!("Cost - r(t)")
xlabel!("Days")
savefig("images/1.a.top.SIRS_EDM_cost_$(g0.c_star)_nu$(round(g0.υ,digits=1)).$(plt_ext)")


## Figure 2.a.bottom
plot()
for g0.υ = list_υ
    #fixall_PBR!(g0;PBR_η=myPBR_η)
    prob = ODEProblem(myh_logit!,W ,[0.0,T],g0)
    #DP5()
    #Euler(),dt=0.001 
    # AutoTsit5(Rosenbrock23())
    sol = solve(prob, AutoTsit5(Rosenbrock23()), save_everystep=true, saveat=1.0)  

    X = mapslices(x->[x;1.0-sum(x)], sol[xi(g0,1:g0.NS-1),:], dims=1)
    
    i_star = (g0.η*(1-g0.σ/g0.β_star))
    Plots.plot!(sol.t, (sol[1,:]./(g0.β'*X)')./(i_star), label="I,υ=$(g0.υ)")
end
plot!(size=plt_size)
ylabel!("I/I*")
xlabel!("Days")
println("I_star = $(g0.η*(1-g0.σ/g0.β_star))")
println("R_star = $((1-g0.η)*(1-g0.σ/g0.β_star))")
title!(L"\tilde{\beta}="*"$β_tilde")
savefig("images/1.b.bottom.SIRS_EDM_I_ratio_$(g0.c_star)_nu$(round(g0.υ,digits=1)).$(plt_ext)")


## Figure 2.b.bottom
plot()
for g0.υ = list_υ
    #fixall_PBR!(g0;PBR_η=myPBR_η)
    prob = ODEProblem(myh_logit!,W ,[0.0,T],g0)
    sol = solve(prob, AutoTsit5(Rosenbrock23()), save_everystep=true, saveat=1.0)  

    #plot!(sol.t,sol.u[1,:], label="cost(t),υ=$(g0.υ)")
    plot!(sol, vars=(3), label="x_1,υ=$(g0.υ)")
end
plot!(size=plt_size)
ylabel!("x_1")
xlabel!("Days")
savefig("images/1.c.bottom.SIRS_EDM_x1_$(g0.c_star)_nu$(round(g0.υ,digits=1)).$(plt_ext)")





##