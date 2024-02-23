##
using DifferentialEquations
using Plots
using Distributions
#using Plotly



#plotly()
gr()
include("lib/SIRS_Game.jl")
include("lib/Dynamics.jl")

USE_TIKZ = false
if (USE_TIKZ)
    pgfplotsx()
    plt_ext = "tikz"
else
    #plotly()
    gr()
    plt_ext = "png"
end

plt_size = (400,300)
#plt_size = (900,900)
default(linewidth = 3, markersize=10, margin = 10*Plots.mm,
    tickfontsize=12, guidefontsize=12, legend_font_pointsize=12)


g0 = SIRS_Game(2,fp)

g0.x_star
g0.σ = 0.1
g0.ω = 0.005
g0.γ = g0.σ
g0.υ = 2.0
g0.β   = [0.15;0.19]
g0.c   = [0.2;0.0]
g0.c_star = 0.15
g0.ρ = 0.0

σ = 1.0

my_dist = Normal(0,σ)

C(r) = C_dist(g0,r,1:g0.NS,dist=my_dist)
fixall_PBR_givenC!(g0, C)

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

υ_list = [10.0,5.0,3.0,0.5]
υ_list = [3.0]
T = 2000.0
f1 = plot()
f2 = plot()
f3 = plot()

#[2,2.65,

#for my_dist in dists[[1]],
## Figure 2.b.top
for g0.υ = υ_list
    C(r) = C_dist(g0,r,1:g0.NS,dist=my_dist)
    fixall_PBR_givenC!(g0, C)
    
    prob = ODEProblem((du, u, p, t)->h_dist!(du, u, p, t,dist=my_dist),W ,[0.0,T],g0)
    sol = solve(prob, AutoTsit5(Rosenbrock23()), save_everystep=true, saveat=1.0)  

    X = mapslices(x->[x;1.0-sum(x)], sol[xi(g0,1:g0.NS-1),:], dims=1)
    rr = (sol[qi(g0,1),:]'.*g0.β).+g0.r_star
    plot!(f1,sol.t, sum(X.*rr,dims=1)', label="$(string(typeof(my_dist))[1:end-9]),cost(t),υ=$(g0.υ)", legend = nothing)
end
plot!(size=plt_size)
plot!(x->g0.c_star,c=:black,linestyle=:dash, label=nothing)
ylabel!("Cost - r(t)")
xlabel!("Days")
ylims!(-0.4,0.4)
savefig("images/dist.a.SIRS_EDM_cost_$(g0.c_star)_nu$(round(g0.υ,digits=1)).$(plt_ext)")


## Figure 2.a.bottom
for g0.υ = υ_list
    C(r) = C_dist(g0,r,1:g0.NS,dist=my_dist)
    fixall_PBR_givenC!(g0, C)
    prob = ODEProblem((du, u, p, t)->h_dist!(du, u, p, t,dist=my_dist),W ,[0.0,T],g0)
    #DP5()
    #Euler(),dt=0.001 
    # AutoTsit5(Rosenbrock23())
    sol = solve(prob, AutoTsit5(Rosenbrock23()), save_everystep=true, saveat=1.0)  

    X = mapslices(x->[x;1.0-sum(x)], sol[xi(g0,1:g0.NS-1),:], dims=1)
    
    i_star = (g0.η*(1-g0.σ/g0.β_star))
    Plots.plot!(f2,sol.t, (sol[1,:]./(g0.β'*X)')./(i_star), label="$(string(typeof(my_dist))[1:end-9]),I,υ=$(g0.υ)",legend = nothing)
end
plot!(size=plt_size)
ylabel!("I/I*")
xlabel!("Days")

println("I_star = $(g0.η*(1-g0.σ/g0.β_star))")
println("R_star = $((1-g0.η)*(1-g0.σ/g0.β_star))")
savefig("images/dist.b.SIRS_EDM_I_ratio_$(g0.c_star)_nu$(round(g0.υ,digits=1)).$(plt_ext)")


## Figure 2.b.bottom
for g0.υ = υ_list
    C(r) = C_dist(g0,r,1:g0.NS,dist=my_dist)
    fixall_PBR_givenC!(g0, C)
    prob = ODEProblem((du, u, p, t)->h_dist!(du, u, p, t,dist=my_dist),W ,[0.0,T],g0)
    sol = solve(prob, AutoTsit5(Rosenbrock23()), save_everystep=true, saveat=1.0)  

    #plot!(sol.t,sol.u[1,:], label="cost(t),υ=$(g0.υ)")
    plot!(f3,sol, vars=(3), label="$(string(typeof(my_dist))[1:end-9]),x_1,υ=$(g0.υ)",legend = nothing)
end
plot!(size=plt_size)
ylabel!("x_1")
xlabel!("Days")
savefig("images/dist.c.SIRS_EDM_x1_$(g0.c_star)_nu$(round(g0.υ,digits=1)).$(plt_ext)")




##