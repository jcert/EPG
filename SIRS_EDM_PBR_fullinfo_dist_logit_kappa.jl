##
using DifferentialEquations
using Plots
using Distributions
using LaTeXStrings
#using Plotly



#plotly()
gr()
include("lib/SIRS_Game.jl")
include("lib/Dynamics.jl")

USE_DEFED = @isdefined USE_TIKZ 
if USE_DEFED && (USE_TIKZ)
    pgfplotsx()
    plt_ext = "tikz"
else
    #plotly()
    gr()
    plt_ext = "png"
end

if @isdefined DEF_CONFIG
    plt_size = (800,600)
    default(;DEF_CONFIG...)
else
    plt_size = (800,600)
    #plt_size = (900,900)
    default(linewidth = 3, markersize=10, margin = 10*Plots.mm,
        tickfontsize=12, guidefontsize=20, legend_font_pointsize=18)

end


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

η_real = 1.0
C(r) = C_logit_payoff(g0,r,0.0,1:g0.NS;η=η_real)
fixall_PBR!(g0; PBR_η=η_real)

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

υ_list = [3.0]
k_list = [0,1,4,10]
T = 1000.0
f1 = plot()
f2 = plot()
f3 = plot()

#[2,2.65,

#
## Figure 2.b.top
for k = k_list, g0.υ = υ_list
    fixall_PBR!(g0; PBR_η=η_real)

    prob = ODEProblem((du, u, p, t)->h_logit!(du,u,p,t;η=η_real,kappa=k),W ,[0.0,T],g0)
    sol = solve(prob, AutoTsit5(Rosenbrock23()), save_everystep=true, saveat=1.0)  

    X = mapslices(x->[x;1.0-sum(x)], sol[xi(g0,1:g0.NS-1),:], dims=1)
    rr = (sol[qi(g0,1),:]'.*g0.β).+g0.r_star
    plot!(f1,sol.t, sum(X.*rr,dims=1)', label="κ=$(k)", legend = nothing)
end
plot!(size=plt_size, legend=:bottomright)
plot!(x->g0.c_star,c=:black,linestyle=:dot, label=nothing)
ylabel!(L"r'(t)x(t)")
xlabel!("days")
#ylims!(-1.2,0.8)
savefig("images/kappa_logit.a.SIRS_EDM_cost_$(g0.c_star)_nu$(round(g0.υ,digits=1)).$(plt_ext)")


## Figure 2.a.bottom
for k = k_list, g0.υ = υ_list
    fixall_PBR!(g0; PBR_η=η_real)
    prob = ODEProblem((du, u, p, t)->h_logit!(du,u,p,t;η=η_real,kappa=k),W ,[0.0,T],g0)
    #DP5()
    #Euler(),dt=0.001 
    # AutoTsit5(Rosenbrock23())
    sol = solve(prob, AutoTsit5(Rosenbrock23()), save_everystep=true, saveat=1.0)  

    X = mapslices(x->[x;1.0-sum(x)], sol[xi(g0,1:g0.NS-1),:], dims=1)
    
    i_star = (g0.η*(1-g0.σ/g0.β_star))
    Plots.plot!(f2,sol.t, (sol[1,:]./(g0.β'*X)')./(i_star), label="κ=$(k)",legend = nothing)
end
plot!(size=plt_size, legend=:topright)
ylabel!(L"$I(t)/I^*$")
xlabel!("days")


println("I_star = $(g0.η*(1-g0.σ/g0.β_star))")
println("R_star = $((1-g0.η)*(1-g0.σ/g0.β_star))")
savefig("images/kappa_logit.b.SIRS_EDM_I_ratio_$(g0.c_star)_nu$(round(g0.υ,digits=1)).$(plt_ext)")


## Figure 2.b.bottom
for k = k_list, g0.υ = υ_list
    fixall_PBR!(g0; PBR_η=η_real)
    prob = ODEProblem((du, u, p, t)->h_logit!(du,u,p,t;η=η_real,kappa=k),W ,[0.0,T],g0)
    sol = solve(prob, AutoTsit5(Rosenbrock23()), save_everystep=true, saveat=1.0)  

    #plot!(sol.t,sol.u[1,:], label="cost(t),υ=$(g0.υ)")
    plot!(f3,sol, idxs=(3), label="κ=$(k)",legend = nothing)
end
plot!(size=plt_size, legend=true)
ylabel!(L"x_1")
yaxis!(f3,(0.48,0.6))
xlabel!("days")
savefig("images/kappa_logit.c.SIRS_EDM_x1_$(g0.c_star)_nu$(round(g0.υ,digits=1)).$(plt_ext)")




##