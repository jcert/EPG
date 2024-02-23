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
    plt_size = (800,800)
    if USE_DEFED && (USE_TIKZ)
        plt_I_size = plt_size.*(1.0,0.9)
    else
        plt_I_size = plt_size
    end
    default(;DEF_CONFIG...)
else
    plt_size = (800,800)
    plt_I_size = plt_size
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

σ = 0.5

dists = [
 Cauchy(0,σ)
 Logistic(0.0,σ)
 Laplace(0,σ)
 Normal(0,σ)
 Gumbel(0.0,σ*sqrt(6/pi^2)) 
 #Exponential(1.0) #halfline support 
 #Pareto(1.0, 1.0) #halfline support 
 #Rayleigh(σ) #halfline support 
 #Epanechnikov(0,σ) #compact support 
 #Semicircle(1.0) #compact support 
 #SymTriangularDist(0.0, σ) #compact support 
 #Uniform(-σ,σ) #compact support 
]

my_dist = dists[1]

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
υ_list = [2.0]
TA = 6000.0
TB = 800.0

tmax = 0.0
f1 = plot()
f2 = plot()
f3 = plot()

#[2,2.65,

#for my_dist in dists[[1]],
for my_dist in dists

    ## Figure 2.b.top
    for g0.υ = υ_list
        C(r) = C_dist(g0,r,1:g0.NS,dist=my_dist)
        fixall_PBR_givenC!(g0, C)
        
        prob = ODEProblem((du, u, p, t)->h_dist!(du, u, p, t,dist=my_dist),W ,[0.0,TB],g0)
        sol = solve(prob, AutoTsit5(Rosenbrock23()), save_everystep=true, saveat=1.0)  

        X = mapslices(x->[x;1.0-sum(x)], sol[xi(g0,1:g0.NS-1),:], dims=1)
        rr = (sol[qi(g0,1),:]'.*g0.β).+g0.r_star
        plot!(f1,sol.t, sum(X.*rr,dims=1)', label="$(string(typeof(my_dist))[1:end-9])", legend = :outerright)
    end
    plot!(f1, size=plt_size )
    plot!(f1, x->g0.c_star,c=:black,linestyle=:dash, label=nothing)
    ylabel!(f1, L"r'(t)x(t)")
    xlabel!(f1, "days")
    ylims!(f1, -0.5,0.3)


    ## Figure 2.a.bottom
    for g0.υ = υ_list
        C(r) = C_dist(g0,r,1:g0.NS,dist=my_dist)
        fixall_PBR_givenC!(g0, C)
        prob = ODEProblem((du, u, p, t)->h_dist!(du, u, p, t,dist=my_dist),W ,[0.0,TB],g0)
        #DP5()
        #Euler(),dt=0.001 
        # AutoTsit5(Rosenbrock23())
        sol = solve(prob, AutoTsit5(Rosenbrock23()), save_everystep=true, saveat=1.0)  

        X = mapslices(x->[x;1.0-sum(x)], sol[xi(g0,1:g0.NS-1),:], dims=1)
        
        i_star = (g0.η*(1-g0.σ/g0.β_star))
        Plots.plot!(f2,sol.t, (sol[1,:]./(g0.β'*X)')./(i_star), label="$(string(typeof(my_dist))[1:end-9])",legend = :outerright)
    
        global tmax = sol.t[findmax((sol[1,:]./(g0.β'*X)')./(i_star))[2]]
    
    end
    plot!(f2,size=plt_I_size,legend=nothing)
    ylabel!(f2,L"I(t)/I^*")
    xlabel!(f2,"days")

    println(f2,"I_star = $(g0.η*(1-g0.σ/g0.β_star))")
    println(f2,"R_star = $((1-g0.η)*(1-g0.σ/g0.β_star))")


    ## Figure 2.b.bottom
    for g0.υ = υ_list
        C(r) = C_dist(g0,r,1:g0.NS,dist=my_dist)
        fixall_PBR_givenC!(g0, C)
        prob = ODEProblem((du, u, p, t)->h_dist!(du, u, p, t,dist=my_dist),W ,[0.0,TB],g0)
        sol = solve(prob, AutoTsit5(Rosenbrock23()), save_everystep=true, saveat=1.0)  

        #plot!(sol.t,sol.u[1,:], label="cost(t),υ=$(g0.υ)")
        plot!(f3,sol, idxs=(3), label="$(string(typeof(my_dist))[1:end-9])",legend = :outerright)
    end
    plot!(f3, size=plt_size)
    ylims!(f3, (0.50,0.7))
    ylabel!(f3, L"x_1(t)")
    xlabel!(f3, "days")

end
savefig(f1, "images/1.a.top.SIRS_EDM_cost_$(g0.c_star)_nu$(round(g0.υ,digits=1)).$(plt_ext)")
savefig(f3, "images/1.c.bottom.SIRS_EDM_x1_$(g0.c_star)_nu$(round(g0.υ,digits=1)).$(plt_ext)")


plot!(f2)
w = 5*12
h = 0.05*2
plot!(Shape(tmax .+ [-w,w,w,-w], 1.45 .+ [-h,-h,h,h]), opacity=.15, label=nothing)
ylabel!(L"I(t)/I^*")
xlabel!("days")
savefig(f2,"images/1.b.bottom.SIRS_EDM_I_ratio_$(g0.c_star)_nu$(round(g0.υ,digits=1)).$(plt_ext)")


plot!(f2, size=plt_size, legend = :outerright)
xlims!(tmax.+(-5.0,5.0))
ylims!((1.4,1.50))
savefig("images/1.d.bottom.SIRS_EDM_I_ratio_peak_$(g0.c_star)_nu$(round(g0.υ,digits=1)).$(plt_ext)")
##