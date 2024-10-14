##
using DifferentialEquations
using Plots
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
    plt_size = (600,500)
    #plt_size = (900,900)
    default(linewidth = 3, markersize=10, margin = 10*Plots.mm,
        tickfontsize=12, guidefontsize=20, legend_font_pointsize=18)

end
    


g0 = SIRS_Game(2,fp)

g0.x_star
g0.σ = 0.1
g0.ω = 0.005
g0.γ = g0.σ
g0.υ = 10.0
g0.β   = [0.15;0.19]
g0.c   = (g,z)->[0.2038-0.2*z[1]; 0.0]
g0.c_star = 6.00
g0.ρ = 0.0

myPBR_η = 1.0

fixall_PBR!(g0;PBR_η=myPBR_η)

# for the bound to be meaningful we need to 
#    have S(x(0),p(0)) = 0  
I = Ib(g0,0.15012)
R = Rb(g0,0.15012)
S = 1.0-I-R

#W = [g0.β[1]*I;g0.β[1]*R;1.0;0.0]
W = [   Bx(g0,[0;0;g0.x_star[1:end-1]...;0])*I;
        Bx(g0,[0;0;g0.x_star[1:end-1]...;0])*R;
        1.0;
        0.0]

g0.c_star = 0.15

## assertions
# betas are in increasing order
@assert all(diff(g0.β).>0)
# c vector is in decreasing order - #TODO now c is a function
#@assert all(diff(g0.c).<0)
# σ < β[1]
@assert all(g0.σ.<g0.β)
#c_star > g0.c[end]
@assert g0.c_star>0
#- #TODO now c is a function
#@assert g0.c_star+g0.c[end]<g0.c[1]

#g0.r_star = [1.0;-10.0]

## Figure 2.a.top
#=
plot()
for g0.υ = [10.0,5.0,3.0,0.5]
    
    fixall_PBR!(g0;PBR_η=myPBR_η)
    prob = ODEProblem(h_logit!,W ,[0.0,1500.0],g0)
    #DP5()
    #Euler(),dt=0.001 
    # AutoTsit5(Rosenbrock23())
    sol = solve(prob, AutoTsit5(Rosenbrock23()), save_everystep=true, saveat=1.0)  

    X = mapslices(x->[x;1.0-sum(x)], sol[xi(g0,1:g0.NS-1),:], dims=1)
    
    i_star = (g0.η*(1-g0.σ/g0.β_star))
    Plots.plot!(sol.t, (sol[1,:]./(g0.β'*X)')./(i_star), label="I,υ=$(g0.υ)")
end
plot!()
ylabel!("I/I*")
xlabel!("days")
println("I_star = $(g0.η*(1-g0.σ/g0.β_star))")
println("R_star = $((1-g0.η)*(1-g0.σ/g0.β_star))")
savefig("images/2.a.top.SIRS_EDM_I_ratio_$(g0.c_star)_nu$(round(g0.υ,digits=1)).$(plt_ext)")
=#

## Figure 2.b.top
plot()
for g0.υ = [10.0,5.0,3.0,1.0]
    fixall_PBR!(g0;PBR_η=myPBR_η)
    prob = ODEProblem(h_logit!,W ,[0.0,1500.0],g0)
    sol = solve(prob, AutoTsit5(Rosenbrock23()), save_everystep=true, saveat=1.0)  

    X = mapslices(x->[x;1.0-sum(x)], sol[xi(g0,1:g0.NS-1),:], dims=1)
    rr = (sol[qi(g0,1),:]'.*g0.β).+g0.r_star
    plot!(sol.t, sum(X.*rr,dims=1)', label="υ=$(g0.υ)")
end
plot!(size=plt_size, legend=:topright)
plot!(x->g0.c_star,c=:black,linestyle=:dash, label=nothing)
ylabel!(L"r'(t)x(t)")
xlabel!("days")
savefig("images/1.a.top.SIRS_EDM_cost_$(g0.c_star)_nu$(round(g0.υ,digits=1)).$(plt_ext)")


## Figure 2.a.bottom
plot()
for g0.υ = [10.0,5.0,3.0,1.0]
    fixall_PBR!(g0;PBR_η=myPBR_η)
    prob = ODEProblem(h_logit!,W ,[0.0,1500.0],g0)
    #DP5()
    #Euler(),dt=0.001 
    # AutoTsit5(Rosenbrock23())
    sol = solve(prob, AutoTsit5(Rosenbrock23()), save_everystep=true, saveat=1.0)  

    X = mapslices(x->[x;1.0-sum(x)], sol[xi(g0,1:g0.NS-1),:], dims=1)
    
    i_star = (g0.η*(1-g0.σ/g0.β_star))
    Plots.plot!(sol.t, (sol[1,:]./(g0.β'*X)')./(i_star), label=LaTeXString(L"\upsilon="*"$(g0.υ)"))
end
plot!(size=plt_size, legend=:topright)
ylabel!(L"I(t)/I^*")
xlabel!("days")
println("I_star = $(g0.η*(1-g0.σ/g0.β_star))")
println("R_star = $((1-g0.η)*(1-g0.σ/g0.β_star))")
savefig("images/1.b.bottom.SIRS_EDM_I_ratio_$(g0.c_star)_nu$(round(g0.υ,digits=1)).$(plt_ext)")


## Figure 2.b.bottom
plot()
for g0.υ = [10.0,5.0,3.0,1.0]
    fixall_PBR!(g0;PBR_η=myPBR_η)
    prob = ODEProblem(h_logit!,W ,[0.0,1500.0],g0)
    sol = solve(prob, AutoTsit5(Rosenbrock23()), save_everystep=true, saveat=1.0)  

    #plot!(sol.t,sol.u[1,:], label="cost(t),υ=$(g0.υ)")
    plot!(sol, idxs=(3), label=LaTeXString(L"\upsilon="*"$(g0.υ)"))
end
plot!(size=plt_size, legend=:topright)
ylabel!(L"x_1(t)")
xlabel!("days")
savefig("images/1.c.bottom.SIRS_EDM_x1_$(g0.c_star)_nu$(round(g0.υ,digits=1)).$(plt_ext)")





##