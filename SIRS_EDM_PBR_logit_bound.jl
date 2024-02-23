##
using DifferentialEquations
using Plots
using Distributions
#using Plotly
using JuMP, Ipopt
using LaTeXStrings


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

myPBR_η = 0.10
myh_logit!(du,u,p,t) = h_logit!(du,u,p,t; η=myPBR_η)
fixall_PBR!(g0; PBR_η=myPBR_η)

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



function Create_Lyap_EDM_PBR(game)
    I_hat(B) = game.η*(B-game.σ)
    R_hat(B) = (1.0-game.η)*(B-game.σ)
    B_bar = game.β_star
    
    (I,R,B)->(I-I_hat(B))+I_hat(B)*log(I_hat(B)/I)+(R-R_hat(B))^2/(2*game.γ)+game.υ^2*(B-B_bar)^2/2
end

function Storage_PBR_Q(game, x,p,Q)

    m = Model(Ipopt.Optimizer)
    set_silent(m)
    @variable(m, y[1:game.NS]>=0)

    @constraint(m, sum(y)==1)

    myQ(x...) = Q(collect(x))
    register(m, :myQ, game.NS, myQ; autodiff = true)
    
    aux = (x'*p-Q(x))
    @NLobjective(m, Max, (sum(y[i]*p[i] for i = 1:length(y) )-myQ(y...))-aux )

    optimize!(m)

    objective_value(m)
end

function pi_star(α;game)
    I_star = Ib(game,g0.β_star)
    Lyap_fun = Create_Lyap_EDM_PBR(game)
    # S(x,p) = max(y'*p-Q(y))  -(x'*p-Q(x)) #this need and optimization of its own


    m = Model(Ipopt.Optimizer)
    set_optimizer_attribute(m, "tol", 1e-12)
    set_optimizer_attribute(m, "acceptable_tol",  1e-12)      
    set_optimizer_attribute(m, "dual_inf_tol",  1e-12)
    set_optimizer_attribute(m, "acceptable_constr_viol_tol",  1e-12)
    set_optimizer_attribute(m, "bound_relax_factor",  0.0)
    set_optimizer_attribute(m, "mu_init",  1e-1)
    set_optimizer_attribute(m, "gamma_theta", 1e-2)
    set_optimizer_attribute(m, "honor_original_bounds",  "yes")
    set_optimizer_attribute(m, "max_iter", 15000)
    set_silent(m)

    @variable(m, I>=0)
    @variable(m, R>=0)
    @variable(m, maximum(game.β)>=B>=minimum(game.β))

    @constraint(m, I+R<=B)

    register(m, :Lyap_fun, 3, Lyap_fun; autodiff = true)
    #@NLconstraint(m, α>=Lyap_fun(I,R,B) )
    
    @NLconstraint(m, (I-g0.η*(B-g0.σ))+g0.η*(B-g0.σ)*log(g0.η*(B-g0.σ)/I)+(R-(1-g0.η)*(B-g0.σ))^2/(2*g0.γ)+g0.υ^2*(B-g0.β_star)^2/2<=α   )

    
    
    @NLobjective(m, Max, I/B )
    optimize!(m)

    value.([I,R,B])
    objective_value(m)/(g0.η*(1-g0.σ/g0.β_star))
    end


##

g0.c_star = 3.0
fixall_PBR!(g0;PBR_η=myPBR_η)

#g0.r_star = [1.0;0.0]
#g0.β_star = 0.7*g0.β[1]+0.3*g0.β[end] 

prob = ODEProblem(myh_logit!,W ,[0.0,250000.0],g0)
x0 = solve(prob, AutoTsit5(Rosenbrock23()), save_everystep=true, saveat=1.0).u[end]  

Storage_PBR_Q(g0,[x0[3];1-x0[3]],g0.r_star,z->Q_logit(g0,z; η=myPBR_η))


plot()
@show β_o = Bx(g0,x0)
r_o = g0.r_star
for  Δβ = [0.005,0.01,0.015,0.02]
    g0.β_star = β_o+Δβ
    #r_tilda = r_o-g0.r_star
    #Bs(r_tilda) == 0 if r_tilda==0
    X = [0.005,0.012,0.025,0.05,0.1:0.01:1...]
    #Y = [ pi_star(g0.υ^2*(β_o-g0.β_star)^2/2; game=g0)[1]/x0[1] for g0.υ=X ]
    Y = [ pi_star(g0.υ^2*(β_o-g0.β_star)^2/2; game=g0) for g0.υ=X ]

    plot!(X,Y,label=L"\tilde\beta="*"$(round(Δβ;digits=3))")
end
plot!()

savefig("images/bound.png")
##