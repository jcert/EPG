using DifferentialEquations
using Plots
using Optim, LinearAlgebra, Distributions

#using Plotly



#plotly()
gr()
include("lib/SIRS_Game.jl")
include("lib/Dynamics.jl")
   
plt_ext = "png"
plt_size = (400,300)

# EstimateFrom will sample from the:
# 1) 
# 2) current choice function : C(p(t)) - this does ok, especially when learning only with the 15 last samples
# and add it to the data that is used to estimate \hat{C} 
EstimateFrom = 2



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

myPBR_η_estimated = 0.01 #highly rational
#myPBR_η_estimated = 10.0 #highly irrational



fixall_PBR!(g0;PBR_η=myPBR_η_estimated)


myPBR_η_real = 0.5
f!(du,u,p,t) = h_logit!(du,u,p,t;η=myPBR_η_real)

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


sols = ( t=[0.0], u=[W])
us = []
ts   = []
D = []
DD = []


pre_C_choice_learned = true
switch_time = 0.0
g0.υ = 3.0

Tmax = 3000
Δt = 30 # sample every Δt days
t = 0.0;
previous_sample_t = 0
num_samples = 1000

prob = ODEProblem(f!, W, [0, Tmax], g0)
integrator = init(prob, Tsit5(); dt=1//2^4, tstops=0:0.3:Tmax)  

function q_converged(us;ϵ=0.02)
    s = 0
    for i in abs.(diff(us[max(end-1000,1):end]))
        s /= 1.01
        s += i
    end
    s<ϵ
end

##

for i in integrator
    push!(us, integrator.u[end])
    push!(sols.u,copy(integrator.u))
    push!(sols.t,integrator.t)
    
    #sample from the current choice function value
    if EstimateFrom == 2
        #ii) sample from the choice function
        x = [C_logit(g0,integrator.u,0.0,i;η=myPBR_η_real)  for i=1:g0.NS ]
    else
        throw("EstimateFrom=$EstimateFrom has not been implemented")
    end

    if previous_sample_t+Δt < integrator.t
        rr = [ fp(g0,integrator.u,0.0,j) for j=1:g0.NS]
        p = Categorical(x)
        C = mean(map(x->x.==(1:g0.NS), rand(p,num_samples) ))
        push!(D,(C,rr))
    end


    if integrator.t>1 && pre_C_choice_learned && q_converged(us)
        l(η;r) = begin
            s = exp.(inv(abs(η[1]))*r)        
            s/sum(s)
        end 
        J(η) = sum( norm(l(η;r=d[2])-d[1], 1)  for d in D) + 0.00001*norm(η)
        
        myPBR_η_estimated = optimize(J,[2.0], autodiff = :forward).minimizer[1]
        myPBR_η_estimated = abs(myPBR_η_estimated)
        fixall_PBR!(g0;PBR_η=myPBR_η_estimated)
        
        pre_C_choice_learned = false
        u_modified!(integrator,true)
        @show integrator.t
        global switch_time = integrator.t
    end

end

y = hcat(sols.u...)

# plots the ratio of infected 
X = mapslices(x->[x;1.0-sum(x)], y[xi(g0,1:g0.NS-1),:], dims=1)
i_star = (g0.η*(1-g0.σ/g0.β_star))
I_ratio = (y[1,:]./(g0.β'*X)')./(i_star)
Plots.plot(sols.t, I_ratio, label="I,υ=$(g0.υ)")
plot!([switch_time,switch_time],[minimum(I_ratio),maximum(I_ratio)], linestyle=:dash, label=false)
ylabel!("I/I*")
xlabel!("Days")
plot!(size=plt_size)
savefig("images/EstimateFrom$(EstimateFrom)_logit_infected_$(g0.c_star)_nu$(round(g0.υ,digits=1)).$(plt_ext)")



# plots the ratio of agents using the first strategy
plot(sols.t, y[3,:], lw=2, label="x_1,υ=$(g0.υ)")
plot!([switch_time,switch_time],[minimum(y[3,:]),maximum(y[3,:])], linestyle=:dash, label=false)
ylabel!("x_1")
xlabel!("Days")
plot!(size=plt_size)
savefig("images/EstimateFrom$(EstimateFrom)_logit_strats_$(g0.c_star)_nu$(round(g0.υ,digits=1)).$(plt_ext)")


# plots the state of the dynamic payoff
plot(sols.t, y[4,:], lw=2, label="q,υ=$(g0.υ)")
plot!([switch_time,switch_time],[minimum(y[4,:]),maximum(y[4,:])], linestyle=:dash, label=false)
plot!(size=plt_size)
savefig("images/EstimateFrom$(EstimateFrom)_logit_qzoomout_$(g0.c_star)_nu$(round(g0.υ,digits=1)).$(plt_ext)")
ylims!(-2,2)
plot!(size=plt_size)
savefig("images/EstimateFrom$(EstimateFrom)_logit_qzoomin_$(g0.c_star)_nu$(round(g0.υ,digits=1)).$(plt_ext)")

