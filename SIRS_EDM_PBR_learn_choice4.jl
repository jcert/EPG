##
using DifferentialEquations
using Plots
using Optim, LinearAlgebra, Distributions

#using Plotly



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

plt_size = (400,300)
default(linewidth = 3, markersize=10, margin = 10*Plots.mm,
    tickfontsize=12, guidefontsize=12, legend_font_pointsize=12)

#from Shinkyu's python script - approximates the data generated there
beta_bar(upper_mu) = 0.16780517281035134 + 0.0021801407600000535*upper_mu - 0.0011290415704391496*upper_mu^2 + 0.0003156422494339572*upper_mu^3 - 4.5077461124856326e-5*upper_mu^4 + 2.5756028853492297e-6*upper_mu^5
 


function line_search(f,target,a,b;ϵ=0.01)
    #@show (a,b),(f(a),target)
    if f(a)>f(b)
        return line_search(f,target,b,a)
    else
        if abs(a-b)<ϵ
            return a
        else
            if f((a+b)/2)>target
                line_search(f,target,a,(a+b)/2)
            else
                line_search(f,target,(a+b)/2,b)
            end
        end
    end
end


function find_parameter_range(game,r_minus_c_tilda,meanR,sample_num;ϵ=0.00001)
    local C(μ) = r_minus_c_tilda'*[C_logit_payoff(game,r_minus_c_tilda,0.0,i;η=μ) for i=1:game.NS]
    range_μ = (line_search(C,meanR-ϵ,0.01,40.0;ϵ=ϵ), 
                line_search(C,meanR+ϵ,0.01,40.0;ϵ=ϵ))
    probability = 1-1/(sample_num*ϵ^2)

    (probability, range_μ)
end


# find_parameter_range(g0,[1.0,-1.0],0.74,100_000;ϵ=0.05)




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


myPBR_η_real = 1.0
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


show_sampling_times = true
pre_C_choice_learned = true
switch_time = 0.0
g0.υ = 3.0

Tmax = 1500
Δt = 30 # sample every Δt days
t = 0.0;
previous_sample_t = 0
num_samples = 1000

prob = ODEProblem(f!, W, [0, Tmax], g0)
integrator = init(prob, Tsit5(); dt=1//2^4, tstops=0:0.3:Tmax)  

function q_converged(us;ϵ=0.02)
    s = 0.0
    for i in abs.(diff(us[max(end-1000,1):end]))
        s /= 1.01
        s += i
    end
    s<ϵ
end

##
fixed_reward_for_survey      = zeros(size(g0.r_star)) 
fixed_reward_for_survey[1]   = 1.0
fixed_reward_for_survey[end] = -1.0

g0.β_star= beta_bar(5.0)
g0.r_star = g0.c


for i in integrator
    global pre_C_choice_learned
    push!(us, integrator.u[end])
    push!(sols.u,copy(integrator.u))
    push!(sols.t,integrator.t)
    
    #sample from the current choice function value
    if EstimateFrom == 2
        #ii) sample from the choice function
        x = [C_logit_payoff(g0,fixed_reward_for_survey,0.0,i;η=myPBR_η_real)  for i=1:g0.NS ]
    else
        throw("EstimateFrom=$EstimateFrom has not been implemented")
    end

    if (previous_sample_t+Δt < integrator.t) && pre_C_choice_learned
        rr = [ fixed_reward_for_survey[j] for j=1:g0.NS]
        p = Categorical(x)
        local C = mean(map(x->x.==(1:g0.NS), rand(p,num_samples) ))
        push!(D,(C,rr))
   

        meanR = mean(map(x->x[2]'*x[1],D))

        @show probability,range_η = find_parameter_range(g0,fixed_reward_for_survey,meanR,length(D)*num_samples;ϵ=0.05)

        if probability>0.95
            fixall_PBR!(g0;PBR_η=mean(range_η))
            pre_C_choice_learned = false
            u_modified!(integrator,true)
        end


        #= ##old approach
        l(η;r) = begin
            s = exp.(inv(abs(η[1]))*r)        
            s/sum(s)
        end 
        J(η) = sum( norm(l(η;r=d[2])-d[1], 1)  for d in D) + 0.00001*norm(η)
        
        myPBR_η_estimated = optimize(J,[2.0], autodiff = :forward).minimizer[1]
        global myPBR_η_estimated = abs(myPBR_η_estimated)
        fixall_PBR!(g0;PBR_η=myPBR_η_estimated)
        
        pre_C_choice_learned = false
        u_modified!(integrator,true)
        =#

        @show integrator.t
        global switch_time = integrator.t
        global previous_sample_t = integrator.t
        
        push!(ts,integrator.t)
    end

end

ts = [ts[end]]
y = hcat(sols.u...)

# plots the ratio of infected 
X = mapslices(x->[x;1.0-sum(x)], y[xi(g0,1:g0.NS-1),:], dims=1)
i_star = (g0.η*(1-g0.σ/g0.β_star))
I_ratio = (y[1,:]./(g0.β'*X)')./(i_star)
Plots.plot(sols.t, I_ratio, label="I,υ=$(g0.υ)")
if show_sampling_times 
    for switch_time in ts
        plot!([switch_time,switch_time],[minimum(I_ratio),maximum(I_ratio)], linestyle=:dash, label=false)
    end
end
ylabel!("I/I*")
xlabel!("Days")
plot!(size=plt_size)
savefig("images/as_it_goes_EstimateFrom$(EstimateFrom)_logit_infected_$(g0.c_star)_nu$(round(g0.υ,digits=1)).$(plt_ext)")



# plots the ratio of agents using the first strategy
plot(sols.t, y[3,:], label="x_1,υ=$(g0.υ)")
if show_sampling_times 
    for switch_time in ts
        plot!([switch_time,switch_time],[minimum(y[3,:]),maximum(y[3,:])], linestyle=:dash, label=false)
    end
end
ylabel!("x_1")
xlabel!("Days")
plot!(size=plt_size)
yaxis!((0.2,1.0))
savefig("images/as_it_goes_EstimateFrom$(EstimateFrom)_logit_strats_$(g0.c_star)_nu$(round(g0.υ,digits=1)).$(plt_ext)")


# plots the state of the dynamic payoff
plot(sols.t, y[4,:], label="q,υ=$(g0.υ)")
if show_sampling_times 
    for switch_time in ts
        plot!([switch_time,switch_time],[minimum(y[4,:]),maximum(y[4,:])], linestyle=:dash, label=false)
    end
end
plot!(size=plt_size)
savefig("images/as_it_goes_EstimateFrom$(EstimateFrom)_logit_qzoomout_$(g0.c_star)_nu$(round(g0.υ,digits=1)).$(plt_ext)")
ylims!(-2,2)
plot!(size=plt_size)
savefig("images/as_it_goes_EstimateFrom$(EstimateFrom)_logit_qzoomin_$(g0.c_star)_nu$(round(g0.υ,digits=1)).$(plt_ext)")

