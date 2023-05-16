using DifferentialEquations
using Plots, StatsPlots
using Optim, LinearAlgebra, Distributions
using JLD2
using ProgressBars
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




##

game = SIRS_Game(2,fp)

game.x_star
game.σ = 0.1
game.ω = 0.005
game.γ = game.σ
game.υ = 2.0
game.β   = [0.15;0.19]
game.c   = [0.2;0.0]
game.c_star = 0.05
game.ρ = 0.0
game.υ = 3.0

myPBR_η_real = 0.5
myPBR_η_estimated = 0.01
f!(du,u,p,t) = h_logit!(du,u,p,t;η=myPBR_η_real,α=0.1)

fixall_PBR!(game;PBR_η=myPBR_η_estimated)


# for the bound to be meaningful we need to 
#    have S(x(0),p(0)) = 0  
Infected = Ib(game,game.β[1])
Recovered = Rb(game,game.β[1])

function simulation(game,I,R,x1,q)

    myPBR_η_estimated = 0.01 #highly rational
    #myPBR_η_estimated = 10.0 #highly irrational

    g0 = deepcopy(game)

    fixall_PBR!(g0;PBR_η=myPBR_η_estimated)

    S = 1.0-I-R

    W = [Bx(game,[0;0;x1;0])*I;Bx(game,[0;0;x1;0])*R;x1;q]

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


    Tmax = 3000
    Δt = 7 # sample every Δt days
    k = ceil(1200/Δt) #number of sampling times
    t = 0.0;
    previous_sample_t = 0
    num_samples = 1000

    prob = ODEProblem(f!, W, [0, Tmax], g0)
    integrator = init(prob, Tsit5(); dt=1//2^5, tstops=0:0.3:Tmax)  

    g0.r_star = LinearAlgebra.I(g0.NS)[:,2].*g0.c_star

    function q_converged(us;ϵ=0.02)
        s = 0
        for i in abs.(diff(us[max(end-1000,1):end]))
            s /= 1.01
            s += i
        end
        s<ϵ
    end

    ##

    small_val = 0.02
    F_hist = []
    F_old = 0.0
    v_old = zeros(g0.NS)

    r_good = zeros(g0.NS)
    F_good = Inf
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



        # @show fp(g0,sols.u[end],0.0,1:2)'*(x->[x;1-sum(x)])(sols.u[end][3])

        
        if integrator.t>(Tmax*0.8) && pre_C_choice_learned
            #g0.r_star .= r_good
            pre_C_choice_learned = false
            #u_modified!(integrator,true)
        end

        if integrator.t>(Δt*k) && pre_C_choice_learned && integrator.t<(Tmax*0.8)
            x = integrator.u[3]
            x = (y->[y;1-sum(y)])(x)
            #F_new = (fp(g0,sols.u[end],0.0,1:2)'*x)^2 + (g0.β'*x-g0.β_star)^2 + sum(-min.(g0.r_star,0.0)) 
            F_new = Float64(integrator.u[end]^2 + sum(-min.(g0.r_star,0.0))^2) 
            
            push!(F_hist, F_new)

            g0.r_star .= g0.r_star - v_old
            g0.r_star .= g0.r_star - 0.02*v_old.*(F_new-F_old)/small_val
            v_old = (rand(g0.NS).-0.5)
            v_old = v_old/norm(v_old)
            v_old = v_old*small_val
            
            g0.r_star .= g0.r_star + v_old
            
            if F_good> F_new 
                r_good .= g0.r_star
                F_good = F_new
            end
            F_old = F_new


            u_modified!(integrator,true)
            k = k+1.0
            # pre_C_choice_learned = false
            # @show integrator.t
            #global switch_time = integrator.t
        end

    end
    switch_time = Tmax*0.8

    (F_hist=F_hist, sols=sols, params=(I=I,R=R,x1=x1,q=q, ), )
end


simulation(game,Infected,Recovered,1.0,0.0)

##


##

try load("example.jld2", "results")
    print("did load data")
    global results = load("example.jld2", "results")
    global N_runs = sum(length, results)
catch

    global results = Array{Vector}(undef,Threads.nthreads())
    for i in eachindex(results)
        results[i] = []
    end

    global N_runs = 10000
    Threads.@threads for i = ProgressBar(1:N_runs)
        local I,R
        I  = rand()
        R  = rand()*(1-I)
        x1 = rand()
        q  = 10*(rand()-0.5)
        
        myid = Threads.threadid()
        push!(results[myid], simulation(game,I,R,x1,q));

    end
    jldsave("example.jld2"; results)

end



D = cat(results...,dims=1)
G = [ d.F_hist for d in D]
GG = hcat(G...)

errorline(GG, errorstyle=:plume, label="avg" )
plot!(maximum(GG,dims=2),label="max",color=:red)
plot!(minimum(GG,dims=2),label="min",color=:green)
title!("cost, runs=$(N_runs)")
savefig("cost.N$(N_runs).png")
##

histogram(GG[1,:], title="initial cost, runs=$(N_runs)")
savefig("histogram.init.cost.N$(N_runs).png")
##

histogram(GG[end,:], title="final cost, runs=$(N_runs)")
savefig("histogram.final.cost.N$(N_runs).png")
##

G = [ d.sols.u for d in D]
GG = map(x->x[end][1], G)
histogram(GG, title="q(0), runs=$(N_runs)")
savefig("histogram.init.q.N$(N_runs).png")
##

G = [ d.sols.u for d in D]
GG = map(x->x[end][end], G)
histogram(GG, title="q(T), runs=$(N_runs)")
savefig("histogram.final.q.N$(N_runs).png")
##



Gu = [ d.sols.u for d in D]
Gt = [ d.sols.t for d in D]


Gt = map(x->x[1:minimum(length.(Gt))],Gt)
Gu = map(x->x[1:minimum(length.(Gt))],Gu)

errorline(Gt[1],map(x->x[1], hcat(Gu...)), errorstyle=:plume, title="I_hat(t), runs=$(N_runs)" )
savefig("I_hat.N$(N_runs).png")
##
ylims!(0,0.01)
savefig("I_hat_zoom.N$(N_runs).png")
##

errorline(Gt[1],map(x->x[2], hcat(Gu...)), errorstyle=:plume, title="R_hat(t), runs=$(N_runs)" )
savefig("R_hat.N$(N_runs).png")
##

errorline(Gt[1],map(x->x[3], hcat(Gu...)), errorstyle=:plume, title="x_1(t), runs=$(N_runs)" )
savefig("x_1.N$(N_runs).png")
##

errorline(Gt[1],map(x->x[4], hcat(Gu...)), errorstyle=:plume, title="q(t), runs=$(N_runs)" )
savefig("q.N$(N_runs).png")
##




##
1

##
2

##
3

##
4

##
5

##



plot(F_hist[1:end])
##

y = hcat(sols.u...)

# plots the ratio of infected 
X = mapslices(x->[x;1.0-sum(x)], y[xi(g0,1:g0.NS-1),:], dims=1)
i_star = (g0.η*(1-g0.σ/g0.β_star))
I_ratio = (y[1,:]./(g0.β'*X)')./(i_star)
Plots.plot(sols.t, I_ratio, label="I,υ=$(g0.υ)")
plot!([switch_time,switch_time],[minimum(I_ratio),maximum(I_ratio)], linestyle=:dash, label=false)
ylabel!("I/I*")
xlabel!("Days")

##
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

