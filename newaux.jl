import JuMP
import GLPK
#this file contains data structures and functions used by the main code 


mutable struct SIRS_Game
    NS::Int             #number of strategies
    q::Vector{Float64} #Dynamic payoff state
    #parameters of the epidemic
    σ::Float64
    ω::Float64
    γ::Float64
    υ::Float64
    η::Float64
    β::Vector{Float64}
    β_star::Float64
    x_star::Vector{Float64}
    c::Vector{Float64}
    c_star::Float64
    r_star::Vector{Float64}
    ρ::Float64

    #payoff function
    F
    
    function SIRS_Game(ns::Int, f, b, g, om, x_star) 
        F  = f
        NS = ns
        σ = 0.000001
        ω = om
        γ = g
        η = ω/(ω+γ)
        υ = 0
        η = ω/(ω+γ)
        β = b
        β_star = β'*x_star
        new(NS,[0.0],σ,ω,γ,υ,η,β,β_star,x_star,[0.0],0.0,[0.0],0.0,F)
    end
    #simpler way of creating a SIRS_Game structure instance
    function SIRS_Game(ns::Int, f) 
        F  = f
        NS = ns
        new(NS,[0.0],0.0,0.0,0.0,0.0,0.0,[0.0,0.0],0.0,[0.0],[0.0],0.0,[0.0],0.0,F)
    end
end


"""
`fixall!(g::SIRS_Game) -> nothing`

Get the SIRS_Game `g` to calculate and update η, β* and r*. Call it whenever you change any parameters in g

# Examples
```julia
julia> g = SIRS_Game(2,x->[2;2])
julia> fixall!(g)
```
""" 
function fixall!(g::SIRS_Game)
    fix_η!(g)
    fix_βx_star!(g)
    fix_r_star!(g)
end



"""
`fix_η!(g::SIRS_Game) -> nothing`

Get the SIRS_Game `g` to calculate and update η. Use `fixall!` instead of calling this directly
# Examples
```julia
julia> g = SIRS_Game(2,x->[2;2])
julia> fix_η!(g)
```
""" 
function fix_η!(g::SIRS_Game)
    g.η = g.ω/(g.ω+g.γ)
end


"""
`fix_βx_star!(g::SIRS_Game) -> nothing`

Get the SIRS_Game `g` to calculate and update β*. Use `fixall!` instead of calling this directly
# Examples
```julia
julia> g = SIRS_Game(2,x->[2;2])
julia> fix_βx_star!(g)
```
""" 
function fix_βx_star!(g::SIRS_Game)
    m = JuMP.Model(GLPK.Optimizer)
    JuMP.@variable(m, x[1:size(g.β,1)] >= 0)
    JuMP.@constraint(m, sum(x) == 1 )
    JuMP.@constraint(m, (g.c.-minimum(g.c))'*x<=g.c_star )
    JuMP.@objective(m, Min, g.β'*x )

    JuMP.optimize!(m)

    g.β_star = g.β'*JuMP.value.(x)
    g.x_star = JuMP.value.(x)
end


"""
`fix_r_star!(g::SIRS_Game) -> nothing`

Get the SIRS_Game `g` to calculate and update r*. Use `fixall!` instead of calling this directly
# Examples
```julia
julia> g = SIRS_Game(2,x->[2;2])
julia> fix_r_star!(g)
```
""" 
function fix_r_star!(g::SIRS_Game)
    cc = g.c.-minimum(g.c)

    g.r_star = [ g.x_star[i]≈0 ? cc[i]-g.ρ : cc[i] for i=1:size(g.β,1)]
end




"""
`r(strategy,index::Int) -> Int`

# Examples
```julia
julia> r(2,1)
6
```
""" 
function r(strategy,index::Int)
    5*(strategy-1)+index
end



"""
`ei(index::String) -> Int`

# Examples
```julia
julia> ei(g,"I")
2
```
""" 
function ei(g::SIRS_Game,index::String)
    if(index=="S" || index=="s")  
        0 #do not use this!
    end
    if(index=="I" || index=="i")  
        2
    end
    if(index=="R" || index=="r")  
        3
    end
end



"""
`xi(g::SIRS_Game,index::Int) -> Int`

# Examples
```julia
julia> xi(g,3)
5
```
""" 
function xi(g::SIRS_Game,index)
    2 .+index
end


"""
`qi(g::SIRS_Game,index::Int) -> Int`

# Examples
```julia
julia> qi(g,3)
5
```
""" 
function qi(g::SIRS_Game,index)
    2+g.NS-1 .+index
end


"""
`Bx(g::SIRS_Game,u) -> Float64`

# Examples
```julia
julia> g = SIRS_Game(2,x->[2;2])
julia> g.β .= [8.0,1.0] 
julia> Bx(g,[1.0,0.0,0.5,0.0])
4.5
```
""" 
function Bx(g::SIRS_Game,u)
    # u is the state
    x = xi(g,1:(g.NS-1)) 
    x = u[x]
    x = [x;1-sum(x)] 

    g.β'*x
end


function It(g,u)
    #g #gamestruct
    #u #state
    
    Ib(g,Bx(g,u))
end

function Rt(g,u)
    #g #gamestruct
    #u #state
    
    Rb(g,Bx(g,u))
end

function Ib(g,B)
    #g #gamestruct
    #u #state
    g.η*(1-g.σ/B)
end

function Rb(g,B)
    #g #gamestruct
    #u #state
    (1-g.η)*Ib(g,B)/g.η
end







"""
`smith(g::SIRS_Game,x::Vector{Float64},t::Float64,i) -> Float`

Smith dynamics. Takes parameters `g` (an SIRS Game), `x` (a vector with the current state, 
for both epidemic and game), `t` (time), and `i` (strategy).


# Examples
```julia
julia> smith(SIRS_Game(2,(g,x,t,j)->[1,0][j],[4.0,1.0],2.0 ,1.0), zeros(10), 1.0, 1)
0.0
```
""" 
function smith(g::SIRS_Game,x::Vector{Float64},t::Float64,i; λ=1.0, τ=0.0 )
    mysum = 0.0
    for j ∈ 1:(g.NS-1)
        mysum +=  x[xi(g,j)]*max(min(λ*(g.F(g,x,t,i)-g.F(g,x,t,j)), τ),0.0)
    end
    mysum += (1.0-sum(x[xi(g,1:(g.NS-1))]) )*max(min(λ*(g.F(g,x,t,i)-g.F(g,x,t,g.NS)), τ),0.0)
    
    for j ∈ 1:g.NS
        mysum += -x[xi(g,i)]*max(min(λ*(g.F(g,x,t,j)-g.F(g,x,t,i)), τ),0.0)
    end
    
    mysum
end



"""
`fp(g,x,t,j) -> Float`
payoff for the different strategies. Particular to this code


# Examples
```julia
julia> g = SIRS_Game(2,(g,x,t,j)->[1,0][j],[4.0,1.0],2.0 ,1.0)
julia> g.ζ .= [8.0,6.0] 
julia> g.ϑ .= [10.0,4.0]
julia> g.θ .= [1.0,8.0] 
julia> g.ν .= [0.5,3.0] 
julia> g.ξ .= [1.0,1.0] 
julia> fp(g,repeat([0.9,0.1,0.0,0.5,0.0],2),0.0,2)
-10.2
julia> fp(g,repeat([0.9,0.1,0.0,0.5,0.0],2),0.0,2)
-6.859999999999999
```
""" 
function fp(g,x,t,j)
    (g.β*x[qi(g,1)]+g.r_star-g.c)[j]#(g0.β*x[qi(g,1)])[j] #+[-1.0 0; 0.0 -1.0]*x[3:4]+[0.01;.999])[j]
end


"""
`h!(du,u,p,t) -> Array{Float}`

total dynamics of our model

""" 
function h!(du,u,p,t)

    for i ∈ 1:(p.NS-1)
        # game dynamic
        du[xi(p,i)] = smith(p,u,t,i;λ=0.1,τ=0.1)
    end


    #make sure that x>=0 and ones(3)'*x == 1 
    x = u[xi(p,1:(p.NS-1))]
    x = [x;1.0-sum(x)]
    x = max.(x,0.0)
    x = x/sum(x)

    if any(u[xi(p,1:(p.NS-1))].>1) || any(u[xi(p,1:(p.NS-1))] .<0)
        u[xi(p,1:(p.NS-1))] .= x[1:(end-1)]
    end


    B = p.β'*x
    
    I = max(u[1],0)
    R = max(u[2],0)
    if u[1]<0
        u[1] = I
    end
    if u[2]<0
        u[2] = R
    end
    
    I_hat = p.η*(B-p.σ)
    R_hat = (1-p.η)*(B-p.σ)

    g = (I_hat-I)/B+p.η*log(I/I_hat)+(R-R_hat)*(1-p.η-R/B)/p.γ+p.υ^2*(p.β_star-B)
    #g = p.η*log(I/I_hat)*(1-p.η)*(R-R_hat)/p.γ+p.ν*(p.β_star-B)
    #println(I,"    ",I_hat,"    ",p.β,"    ",u[3:4])

    # payoff dynamics
    du[qi(p,1)] = g

    x_dot = du[xi(p,1:(p.NS-1))]
    x_dot = [x_dot; -sum(x_dot)]
    B_dot = x_dot'*p.β

    # epidemic dynamic
    du[1] = I*(I_hat+R_hat-I-R)+B_dot*I/B
    du[2] = p.ω*(R_hat-R)-p.γ*(I_hat-I)+B_dot*R/B
end


