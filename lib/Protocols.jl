include("SIRS_Game.jl")


#this file contains data structures and functions used by the main code 


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
function smith(g::SIRS_Game,x::Vector{},t,i; λ=1.0, τ=0.0 )
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
`logit(g::SIRS_Game,x::Vector{Float64},t::Float64,i) -> Float`

logit dynamics. Takes parameters `g` (an SIRS Game), `x` (a vector with the current state, 
for both epidemic and game), `t` (time), and `i` (strategy).


# Examples
```julia
julia> logit(SIRS_Game(2,(g,x,t,j)->[1,0][j],[4.0,1.0],2.0 ,1.0), zeros(10), 1.0, 1)
0.0
```
""" 
function logit(g::SIRS_Game,x::Vector{},t,i; η=1.0)
    mysum = 0
    for j ∈ 1:(g.NS)
        mysum += exp(inv(η)* g.F(g,x,t,j))
        #@show exp(inv(η)* g.F(g,x,t,j)), η
    end
    mysum =  exp(inv(η)*g.F(g,x,t,i))/mysum
    mysum -=  x[xi(g,i)]
    
    mysum
end



function C_logit(g::SIRS_Game,x::Vector{},t,i; η=1.0)
    mysum = 0
    for j ∈ 1:(g.NS)
        mysum += exp(inv(η)* g.F(g,x,t,j))
        #@show exp(inv(η)* g.F(g,x,t,j)), η
    end
    mysum =  exp(inv(η)*g.F(g,x,t,i))/mysum
    
end






