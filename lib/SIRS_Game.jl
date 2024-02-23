import JuMP
import GLPK, Ipopt

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
`fixall_IPC!(g::SIRS_Game) -> nothing`

Get the SIRS_Game `g` to calculate and update η, β* and r*. Call it whenever you change any parameters in g

# Examples
```julia
julia> g = SIRS_Game(2,x->[2;2])
julia> fixall_IPC!(g)
```
""" 
function fixall_IPC!(g::SIRS_Game)
    fix_η!(g)
    fix_βx_star!(g)
    fix_r_star_IPC!(g)
end

"""
`fixall_PBR!(g::SIRS_Game) -> nothing`

Get the SIRS_Game `g` to calculate and update η, β* and r*. Call it whenever you change any parameters in g

# Examples
```julia
julia> g = SIRS_Game(2,x->[2;2])
julia> fixall_PBR!(g)
```
""" 
function fixall_PBR!(g::SIRS_Game; PBR_η)
    fix_η!(g)
    fix_r_star_PBR!(g; PBR_η)
    
    C(r) = begin
        s = exp.((g.r_star-g.c)/PBR_η)
        s/sum(s)
    end

    #fix_βx_star!(g)
    g.β_star = g.β'*C(g.r_star)
    g.x_star = C(g.r_star)

end


"""
`fixall_PBR_estimated!(g::SIRS_Game) -> nothing`

Get the SIRS_Game `g` to calculate and update η, β* and r*. Call it whenever you change any parameters in g

# Examples
```julia
julia> g = SIRS_Game(2,x->[2;2])
julia> fixall_PBR_estimated!(g, estimated_C=r->exp.(r)./sum(exp,r) )
```
""" 
function fixall_PBR_estimated!(g::SIRS_Game; estimated_C)
    fix_η!(g)
    fix_r_star_PBR_estimated!(g; estimated_C=estimated_C)
    

    #fix_βx_star!(g)
    g.β_star = g.β'*estimated_C(g.r_star)
    g.x_star = estimated_C(g.r_star)

end


"""
`fixall_PBR_givenC!(g::SIRS_Game, ChoiceFun) -> nothing`

Get the SIRS_Game `g` to calculate and update η, β* and r*. Call it whenever you change any parameters in g

# Examples
```julia
julia> g = SIRS_Game(2,x->[2;2])
julia> fixall_PBR!(g)
```
""" 
function fixall_PBR_givenC!(g::SIRS_Game, ChoiceFun)
    fix_η!(g)
    fix_r_star_PBR_givenC!(g, ChoiceFun)

    #fix_βx_star!(g)
    g.β_star = g.β'*ChoiceFun(g.r_star)
    g.x_star = ChoiceFun(g.r_star)

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
`fix_r_star_IPC!(g::SIRS_Game) -> nothing`

Get the SIRS_Game `g` to calculate and update r*. Use `fixall!` instead of calling this directly
# Examples
```julia
julia> g = SIRS_Game(2,x->[2;2])
julia> fix_r_star_IPC!(g)
```
""" 
function fix_r_star_IPC!(g::SIRS_Game)
    cc = g.c.-minimum(g.c)

    g.r_star = [ g.x_star[i]≈0 ? cc[i]-g.ρ : cc[i] for i=1:size(g.β,1)]
end


"""
`fix_r_star_PBR!(g::SIRS_Game) -> nothing`

Get the SIRS_Game `g` to calculate and update r*. Use `fixall!` instead of calling this directly
# Examples
```julia
julia> g = SIRS_Game(2,x->[2;2])
julia> fix_r_star_IPC!(g)
```
""" 
function fix_r_star_PBR!(g::SIRS_Game; PBR_η)

    m = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_silent(m) 
    JuMP.@variable(m, r[1:g.NS]>=0)

    
    C(r) = begin
        s = exp.((r-g.c)/PBR_η)

        s/sum(s)
    end
    
    F(r...) = begin
        (collect(r)'*C(collect(r)))
    end
    JuMP.register(m, :F, length(r), F; autodiff = true)
    G(r...) = begin
        g.β'*C(collect(r))
    end
    JuMP.register(m, :G, length(r), G; autodiff = true)

    JuMP.@NLconstraint(m, F(r...) <= g.c_star )
    JuMP.@NLobjective(m, Min, G(r...) )

    JuMP.optimize!(m)

    g.r_star = JuMP.value.(r)

    #@show C(g.r_star)
    #@show g.r_star'*C(g.r_star),  g.β'*C(g.r_star)


    g.r_star
end


"""
`fix_r_star_PBR!(g::SIRS_Game, choice_fun) -> nothing`

Get the SIRS_Game `g` to calculate and update r*. Use `fixall!` instead of calling this directly
# Examples
```julia
julia> g = SIRS_Game(2,x->[2;2])
julia> fix_r_star_IPC!(g)
```
""" 
function fix_r_star_PBR_givenC!(g::SIRS_Game, choice_fun)

    m = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_silent(m) 
    JuMP.@variable(m, r[1:g.NS]>=0)

    
    C(r) = choice_fun(r)
    
    F(r...) = begin
        (collect(r)'*C(collect(r)))
    end
    JuMP.register(m, :F, length(r), F; autodiff = true)
    G(r...) = begin
        g.β'*C(collect(r))
    end
    JuMP.register(m, :G, length(r), G; autodiff = true)

    JuMP.@NLconstraint(m, F(r...) <= g.c_star )
    JuMP.@NLobjective(m, Min, G(r...) )

    JuMP.optimize!(m)

    g.r_star = JuMP.value.(r)

    #@show C(g.r_star)
    #@show g.r_star'*C(g.r_star),  g.β'*C(g.r_star)


    g.r_star
end

"""
`fix_r_star_PBR!(g::SIRS_Game) -> nothing`

Get the SIRS_Game `g` to calculate and update r*. Use `fixall!` instead of calling this directly
# Examples
```julia
julia> g = SIRS_Game(2,x->[2;2])
julia> fix_r_star_IPC!(g)
```
""" 
function fix_r_star_PBR_estimated!(g::SIRS_Game; estimated_C)

    m = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_silent(m) 
    JuMP.@variable(m, r[1:g.NS]>=0)

    
    F(r...) = begin
        (collect(r)'*estimated_C(collect(r)))
    end
    JuMP.register(m, :F, length(r), F; autodiff = true)
    G(r...) = begin
        g.β'*estimated_C(collect(r))
    end
    JuMP.register(m, :G, length(r), G; autodiff = true)

    JuMP.@NLconstraint(m, F(r...) <= g.c_star )
    JuMP.@NLobjective(m, Min, G(r...) )

    JuMP.optimize!(m)

    g.r_star = JuMP.value.(r)

    g.r_star
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
