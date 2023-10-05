
import Dates   
using Optim
using DifferentialEquations
using StatsPlots
using DataFrames
using CSV

include(("state_data/state.jl"))



PI_B = 1000
PI_H = 30
MU_M = 1/42
MU_B = 1/1000
MU_H = 1/(70*365) #Reciprocal of length of human life in days
q = 0.33
c = 0.9
b_1 = 0.09 #adjusts peak
b_2 = 0.09*(1-c*q)
BETA_1 = 0.16 #adjusts peak and end value
BETA_2 = 0.88 #adjusts peak
BETA_3 = 0.88 #adjusts peak
d_B = 5 * 10^-5 #no noticable difference
d_H = 5 * 10^-7 #no noticable difference
ALPHA = 1/14 #Reciprocal of Incubation Period in 1/Days, Shifts end value
DELTA = 1 #adjusts peak
TAU = 1/14 #No noticable difference

state = "CA"
year = 2013

odedata = State.load(state)
odedata = filter(:date => x -> Dates.year(x) == year, odedata)
odedata = odedata[6:12, :]

population = State.getPopulation(state, year)

function wnv(du, u, p, t)

    V_0, V_1, V_2 = p[1], p[2], p[3]
    #Model parameters
    M_i = V_0*sin(V_1 + 2*pi*t/V_2) #Infected mosquitoes
    #Current state
    S, E, I = u

    N_H = S + E + I
    b_2 = 0.09*(1-c*q)

    #Evaluate differential equations

    du[1] = PI_H - (b_2*BETA_3*M_i*S)/(N_H) - MU_H*S #S, susceptible humans
    du[2] = (b_2*BETA_3*M_i*S)/(N_H) - MU_H*E - E*ALPHA #E, asymptomatically infected humans
    
    du[3] = E*ALPHA - MU_H*I - DELTA*I#I, symptomatically infected humans
    
    return nothing
end

function diffEqError(x)
    #set parameters and initial conditions
    if(x[1] < 0)
        return 1000000
    end
    u0 = [39000000, 0, 0]
    p = [x[1], x[2], x[3]]
    t_end = 180
    tspan = (0.0, t_end)
    prob = ODEProblem(wnv, u0, tspan, p)
    sol = solve(prob, Rodas5(), saveat=1, dt=1e-6)
    realsol = []

    push!(realsol, 0)
    push!(realsol, sum(sol[2, :30])*ALPHA)
    push!(realsol, sum(sol[2, 31:60])*ALPHA)
    push!(realsol, sum(sol[2, 61:90])*ALPHA)
    push!(realsol, sum(sol[2, 91:120])*ALPHA)
    push!(realsol, sum(sol[2, 121:150])*ALPHA)
    push!(realsol, sum(sol[2, 151:180])*ALPHA)

    return sum(abs.(realsol .- odedata[!, :count]))
end


#=
#set parameters and initial conditions
u0 = [39000000, 0, 0]
tspan = (0.0, 181)
p = [10.0, 2.0, 1.0]
prob = ODEProblem(wnv, u0, tspan, p)

sol = solve(prob, Rodas5(), saveat=1, dt=1e-6)

plot(sol[3, :], label="Predicted")
plot!(odedata[!, :count], label="Observed")

=#


x = [10000.0, 2.0, 360.0]

#res = optimize(error, x, (), Optim.Options(iterations=1000, store_trace=true))

lbounds = [0.0, 0.0, 0.0]
ubounds = [200000.0, 10.0, 1000.0]

#result = optimize(diffEqError, lbounds, ubounds, x, NelderMead())

#result = optimize(diffEqError, lbounds, ubounds, x)
#result = optimize(diffEqError, lbounds, ubounds, x, Fminbox(GradientDescent()))

#result = optimize(diffEqError, x, SimulatedAnnealing())

#result = optimize(diffEqError, x, ParticleSwarm(;lower=lbounds, upper=ubounds, n_particles=15))
#result = optimize(diffEqError, x, ParticleSwarm(;lower=[], upper=[], n_particles=15))
result = optimize(diffEqError, x, NelderMead())

print(result.minimizer)

#evaluate results
u0 = [39000000, 0, 0]
p = [result.minimizer[1], result.minimizer[2], result.minimizer[3]]
t_end = 180
tspan = (0.0, t_end)
prob = ODEProblem(wnv, u0, tspan, p)
sol = solve(prob, Rodas5(), saveat=1, dt=1e-6)
realsol = []

push!(realsol, 0)
push!(realsol, sum(sol[2, :30])*ALPHA)
push!(realsol, sum(sol[2, 31:60])*ALPHA)
push!(realsol, sum(sol[2, 61:90])*ALPHA)
push!(realsol, sum(sol[2, 91:120])*ALPHA)
push!(realsol, sum(sol[2, 121:150])*ALPHA)
push!(realsol, sum(sol[2, 151:180])*ALPHA)

plot(realsol, label="Predicted")
plot!(odedata[!, :count], label="Observed")
savefig("[PREDICT]CA"*string(year)*".png")