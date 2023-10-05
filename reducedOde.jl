
import Dates   
using Optim
using DifferentialEquations
using StatsPlots
using DataFrames
using CSV

include(("state_data/state.jl"))


#params
#PI_M = variable
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

#=
population counts:
M_u = Uninfected mosquitoes
M_i = Infected mosquitoes
B_u = Uninfected birds
B_i = Infected birds
S = Susceptible
E = Asymptomatically Infected
I = Symptomatically Infected
H = Hospitalized Patients
R = Recovered

N_M = M_u + M_i
N_B = B_u + B_i
N_H = S + E + I
=#

#=
for state in State.listOfStates()
    if state[begin] < 'N' or 
        continue
    end
=#

state = "CA"
year = 2017

odedata = State.load(state)
odedata = filter(:date => x -> Dates.year(x) == year, odedata)
odedata = odedata[6:12, :]

population = State.getPopulation(state, year)

function wnv(du, u, p, t)

    V_O, V_1 = p[1]
    #Model parameters
    M_i = V_O*(1+0.887*sin(2*pi*t/180))
    #Current state
    S, E, I = u

    N_H = S + E + I
    b_2 = 0.09*(1-c*q)

    #Evaluate differential equations

    du[1] = PI_H - (b_2*BETA_3*M_i*S)/(N_H) - MU_H*S #susceptible humans
    du[2] = (b_2*BETA_3*M_i*S)/(N_H) - MU_H*E - ALPHA*E #asymptomatically infected humans
    
    du[3] = ALPHA*E - MU_H*I - DELTA*I#symptomatically infected humans


    return nothing
end

function diffEqError(x)
    #set parameters and initial conditions
    u0 = [39000000, 0, 0]
    p = [x[1]]
    t_end = 180
    tspan = (0.0, t_end)
    prob = ODEProblem(wnv, u0, tspan, p)
    sol = solve(prob, Rodas5(), saveat=1, dt=1e-6)
    sol_pred = [sol[3,1], sol[3, trunc(Int,1/6 * t_end)], sol[3, trunc(Int, 2/6*t_end)], sol[3, trunc(Int, 3/6*t_end)], sol[3, trunc(Int, 4/6*t_end)], sol[3, trunc(Int, 5/6*t_end)], sol[3, trunc(Int, t_end)]]
    #plot(sol_pred, label="Predicted")
    #plot!(odedata[!, :count], label="Observed")
    return sum(abs.(sol_pred .- odedata[!, :count]))
end


#=
#set parameters and initial conditions
u0 = [39000000, 0, 0]
tspan = (0.0, 181)
p = [0.0]
prob = ODEProblem(wnv, u0, tspan, p)

sol = solve(prob, Rodas5(), saveat=1, dt=1e-6)

plot(sol[3, :], label="Predicted")
plot!(odedata[!, :count], label="Observed")


=#

x = [100000.0]

#res = optimize(error, x, (), Optim.Options(iterations=1000, store_trace=true))

lbounds = [0.0]
ubounds = [20000000.0]

result = optimize(diffEqError, lbounds, ubounds, x, NelderMead())

#result = optimize(diffEqError, lbounds, ubounds, x)
#result = optimize(diffEqError, lbounds, ubounds, x, Fminbox(GradientDescent()))

#result = optimize(diffEqError, x, SimulatedAnnealing())

#result = optimize(diffEqError, x, ParticleSwarm(;lower=lbounds, upper=ubounds, n_particles=15))
#result = optimize(diffEqError, x, ParticleSwarm(;lower=[], upper=[], n_particles=15))
#result = optimize(diffEqError, x, NelderMead())


#evaluate results

u0 = [39000000, 0, 0]
p = [result.minimizer[1]]
t_end = 181
tspan = (0.0, t_end)
prob = ODEProblem(wnv, u0, tspan, p)
sol = solve(prob, Rodas5(), saveat=1, dt=1e-4)
plot(sol[3, :], label="Predicted")

sol_pred = [sol[3, 1], sol[3, 31], sol[3, 61], sol[3, 91], sol[3, 121], sol[3, 151], sol[3, 181]]
plot(sol_pred, label="Predicted")

plot!(odedata[!, :count], label="Observed")

#savefig("[PREDICT]CA"*string(year)*".png")
print(result.minimizer)

