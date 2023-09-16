
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
year = 2014

odedata = State.load(state)
odedata = filter(:date => x -> Dates.year(x) == year, odedata)
odedata = odedata[6:12, :]

population = State.getPopulation(state, year)

function wnv(du, u, p, t)
    #Model parameters
    PI_M = p[1]
    #Current state
    M_u, M_i, B_u, B_i, S, E, I, H, R = u

    N_M = M_u + M_i
    N_B = B_u + B_i
    N_H = S + E + I
    b_2 = 0.09*(1-c*q)

    #Evaluate differential equations
    du[1] = PI_M - (b_1*BETA_1*M_u*B_i)/N_B - MU_M*M_u #uninfected mosquitoes
    du[2] = (b_1*BETA_1*M_u*B_i)/(N_B) - MU_M*M_i #infected mosquitoes

    du[3] = PI_B - (b_1*BETA_2*M_i*B_u)/(N_B) - MU_B*B_u #uninfected birds
    du[4] = (b_1*BETA_2*M_i*B_u)/(N_B) - MU_B*B_i - d_B*B_i #infected birds

    du[5] = PI_H - (b_2*BETA_3*M_i*S)/(N_H) - MU_H*S #susceptible humans
    du[6] = (b_2*BETA_3*M_i*S)/(N_H) - MU_H*E - ALPHA*E #asymptomatically infected humans
    
    du[7] = ALPHA*E - MU_H*I - DELTA*I#symptomatically infected humans
    
    du[8] = DELTA*I - TAU*H - MU_H*H - d_H*H #hospitalized humans
    du[9] = TAU*H - MU_H*R #recovered humans

    return nothing
end

function diffEqError(x)
    #set parameters and initial conditions
    u0 = [499*x[1], x[1], x[2], x[3], population, 0, 0, 0, 0]
    p = [x[4]]
    t_end = 181
    tspan = (0.0, t_end)
    prob = ODEProblem(wnv, u0, tspan, p)
    sol = solve(prob, Rodas5(), saveat=1, dt=1e-6)
    sol_pred = [sol[7,1], sol[7, trunc(Int,1/6 * t_end)], sol[7, trunc(Int, 2/6*t_end)], sol[7, trunc(Int, 3/6*t_end)], sol[7, trunc(Int, 4/6*t_end)], sol[7, trunc(Int, 5/6*t_end)], sol[7, trunc(Int, t_end)]]
    #plot(sol_pred, label="Predicted")
    #plot!(odedata[!, :count], label="Observed")
    return sum(abs.(sol_pred .- odedata[!, :count]))
end

x = [5000.0, 50000.0, 1000.0, 1200]

#res = optimize(error, x, (), Optim.Options(iterations=1000, store_trace=true))

lbounds = [0.0, 0.0, 0.0, 0.0]
ubounds = [10000.0, 1000000.0, 20000.0, 20000.0]

#result = optimize(diffEqError, lbounds, ubounds, x)
#result = optimize(diffEqError, lbounds, ubounds, x, Fminbox(GradientDescent()))

#result = optimize(diffEqError, x, SimulatedAnnealing(), Optim.Options(iterations=10000))

#result = optimize(diffEqError, x, ParticleSwarm(;lower=lbounds, upper=ubounds, n_particles=15))
#result = optimize(diffEqError, x, ParticleSwarm(;lower=[], upper=[], n_particles=15))
result = optimize(diffEqError, x, NelderMead())


#evaluate results

u0 = [499*result.minimizer[1], result.minimizer[1], result.minimizer[2], result.minimizer[3], population, 0, 0, 0, 0]
p = [result.minimizer[4]]
t_end = 181
tspan = (0.0, t_end)
prob = ODEProblem(wnv, u0, tspan, p)
sol = solve(prob, Rodas5(), saveat=1, dt=1e-4)
plot(sol[7, :], label="Predicted")

sol_pred = [sol[7, 1], sol[7, 31], sol[7, 61], sol[7, 91], sol[7, 121], sol[7, 151], sol[7, 181]]
plot(sol_pred, label="Predicted")

plot!(odedata[!, :count], label="Observed", show=true)


savefig("NelderMeadCA"*string(year)*".png")
print(result.minimizer)

