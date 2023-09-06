
import Dates    
using Optim
using DifferentialEquations
using StatsPlots
using Turing



include(("../CA_data/california.jl"))

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

year = 2014
odedata = California.load()
odedata = filter(:date => x -> Dates.year(x) == year, odedata)
odedata = odedata[6:12, :]


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

u0 = [90000, 10000, 5000, 500, 39000000, 0, 0, 0, 0]
tspan = (0.0, 181)
p = [1000]
prob = ODEProblem(wnv, u0, tspan, p)
plot(solve(prob, Tsit5()))


@model function fitlv(data::AbstractVector, prob)
    # Prior distributions.
    σ ~ InverseGamma(2.0, 3.0)
    PI_M = Uniform(0.0, 2000.0)
    M_u = Uniform(0.0, 1000000.0)
    M_i = Uniform(0.0, 20000.0)
    B_u = Uniform(0.0, 100000.0)
    B_i = Uniform(0.0, 20000.0)


    # Simulate Lotka-Volterra model but save only the second state of the system (predators).
    u0=[M_u, M_i, B_u, B_i, 39000000.0, 0.0, 0.0, 0.0, 0.0]
    p = [PI_M]
    t_end = 181
    sol = solve(prob, Tsit5(); p=p, u0=u0, saveat=1, save_idxs=7)
    sol_pred = [sol[7,1], sol[7, trunc(Int,1/6 * t_end)], sol[7, trunc(Int, 2/6*t_end)], sol[7, trunc(Int, 3/6*t_end)], sol[7, trunc(Int, 4/6*t_end)], sol[7, trunc(Int, 5/6*t_end)], sol[7, trunc(Int, t_end)]]

    # Observations of the predators.
    data ~ MvNormal(sol_pred.u, σ^2 * I)

    return nothing
end

model = fitlv(odedata[!, :count], prob)

chain2 = sample(model, NUTS(0.45), MCMCSerial(), 5000, 3; progress=false)


