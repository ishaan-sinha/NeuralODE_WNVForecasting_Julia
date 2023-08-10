using Turing
using DifferentialEquations

# Load StatsPlots for visualizations and diagnostics.
using StatsPlots

using LinearAlgebra

include(joinpath("CA_data", "california.jl"))



#PI_M = variable
PI_B = 1000
PI_H = 30
#MU_M = variable
MU_B = 1000
MU_H = 1/(70*365)
#q = 0 --> 1
#c = 0 --> 1 
b_1 = 0.09
#b_2 = 0.09(1-cq)
BETA_1 = 0.16
BETA_2 = 0.88
BETA_3 = 0.88
d_B = 5 * 10^-5
d_H = 5 * 10^-7
ALPHA = 1/14
DELTA = 1
TAU = 1/14

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


function wnv(du, u, p, t)
    #Model parameters
    PI_M, MU_M, q, c = p
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

#Define initial-value problem
u0 = [50000.0, 1000.0, 10000.0, 1000.0, 100000.0, 0, 0, 0, 0]
p = [50, 50, 0.5, 0.5]
tspan = (0.0, 12.0)
prob = ODEProblem(wnv, u0, tspan, p)

#Plot simulation
sol = solve(prob, Tsit5(), saveat=1)
#plot(sol)

plot(sol[7,:], label="Symptomatically Infected Humans")

odedata = California.load()[1:12, :]
plot!(odedata[!, :count], label="Observed Symptomatically Infected Humans")