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
MU_H = 1/(70*365) #Reciprocal of length of human life in days
#q = 0 --> 1
#c = 0 --> 1 
b_1 = 0.09 #adjusts peak
#b_2 = 0.09(1-cq)
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
u0 = [100000.0, 5000.0, 50000.0, 1000.0, 39000000.0, 0, 0, 0, 0]


p = [22000, .20, 0, 0]

tspan = (0.0, 180.0)
prob = ODEProblem(wnv, u0, tspan, p)

#Plot simulation
#sol = solve(prob, Tsit5(); saveat=0.1)

#Stiff solving
sol = solve(prob, Rodas5(), saveat=1)


#plot(sol)

plot(sol[7,:], label="Symptomatically Infected Humans")

odedata = California.load()[5:11, :]

print(odedata)
#plot!(odedata[!, :count], label="Observed Symptomatically Infected Humans")
x_odedata = [30*i for i in 0:6]
y_odedata = odedata[!, :count]
plot!(scatter!(x_odedata, y_odedata, label="Observed Symptomatically Infected Humans"))


@model function fitlv(data, prob)
    #Prior distributions

    