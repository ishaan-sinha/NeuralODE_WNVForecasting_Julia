using Turing
using DifferentialEquations

# Load StatsPlots for visualizations and diagnostics.
using StatsPlots

using LinearAlgebra

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

=#


function wnv(du, u, p, t)
    #Model parameters

