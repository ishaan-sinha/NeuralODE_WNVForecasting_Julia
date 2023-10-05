using DataDrivenDiffEq
using ModelingToolkit
using OrdinaryDiffEq
using DataDrivenSparse
using LinearAlgebra

# Create a test problem
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

x = [-40.94606,18207.729,-6927.3213,-339.48212]

u0 = [499*x[1], x[1], x[2], x[3], 39000000, 0, 0, 0, 0]
p = [x[4]]
t_end = 181
tspan = (0.0, t_end)
prob = ODEProblem(wnv, u0, tspan, p)
sol = solve(prob, Rodas5(), saveat=1, dt=1e-6)

## Start the automatic discovery
ddprob = DataDrivenProblem(sol)


u0 = [1.0; 0.0; 0.0]
tspan = (0.0, 100.0)
dt = 0.1
prob = ODEProblem(lorenz, u0, tspan)
sol = solve(prob, Tsit5(), saveat = dt)

## Start the automatic discovery
ddprob = DataDrivenProblem(sol)

@variables t M_u(t) M_i(t) B_u(t) B_i(t) S(t) E(t) I(t) H(t) R(t)
u = [M_u; M_i; B_u; B_i; S; E; I; H; R]
basis = Basis(polynomial_basis(u, 3), u, iv = t)
opt = STLSQ(exp10.(-5:0.1:-1))
ddsol = solve(ddprob, basis, opt, options = DataDrivenCommonOptions(digits = 1))
println(get_basis(ddsol))