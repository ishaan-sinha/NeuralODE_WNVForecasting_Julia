using DifferentialEquations

function lotka_volterra(du,u,p,t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = α*x - β*x*y
    du[2] = dy = -δ*y + γ*x*y
    end

u0_f(p,t0) = [p[2],p[4]]
tspan_f(p) = (0.0,10*p[4])
p = [1.5,1.0,3.0,1.0]
prob = ODEProblem(lotka_volterra,u0_f,tspan_f,p)

sol = solve(prob)

using Plots
plt = plot(sol)
display(plt)

print("Reached End")
