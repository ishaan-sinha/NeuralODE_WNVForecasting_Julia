using DifferentialEquations

# ODE function
f1 = function (du,u,p,t)
  du[1] = - p[1]*p[2] * u[1]
end

# Initial conditions and parameter values.
# p1 is the parameter to be estimated.
# p2 and u0 are experimental parameters known for each dataset.
u0 = [1.,2.]
p1 = .3
p2 = [.1,.2]
tspan = (0.,10.)
t=range(0,10,5)

# Creating data for 1st experimental dataset.
## Experimental conditions: u0 = 1. ; p[2] = .1
prob1 = ODEProblem(f1,[u0[1]],tspan,[p1,p2[1]])
sol1=solve(prob1,Tsit5(),saveat=t)
data1 = Array(sol1)

# Creating data for 2nd experimental dataset.
## Experimental conditions: u0 = 1. ; p[2] = .2
prob2 = ODEProblem(f1,[u0[1]],tspan,[p1,p2[2]])
sol2=solve(prob2,Tsit5(),saveat=t)
data2 = Array(sol2)

# Creating data for 3rd experimental dataset.
## Experimental conditions: u0 = 2. ; p[2] = .1
prob3 = ODEProblem(f1,[u0[2]],tspan,[p1,p2[1]])
sol3=solve(prob3,Tsit5(),saveat=t)
data3 = Array(sol3)

function prob_func(prob,i,repeat)
  i < 3 ? u0 = [1.0] : u0 = [2.0]
  i == 2 ? p = (prob.p[1],0.2) : p = (prob.p[1],0.1)
  ODEProblem{true}(f1,u0,(0.0,10.0),p)
end
prob = MonteCarloProblem(prob1,prob_func = prob_func)

# Just to show what this looks like
sol = solve(prob,Tsit5(),saveat=t,parallel_type = :none,
            num_monte = 3)

loss1 = L2Loss(t,data1)
loss2 = L2Loss(t,data2)
loss3 = L2Loss(t,data3)
loss(sol) = loss1(sol[1]) + loss2(sol[2]) + loss3(sol[3])

function my_problem_new_parameters(prob,p)
  prob.prob.p[1] = p[1]
  prob
end
obj = build_loss_objective(prob,Tsit5(),loss,
                           prob_generator = my_problem_new_parameters,
                           num_monte = 3,
                           saveat = t)

using Optim
res = optimize(obj,0.0,1.0)