import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import fsolve

#Problem 1

# Set up the parameters
alpha = 0.5
kappa = 1.0
v = 1/(2 * 16**2)
w = 1.0
tau = 0.3
G_values = [1.0, 2.0]
w_values = np.linspace(0.1, 2.0, 100)  # Create a range of w values
tau_values = np.linspace(0.1, 0.9, 100)  # Create a range of tau values
sigma_values = [0.001, 1.5]
rho_values = [1.001, 1.5]
epsilon = 1.0


# Calculate w_tilde
w_tilde = (1 - tau) * w

# Define the function for optimal labor supply choice
def optimal_labor_supply(w_tilde, kappa, alpha, v):
    return -kappa * np.sqrt(kappa**2 + 4 * (alpha/v) * w_tilde**2) / (2 * w_tilde)

# Calculate optimal labor supply for each G
for G in G_values:
    L_star = optimal_labor_supply(w_tilde, kappa, alpha, v)
    print(f"For G = {G}, the optimal labor supply choice is {L_star}")


# "CHAT FORKLARE KODEN SÅDAN (MEN VED IKKE OM DET SKAL MED":

#This code first sets up the parameters as given in the problem.
#  It then defines a function optimal_labor_supply that calculates 
# the optimal labor supply given w_tilde, kappa, alpha, and v. 
# Finally, it calculates and prints the optimal labor supply for each value of G in G_values.


# Define the utility function
def utility(L, G, alpha, v):
    return np.log(L**alpha * G**(1 - alpha)) - v * L**2 / 2

# Define the function for optimal labor supply choice
def optimal_labor_supply(w_tilde, kappa, alpha, v):
    return -kappa * np.sqrt(kappa**2 + 4 * (alpha/v) * w_tilde**2) / (2 * w_tilde)

# Question 2: Illustrate how L*(w_tilde) depends on w
L_star_values = [optimal_labor_supply((1 - tau) * w, kappa, alpha, v) for w in w_values]
plt.figure(figsize=(10, 6))
plt.plot(w_values, L_star_values)
plt.xlabel('w')
plt.ylabel('L*(w_tilde)')
plt.title('Dependence of Optimal Labor Supply on w')
plt.grid(True)
plt.show()

# Question 3: Plot the implied L, G, and worker utility for a grid of tau-values
L_values = [optimal_labor_supply((1 - tau) * w, kappa, alpha, v) for tau in tau_values]
utilities = [utility(L, G, alpha, v) for L, G in zip(L_values, G_values)]
plt.figure(figsize=(10, 6))
plt.plot(tau_values, L_values, label='L')
plt.plot(tau_values, G_values, label='G')
plt.plot(tau_values, utilities, label='Utility')
plt.xlabel('tau')
plt.ylabel('Value')
plt.title('Implied L, G, and Worker Utility for Different Tau Values')
plt.legend()
plt.grid(True)
plt.show()

# Question 4: Find the socially optimal tax rate tau* maximizing worker utility
def negative_utility(tau):
    w_tilde = (1 - tau) * w
    L = optimal_labor_supply(w_tilde, kappa, alpha, v)
    return -utility(L, G, alpha, v)

result = minimize(negative_utility, 0.5, bounds=[(0.1, 0.9)])
optimal_tau = result.x[0]
print(f"The socially optimal tax rate is {optimal_tau}")


##CHAT BESKRIVER KODERNE FRA 2-4 SÅDAN HER
#This code extends the previous code by adding plots 
# for the dependence of optimal labor supply on w and the implied 
# L, G, and worker utility for a grid of tau values. It also uses 
# the scipy.optimize.minimize function to find the socially optimal 
# tax rate that maximizes worker utility. 
# The negative_utility function is defined because minimize 
# finds the minimum of a function, and we want to find the maximum utility.


# Define the new utility function
def utility(L, G, alpha, sigma, rho, v, epsilon):
    C = kappa + (1 + tau) * w * L
    return ((alpha * C**((sigma - 1) / sigma) + (1 - alpha) * G**((sigma - 1) / sigma))**((1 - rho) / sigma) - 1) / (1 - rho) - v * L**(1 + epsilon) / (1 + epsilon)

# Define the function for the equilibrium condition
def equilibrium(G, w_tilde, alpha, sigma, rho, v, epsilon):
    L = optimal_labor_supply(w_tilde, kappa, alpha, v)
    return G - tau * w * L

# Question 5: Find the G that solves the equilibrium condition
for sigma, rho in zip(sigma_values, rho_values):
    G_solution = fsolve(equilibrium, 1.0, args=(w_tilde, alpha, sigma, rho, v, epsilon))
    print(f"For sigma = {sigma} and rho = {rho}, the solution for G is {G_solution[0]}")

# Question 6: Find the socially optimal tax rate tau* maximizing worker utility while keeping the equilibrium condition
def negative_utility_with_equilibrium(tau, w, alpha, sigma, rho, v, epsilon):
    w_tilde = (1 - tau) * w
    G = fsolve(equilibrium, 1.0, args=(w_tilde, alpha, sigma, rho, v, epsilon))
    L = optimal_labor_supply(w_tilde, kappa, alpha, v)
    return -utility(L, G, alpha, sigma, rho, v, epsilon)

for sigma, rho in zip(sigma_values, rho_values):
    result = minimize(negative_utility_with_equilibrium, 0.5, args=(w, alpha, sigma, rho, v, epsilon), bounds=[(0.1, 0.9)])
    optimal_tau = result.x[0]
    print(f"For sigma = {sigma} and rho = {rho}, the socially optimal tax rate is {optimal_tau}")


##CHAT BESKRIVER KODERNE FRA 5-6 SÅDAN HER
#This code extends the previous code by adding the 
# new utility function and the equilibrium condition. 
# It then uses scipy.optimize.fsolve to find the G that solves 
# the equilibrium condition for each set of sigma and rho values. 
# Finally, it finds the socially optimal tax rate tau* that maximizes
#  worker utility while keeping the equilibrium condition, again for each 
# set of sigma and rho values.