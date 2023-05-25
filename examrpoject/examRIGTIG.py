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
sigma_values = [1.001, 1.5]
rho_values = [1.001, 1.5]
epsilon = 1.0

# Calculate w_tilde
w_tilde = (1 - tau) * w

# Define the function for optimal labor supply choice
def optimal_labor_supply(w_tilde, kappa, alpha, v):
    return (np.sqrt(kappa**2 + 4 * alpha/v * w_tilde**2) - kappa) / (2 * w_tilde)

# Calculate optimal labor supply for each G
for G in G_values:
    L_star = optimal_labor_supply(w_tilde, kappa, alpha, v)

# Define the utility function
def utility(L, G, alpha, v):
    return np.log(L**alpha * G**(1 - alpha)) - v * L**2 / 2

# Question 4: Find the socially optimal tax rate tau* maximizing worker utility

# Define a function to calculate the negative utility given a tax rate tau
def negative_utility(tau):
    # Calculate the modified wage variable w_tilde using the tax rate tau
    w_tilde = (1 - tau) * w
    # Calculate the optimal labor supply given the modified wage variable and other parameters
    L = optimal_labor_supply(w_tilde, kappa, alpha, v)
    # Calculate the utility value given the optimal labor supply, G, and other parameters
    utility_value = utility(L, G, alpha, v)
    # Return the negative utility value to be minimized
    return -utility_value

#Question 5-6

# Define the new utility function
def utility_1(L, G, alpha, sigma, rho, v, epsilon):
    C = kappa + (1 + tau) * w * L
    return (((alpha * C**((sigma - 1) / sigma) + (1 - alpha) * G**((sigma-1) / (sigma)))**((sigma / (sigma-1)))**((1 - rho)) - 1) / (1 - rho)) - v * L**(1 + epsilon) / (1 + epsilon)

# Define the function for the equilibrium condition
def equilibrium(G, w_tilde, alpha, sigma, rho, v, epsilon):
    L = optimal_labor_supply(w_tilde, kappa, alpha, v)
    return G - tau * w_tilde * L

#
# Define a function to calculate the negative utility with equilibrium given a tax rate tau and other parameters
def negative_utility_with_equilibrium(tau, w, alpha, sigma, rho, v, epsilon):
    # Calculate the modified wage variable w_tilde using the tax rate tau
    w_tilde = (1 - tau) * w
    # Calculate the production factor G using the equilibrium condition
    G = fsolve(equilibrium, 1.0, args=(w_tilde, alpha, sigma, rho, v, epsilon))
    # Calculate the optimal labor supply given the modified wage variable and other parameters
    L = optimal_labor_supply(w_tilde, kappa, alpha, v)
    # Calculate the utility value given the optimal labor supply, G, and other parameters
    utility_value = utility_1(L, G, alpha, sigma, rho, v, epsilon)
    # Return the negative utility value to be minimized
    return -utility_value

#we define the functions negative_utility() and negative_utility_with_equilibrium()  
# to find the socially optimal tax rate that maximizes worker utility. 
# To achieve this, the utility function is multiplied by -1, making it a negative utility. 
# By minimizing the negative utility, we are  maximizing the original utility function.

#Problem 2

# Set up the parameters
eta_2 = 0.5
w_2 = 1.0
kappa_values_2 = [1.0, 2.0]
rho_2 = 0.9
iota_2 = 0.01
sigma_epsilon_2 = 0.1
R_2 = (1 + 0.01)**(1/12)
K_2 = 10000  # Number of random shock series
T_2 = 120  # Number of periods

# Define the function for optimal labor supply choice
def optimal_labor_supply_2(kappa_2, eta_2, w_2):
    return ((1 - eta_2) * kappa_2 / w_2)**(1 / eta_2)

# Define the function for profits
def profits_2(kappa_2, l_2, eta_2, w_2):
    return kappa_2 * l_2**(1 - eta_2) - w_2 * l_2

# Generate random shock series 
np.random.seed(0)  # For reproducibility
epsilon_series_2 = np.random.normal(loc=-0.5*sigma_epsilon_2**2, scale=sigma_epsilon_2, size=(K_2, T_2))
kappa_series_2 = np.exp(rho_2 * np.log(np.append(np.ones((K_2, 1)), np.exp(epsilon_series_2[:, :-1]), axis=1)) + epsilon_series_2)

# Define the function for the policy 
def policy(l_prev, l_star, Delta):
    return l_star if abs(l_prev - l_star) > Delta * abs(l_prev) else l_prev

# Define a function to calculate the negative value of H (mean of H_values)
def negative_H(Delta):
    # Create an empty list to store the values of H
    H_values = []
    # Iterate over each value of k (rows) in the kappa_series_2 array
    for k in range(K_2):
        # Get the kappa_k values for the current row
        kappa_k = kappa_series_2[k, :]
        # Calculate the optimal labor supply for the initial period
        l_prev = optimal_labor_supply_2(kappa_k[0], eta_2, w_2)
        # Calculate the total profit for the initial period
        total_profit = profits_2(kappa_k[0], l_prev, eta_2, w_2)
        # Iterate over each value of t (columns) in the kappa_series_2 array (excluding the first column)
        for t in range(1, T_2):
            # Calculate the optimal labor supply for the current period
            l_star = optimal_labor_supply_2(kappa_k[t], eta_2, w_2)
            # Calculate the policy-based labor supply for the current period using the policy() function and the given Delta
            l = policy(l_prev, l_star, Delta)
            # Calculate the profit for the current period and update the total profit
            total_profit += R_2**(-t) * (profits_2(kappa_k[t], l, eta_2, w_2) - (l != l_prev) * iota_2)
            # Update the previous labor supply with the current labor supply
            l_prev = l
        # Append the total profit to the H_values list
        H_values.append(total_profit)
    # Calculate the negative mean of H_values and return it
    return -np.mean(H_values)


# Here, I suggest a policy that hires or fires only if the difference between l_prev and l_star is greater than Delta * l_star instead of Delta * l_prev.
def alternative_policy(l_prev, l_star, Delta):
    return l_star if abs(l_prev - l_star) > Delta * abs(l_star) else l_prev

#Problem 3

# Define the Griewank function
# Define the Griewank function
def griewank(x):
    """
    Calculate the value of the Griewank function for a given input vector x.
    This function evaluates the Griewank function using a specific formula.
    Args:
        x (array-like): Input vector
    Returns:
        float: Value of the Griewank function
    """
    A = np.sum(x**2 / 4000)  # Calculate the first term of the Griewank function
    B = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))  # Calculate the second term of the Griewank function
    return A - B + 1  # Return the value of the Griewank function


# Bounds for x and tolerance τ > 0.
bounds = [(-600, 600), (-600, 600)]
τ_3 = 1e-8

# The number of warm-up iterations, K > 0, and the maximum number of iterations, underlined_K > K.
K_3 = 1000
underlined_K = 10

# Optimal solution
optimal_solution_3 = None
optimal_value_3 = np.inf

# Store effective initial guesses
initial_guesses = []

# Iterations
for k in range(K_3):
    # A. Draw a random x^k uniformly within the chosen bounds.
    x0 = np.random.uniform(bounds[0][0], bounds[0][1], 2)
    initial_guesses.append(x0)
    
    # B. If k < underlined_K, go to step E.
    if k < underlined_K:     #We skip to E
        # E. Run the optimizer with x^k0 as initial guess and get the result x^(k*)
        res = minimize(griewank, x0, method='BFGS', tol=τ_3)
    
        # F. Set x^* = x^(k*) if k = 0 or f(x^(k*)) < f(x^*)
        if res.fun < optimal_value_3:
            optimal_value_3 = res.fun
            optimal_solution_3 = res.x
            
        # G. If f(x^*) < τ, go to step 4.
        if optimal_value_3 < τ_3:
            break
