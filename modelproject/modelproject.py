import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# Set up the baseline model/parameters for country A and B (i.e Table 1)

params = {
    "A": {"s": 0.25, "delta": 0.05, "n": 0.02, "alpha": 0.33, "k0": 1.0},
    "B": {"s": 0.35, "delta": 0.04, "n": 0.01, "alpha": 0.40, "k0": 0.8},
}

# Cf exersice description the 'Total factor productivity (A)' is assumed to be constant
total_factor_productivity = 1

#Create a dataframe with the chosen values for the different parametsers
dataframe = {'': ['Country A', 'Country B'], 'Saving(s)': [0.25 , 0.35], 'Depriciation(δ)': [0.05 , 0.04], 'Population growth(n)': [0.02 , 0.01], 'Capital share(α)': [0.33, 0.40], 'Capital per worker (k_0)': [1.00, 0.80]}

df = pd.DataFrame(dataframe)

#a) 

# Function to calculate steadystate capital per worker (k^*) and output per worker (y^*)
def calculate_steady_state(params, A=1):
    s, delta, n, alpha = params["s"], params["delta"], params["n"], params["alpha"]
    k_star = (s * A / (delta + n))**(1 / (1 - alpha))
    y_star = A * k_star**alpha
    return k_star, y_star

# Calculate steadystate values for Country A and Country B (Call the functions for steady state)

steady_state_A = calculate_steady_state(params["A"], total_factor_productivity)
steady_state_B = calculate_steady_state(params["B"], total_factor_productivity)


#b)

# Function to simulate the Solow Growth Model over time
def simulate_solow_growth(params, A=1, T=100):
    s, delta, n, alpha, k0 = params["s"], params["delta"], params["n"], params["alpha"], params["k0"]
    k = np.zeros(T+1)
    y = np.zeros(T+1)
    k[0] = k0
    y[0] = A * k0**alpha
    
    for t in range(T):
        k[t+1] = k[t] + s * y[t] - (delta + n) * k[t] #Simulating over a period of T=100
        y[t+1] = A * k[t+1]**alpha #Simulating over a period of T=100

    return k, y

# Simulate the Solow Growth Model for Country A and Country B over 100 years
k_A, y_A = simulate_solow_growth(params["A"], total_factor_productivity)
k_B, y_B = simulate_solow_growth(params["B"], total_factor_productivity)


#c)

# We use the list from the previous question and define a catch up time
#The function iterates through the time steps and checks if the output per worker for Country B is greater than or equal to the output per worker for Country A. 
def find_catch_up_time(y_A, y_B):
    for t in range(len(y_A)):
        if y_B[t] >= y_A[t]:
            return t
    return None



#d)

#defining a function with the parameter values as well as saving rates

def analyze_sensitivity_savings_rate(params, A, s_values):
    k_star_values = [] #Initializing over empty lists to store values
    y_star_values = [] #Initializing over empty lists to store values
    
    for s in s_values:
        params_updated = params.copy() #creating a copy of the params dictionary and updates the saving rate (s) with the current value from the loop.
        params_updated["s"] = s
        steady_state = calculate_steady_state(params_updated, A)
        k_star_values.append(steady_state[0])
        y_star_values.append(steady_state[1])
    
    return k_star_values, y_star_values

# Define a range of savings rate values to analyze
s_values = np.linspace(0.01, 0.5, 100)

# Analyze the sensitivity of steady-state capital per worker (k^*) and output per worker (y^*) to the savings rate (s) for Country A
k_star_values_A, y_star_values_A = analyze_sensitivity_savings_rate(params["A"], total_factor_productivity, s_values)



# e)

def analyze_sensitivity_depreciation_rate(params, A, delta_values):
    k_star_values = []
    y_star_values = []
    
    for delta in delta_values:
        params_updated = params.copy()
        params_updated["delta"] = delta
        steady_state = calculate_steady_state(params_updated, A)
        k_star_values.append(steady_state[0])
        y_star_values.append(steady_state[1])
    
    return k_star_values, y_star_values
# Define the depreciation rates to analyze
delta_values_specific = [0.02, 0.05, 0.08]

# Analyze the sensitivity of steady-state capital per worker (k*) and output per worker (y*) to the depreciation rate (δ) for Country A
k_star_values_A_delta_specific, y_star_values_A_delta_specific = analyze_sensitivity_depreciation_rate(params["A"], total_factor_productivity, delta_values_specific)



# Set the above result up in a Table (2)

# Create a dictionary with the data
data = {
    "Depreciation Rate (δ)": delta_values_specific,
    "Capital per Worker (k*)": k_star_values_A_delta_specific,
    "Output per Worker (y*)": y_star_values_A_delta_specific
}

# Create a DataFrame using the data dictionary
table_2 = pd.DataFrame(data)

# Set the index to start from 1 instead of 0 (as python usually does)
table_2.index += 1






