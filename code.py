#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# In[2]:


# Load your dataset
def loading_preprocessing_data(filepath):
    data = pd.read_csv("C:/Users/yogan/Downloads/04_cricket_1999to2011.csv")
    return data


# In[3]:


#preprocessing the data
data = pd.read_csv("C:/Users/yogan/Downloads/04_cricket_1999to2011.csv")
data = data[data['Innings'] == 1]
y_col = 'Runs'
w_col = 'Wickets.in.Hand'
u_col = 'Over' ##overs left
match_col = 'Match'
runs_col = 'Runs.Remaining'

# Ensure 'Wickets.in.Hands' is numeric and convert to integers
data[w_col] = pd.to_numeric(data[w_col], errors='coerce').fillna(0).astype(int)

# Drop rows with missing values in relevant columns
data = data.dropna(subset=[y_col, w_col, u_col])
# Remove rows where wickets in hand are 0
data = data[data[w_col] > 0]
# Subtract overs-left from a fixed value, if needed (e.g., 50 overs match)
data[u_col] = 50 - data[u_col]
#print (data[u_col])


# In[4]:


## for Z0_w for every wicket
grouped_data = data.groupby([match_col, w_col])
# Find the maximum runs for each match with a given number of wickets in hand
max_runs_per_group = grouped_data[runs_col].max().reset_index()
# Group by wickets in hand (w_col) and find the average of these maximum runs
Z0_values = max_runs_per_group.groupby(w_col)[runs_col].mean()
# Convert Z0_values to a numpy array (with indices from 1 to 10 for w=1 to w=10)
initial_Z0_w = np.array([Z0_values.get(i, 0) for i in range(1, 11)])
# Print the Z0 values for each wickets in hand (w)
for i in range(1, 11):
    print(f"initial_Z0_w[{i}] (Average of max runs with {i} wickets in hand): {initial_Z0_w[i-1]}")


# In[5]:


# Define the model function Z(u, w)
def run_production_function(u, Z0_w, L_w):
    # Ensure Z0_w and L_w are scalars for each call
    return Z0_w * (1 - np.exp(-L_w * u / Z0_w))


# In[6]:


#check log_x
def check_log(x, epsilon=1e-10):
    return np.log(np.maximum(x, epsilon))


# In[7]:


#Loss function For L_w
def loss_function(y_pred, y_true):
    return (y_pred + 1) * check_log((y_pred + 1) / (y_true + 1)) - y_pred + y_true


# In[8]:


#3 optimizing
def optimization_function(params, u, w, y_true):
    Z0_w, L_w = params
    y_pred = run_production_function(u, Z0_w, L_w)
    return np.sum(loss_function(y_pred, y_true))


# In[9]:


# Optimizing for Z0_w and L-w
def fit_preliminary_model(data):
    wickets = data['Wickets.in.Hand'].unique()
    preliminary_params = {}

    for w in wickets:
        u = data[data['Wickets.in.Hand'] == w]['Over']
        y_true = data[data['Wickets.in.Hand'] == w]['Runs.Remaining']
        y_true=np.clip(y_true,1e-10,None)
        
        # Initial guess for Z0(w) and L(w)
        Z0_w = np.array([Z0_values.get(i, 0) for i in range(1, 11)])
        initial_guess = [10, 10]
        result = minimize(optimization_function, initial_guess, args=(u, w, y_true), method='L-BFGS-B')
        
        Z0_w, L_w = result.x
        preliminary_params[w] = (Z0_w, L_w)

    
    
    return preliminary_params


# In[10]:


## print and plot preliminary function
def print_preliminary_params(preliminary_params):
    print("Preliminary Parameters (20 in total: Z0 and L for each wicket)")
    print("The preliminary model estimated the following values for Z0 and L for each wicket:")
    for w, (Z0_w, L_w) in preliminary_params.items():
        print(f"• Wicket {int(w)}: Z0 = {Z0_w:.2f}, L = {L_w:.2f}")
    print()  # Blank line for readability
    plt.figure(figsize=(10, 6))
    u = np.linspace(1, 50, 50)  # Overs from 1 to 50
    
    for w, (Z0_w, L_w) in preliminary_params.items():
        runs_predicted = run_production_function(u, Z0_w, L_w)  # Predicted runs
        plt.plot(u, runs_predicted, label=f'Wickets {w}')

    plt.xlabel('Overs Remaining (u)', fontsize=14)
    plt.ylabel('Predicted Runs (Z(u, w))', fontsize=14)
    plt.title('Run Production Functions', fontsize=16)
    plt.legend(title='Wickets in Hand')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("preliminary fit.png", format="png")
    plt.show()



# In[11]:


# Function to compute the weighted average of slope L
def calculate_L(preliminary_params, data):
    total_points = 0
    weighted_sum_L = 0

    for w, (Z0_w, L_w) in preliminary_params.items():
        num_points = len(data[data['Wickets.in.Hand'] == w])
        weighted_sum_L += L_w * num_points
        total_points += num_points

    return weighted_sum_L / total_points


# In[12]:


## for final optimation of Zo_w
def fit_final_model(data, L):
    wickets = data['Wickets.in.Hand'].unique()
    final_params = {}

    for w in wickets:
        u = data[data['Wickets.in.Hand'] == w]['Over']
        y_true = data[data['Wickets.in.Hand'] == w]['Runs.Remaining']
        y_true=np.clip(y_true,1e-10,None)
        # Fix L and optimize only Z0
        def fixed_L_optimization(Z0_w):
            return np.sum(loss_function(run_production_function(u, Z0_w, L), y_true))
        
        result = minimize(fixed_L_optimization, [100], method='L-BFGS-B')
        Z0_w = result.x[0]
        final_params[w] = Z0_w
    
    return final_params


# In[13]:


def print_final_params(final_params, L):
    print("Final Parameters (11 in total: 10 Z0 values and 1 common L)")
    print("The final model estimated the following values for Z0 for each wicket, using a common L value:")
    for w, Z0_w in final_params.items():
        print(f"• Wicket {int(w)}: Z0 = {Z0_w:.2f}")
    print(f"• L = {L:.2f}")
    print()  
    plt.figure(figsize=(10, 6))
    overs = np.linspace(1, 50, 50)  # Overs from 1 to 50
    
    for w, Z0_w in final_params.items():
        # Generate the predicted runs for different overs
        runs_predicted = run_production_function(overs, Z0_w, L)
        plt.plot(overs, runs_predicted, label=f'Wickets {int(w)}')  # Plot for each wicket

    plt.xlabel('Overs Remaining (u)', fontsize=14)
    plt.ylabel('Predicted Runs (Z(u, w))', fontsize=14)
    plt.title('Run Production Functions', fontsize=16)
    plt.legend(title='Wickets in Hand')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("best_fit.png", format="png")
    plt.show()


# In[14]:


def calculate_normalized_loss(params, data, L):
    total_loss = 0
    total_points = 0

    for w, Z0_w in params.items():
        # Extract the overs and true runs for the current number of wickets
        u = data[data['Wickets.in.Hand'] == w]['Over']
        y_true = data[data['Wickets.in.Hand'] == w]['Runs.Remaining']
        
        # Clip y_true to avoid negative values or zeros that could cause issues with the logarithm
        y_true = np.clip(y_true, 1e-10, None)

        # Predict runs using the current Z0_w and the common L
        y_pred = run_production_function(u, Z0_w, L)

        # Clip y_pred to avoid numerical issues in the loss function
        y_pred = np.clip(y_pred, 1e-10, None)

        # Accumulate the total loss and the total number of points
        total_loss += np.sum(loss_function(y_pred, y_true))
        total_points += len(u)

    # Return the average loss per data point (normalized loss)
    return total_loss / total_points


# In[15]:


##fitting preliminary model
preliminary_params = fit_preliminary_model(data)
print(print_preliminary_params(preliminary_params))

#  Calculate common L
L = calculate_L(preliminary_params, data)

#  Fit final model
final_params = fit_final_model(data, L)
print(print_final_params(final_params, L))


# Calculate normalized loss
normalized_loss = calculate_normalized_loss(final_params, data, L)
print(f'Normalized Loss: {normalized_loss}')

