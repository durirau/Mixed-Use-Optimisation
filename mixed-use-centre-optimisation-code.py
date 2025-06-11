#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: boerner, rau
"""

import sys
import pandas as pd
import numpy as np
from pyscipopt import Model, quicksum

######################################
# 1. ANALYSIS PARAMETERS
######################################

# A list of weighting factors for the preference term to be tested in the sensitivity analysis.
c_pref_list = [0.4, 0.5, 0.6]

# The corresponding weights for the rent term are derived as (1 - c_pref).
# round() is used to prevent minor floating-point inaccuracies.
c_rent_list = [round(1 - p, 2) for p in c_pref_list]


######################################
# 2. DATA IMPORT FROM EXCEL
######################################

# Read the preferences and rents from the specified Excel file.
# It is assumed that the data begins from the second row (header=1).
df = pd.read_excel(
    'preferences-and-rents.xlsx',
    header=1,
    usecols=['Average preferences', 'calculated rents']
)

# Extract the columns as NumPy arrays for numerical processing.
# The raw, original preference values.
preferences_raw = df['Average preferences'].to_numpy()
# The raw, original rent values are stored before normalisation.
rents_raw = df['calculated rents'].to_numpy()

# The normalised rents are created for use in the optimisation model's objective function.
rents_normalised = rents_raw / rents_raw.sum()


# Model dimensions: m = number of storeys, n = number of departments.
m, n = 7, 6
# Verify that the number of preference entries matches the expected count for the model structure.
expected_entries = (n - 1) + (m - 1) * n
assert preferences_raw.size == expected_entries, \
    f"Expected {expected_entries} preference entries, but found {preferences_raw.size}."


######################################
# 3. CONSTRUCT THE PREFERENCE MATRIX
######################################

# The 'preference_matrix_raw' holds the original, un-normalised preference values in a 7x6 structure.
preference_matrix_raw = np.zeros((m, n))
# The first storey (ground floor) has only 5 preference values, as one department cannot be located there.
preference_matrix_raw[0, :n-1] = preferences_raw[0 : (n-1)]
# All subsequent storeys have a preference value for each of the 6 departments.
for i in range(1, m):
    start = (n-1) + (i-1) * n
    end   = (n-1) + i * n
    preference_matrix_raw[i, :] = preferences_raw[start:end]

# Matrix 'A' contains the row-normalised preference values for the model.
# A[i,j] represents the relative preference for department j on storey i.
A = (preference_matrix_raw.T / preference_matrix_raw.sum(axis=1)).T


######################################
# 4. FORMULATE AND RUN THE OPTIMISATION MODEL
######################################
# (This section contains the core optimisation logic)

# Additional model parameters.
w_mean = [0.1467, 0.162, 0.138, 0.1226, 0.1158, 0.135, 0.1821]  # Weights per storey.
U      = [4] * m  # Maximum number of departments permitted per storey.

# Initialise a dictionary to collect the solutions from all runs.
all_solutions = {f"[{i},{j}]": [] for i in range(m) for j in range(n)}
sensitivity_labels = list(zip(c_pref_list, c_rent_list))

print("Starting optimisation runs for different weightings...")

# Loop over all (c_pref, c_rent) pairs for the sensitivity analysis.
for k, (c_pref, c_rent) in enumerate(sensitivity_labels):
    print(f"  Run {k+1}/{len(sensitivity_labels)}: c_pref={c_pref}, c_rent={c_rent}")
    
    model = Model("DepartmentStore")

    # === DECISION VARIABLES ===
    # x[i,j]: proportion of department j allocated to storey i (continuous).
    x = {(i,j): model.addVar(vtype='C', lb=0, ub=1, name=f"x_{i}-{j}")
         for i in range(m) for j in range(n)}
    # z[i,j]: 1 if department j is located on storey i, 0 otherwise (binary).
    z = {(i,j): model.addVar(vtype='B', name=f"z_{i}-{j}")
         for i in range(m) for j in range(n)}

    # === GENERAL CONSTRAINTS ===
    # Constraints applicable to each storey.
    for i in range(m):
        model.addCons(quicksum(z[i,j] for j in range(n)) <= U[i])  # Max U departments per storey.
        model.addCons(quicksum(x[i,j] for j in range(n)) == 1)   # Each storey's area is fully utilised.
        for j in range(n):
            model.addCons(x[i,j] <= z[i,j])  # Links the continuous variable x to the binary variable z.

    # === LOGICAL CONSTRAINTS ===
    # Specific business rules for the department store layout.
    model.addCons(z[0,5] == 0) # Forbids department 5 on storey 0.
    model.addCons(z[0,4] + z[1,4] <= 1)
    model.addCons(z[3,4] + z[5,3] <= 1)
    model.addCons(z[3,5] + z[5,4] <= 1)
    for i in (3,4,5):
        apart_idx = 3 if i==3 else 2
        others = [j for j in range(n) if j != apart_idx]
        model.addCons(quicksum(z[i,j] for j in others) <= 5*(1 - z[i,apart_idx]))
    model.addCons(z[3,3] + z[5,2] <= 1 + z[4,2])
    model.addCons(z[3,0] <= quicksum(z[2,j] for j in range(n)))
    model.addCons(z[6,4] <= quicksum(z[5,j] for j in (0,1,2)))

    # === OBJECTIVE FUNCTION ===
    # Formulated via an auxiliary variable for clarity.
    obj_pref = model.addVar(vtype='C', name="obj_pref")
    # A flattened list of variables, excluding the forbidden combination (0,5).
    x_flat   = [x[i,j] for i in range(m) for j in range(n) if not (i==0 and j==5)]
    
    # The preference term: Minimises the weighted squared deviation from ideal preferences.
    model.addCons(
        quicksum(c_pref * w_mean[i] * quicksum((x[i,j] - A[i,j])**2 for j in range(n))
                 for i in range(m))
        == obj_pref
    )
    
    # The rent term is maximised (or, its negative is minimised).
    # The final objective balances preference satisfaction against rent maximisation.
    model.setObjective(quicksum([-c_rent * np.dot(rents_normalised, x_flat)]) + obj_pref,
                       "minimize")

    # === SOLVE AND COLLECT RESULTS ===
    model.setParam('limits/time', 1000)
    model.presolve()
    model.optimize()

    status = model.getStatus()
    for i in range(m):
        for j in range(n):
            val = (model.getSolVal(model.getBestSol(), x[i,j])
                   if status!='infeasible' else np.nan)
            all_solutions[f"[{i},{j}]"].append(val)


######################################
# 5. AGGREGATE AND EXPORT RESULTS
######################################

print("\nAggregating results into a single table...")

# 1. Create the base results DataFrame.
index_order = [f"[{i},{j}]" for i in range(m) for j in range(n)]
df_results = pd.DataFrame(all_solutions, index=range(len(sensitivity_labels))).T
df_results.columns = pd.MultiIndex.from_tuples(sensitivity_labels, names=["c_pref", "c_rent"])
df_results = df_results.loc[index_order]
df_results.index.name = "Variable (i,j)"

# 2. Create columns for the original, un-normalised input data.
# A list of original preference values, ordered to match the DataFrame index.
preference_values_raw = [preference_matrix_raw[i, j] for i in range(m) for j in range(n)]

# A list of original rent values. A value of 0 is used for the forbidden (0,5) combination.
rent_map_raw = {(i, j): rent for (i, j), rent in zip([(i, k) for i in range(m) for k in range(n) if not (i == 0 and k == 5)], rents_raw)}
rent_values_raw = [rent_map_raw.get((i, j), 0) for i in range(m) for j in range(n)]

# 3. Insert the new columns containing the original data at the beginning of the results table.
df_results.insert(0, 'Rent (Original)', rent_values_raw)
df_results.insert(0, 'Preference (Original)', preference_values_raw)

# 4. Export the final, combined table to a single Excel file.
output_filename = "DepartmentStore_Results_Compact.xlsx"
df_results.to_excel(output_filename)

print(f"\nExport complete. The compact results table has been saved to '{output_filename}'.")
