#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Urban Redevelopment Optimization Model

This script solves a mixed-integer nonlinear programming (MINLP) problem 
to determine the optimal allocation of use-cases (e.g., retail, housing) 
across different stories of a building. The objective is to find a balance 
between maximizing rental income (investor's perspective) and fulfilling 
the preferences of citizens (survey data).

The model is formulated and solved using the PySCIPOpt library, an interface 
for the SCIP Optimization Suite.

A sensitivity analysis is performed by varying the weights of the two objective
function components (rental income and citizen preferences).

@author: anonymous for blind review
@version: 1.0 (Refactored for publication)
"""

# =============================================================================
# 1. IMPORTS - Libraries required for the script
# =============================================================================
import pandas as pd  # Used for reading and handling data from Excel files.
import numpy as np   # Used for numerical operations, especially array/matrix manipulation.
from pyscipopt import Model, quicksum  # Core components from the SCIP solver interface.
                                       # Model: The main optimization model object.
                                       # quicksum: A performance-optimized sum function for constraints/objectives.

# =============================================================================
# 2. CONFIGURATION - All parameters and settings for the model
# =============================================================================

# --- File Paths ---
INPUT_DATA_FILE = 'mieten.xlsx'
OUTPUT_SOLUTION_FILE = 'Solution_Analysis.xlsx'
MODEL_DEBUG_FILE_PREFIX = 'model_Kaufhaus'

# --- Model Dimensions ---
NUM_STORIES = 7       # Total number of stories in the building (indexed 0 to 6).
NUM_USE_CASES = 6     # Number of different use-cases per story (indexed 0 to 5).

# --- Data Reading Parameters ---
# Row in the Excel file from which to start reading the data.
# This should correspond to the first row of numerical data.
DATA_START_ROW = 19   
# Column indices in the Excel file for specific data.
# Note: Excel columns A,B,C correspond to indices 0,1,2.
PREFERENCES_COL_IDX = 1
RENTS_COL_IDX = 2

# --- Model Parameters ---
# Mean importance weight of each story, extracted from a survey.
# Index i corresponds to story i.
STORY_IMPORTANCE_WEIGHTS = [0.1467, 0.162, 0.138, 0.1226, 0.1158, 0.135, 0.1821]

# Maximum number of *different* use-cases allowed per story.
# Here, a list of length NUM_STORIES, where each entry is the limit for that story.
MAX_DISTINCT_USES_PER_STORY = NUM_STORIES * [4]

# --- Sensitivity Analysis Parameters ---
# The script runs the optimization for each pair of weights from these lists.
# The lists must have the same length.
# c_pref: Weight for the citizen preference term in the objective function.
WEIGHT_PREFERENCES_LIST = [0.3,  0.35, 0.4,  0.45, 0.5,  0.55, 0.6, 0.65, 0.7, 1.0]
# c_mieten: Weight for the rental income term in the objective function.
WEIGHT_RENTS_LIST       = [0.7,  0.65, 0.6,  0.55, 0.5,  0.45, 0.4, 0.35, 0.3, 0.0]

# --- Solver Settings ---
# Time limit in seconds for a single, full optimization run.
SOLVER_TIME_LIMIT_FULL = 64800 # 18 hours
# Reduced time limit for each run within the sensitivity analysis.
SOLVER_TIME_LIMIT_SENSITIVITY = 1000 # ~16 minutes


# =============================================================================
# 3. DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_preprocess_data(filename, start_row, rents_col, pref_col, m, n):
    """
    Loads rent and preference data from an Excel file and preprocesses it.

    Args:
        filename (str): Path to the Excel file.
        start_row (int): Header row number to start reading from.
        rents_col (int): Column index for rental data.
        pref_col (int): Column index for preference data.
        m (int): Number of stories.
        n (int): Number of use-cases.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Normalized rental data (flattened).
            - np.ndarray: Normalized preference data (as a m x n matrix).
    """
    print("--- Loading and preprocessing data ---")
    
    # Read and normalize rental income data
    df_rents = pd.read_excel(filename, skiprows=start_row, usecols=[rents_col])
    rents = df_rents.values
    # Normalize rents by their sum to get relative values.
    # Note: A check for sum being zero might be needed for robustness.
    sum_rents = np.sum(rents)
    normalized_rents = rents / sum_rents
    
    # Read citizen preference data
    df_prefs = pd.read_excel(filename, skiprows=start_row, usecols=[pref_col])
    preferences = df_prefs.values
    
    # Reshape the flat preference list into a (m x n) matrix.
    # The structure of the Excel file dictates this specific reshaping logic.
    # The first story (index 0) has n-1 use cases, others have n.
    raw_preference_matrix = np.zeros((m, n))
    raw_preference_matrix[0, 0:n-1] = np.transpose(preferences[0:n-1])
    for i in range(1, m):
        start = (n-1) + (i-1) * n
        end = (n-1) + i * n
        raw_preference_matrix[i, 0:n] = np.transpose(preferences[start:end])
    
    # Normalize each row of the preference matrix to sum to 1.
    # This converts absolute preference scores into proportional preferences per story.
    normalized_preference_matrix = np.zeros_like(raw_preference_matrix)
    for i, row in enumerate(raw_preference_matrix):
        row_sum = np.sum(row)
        if row_sum > 0:
            normalized_row = row / row_sum
            normalized_preference_matrix[i] = normalized_row
    
    print("--- Data processing complete ---\n")
    return normalized_rents.flatten(), normalized_preference_matrix


# =============================================================================
# 4. OPTIMIZATION MODEL DEFINITION AND SOLVING
# =============================================================================

def build_and_solve_model(params):
    """
    Builds and solves the MINLP model for a given set of parameters.

    Args:
        params (dict): A dictionary containing all necessary parameters for the model,
                       including weights, data, and dimensions.

    Returns:
        dict or None: A dictionary with the solution values for variables 'x'
                      if a feasible solution is found, otherwise None.
    """
    # Create a new SCIP model instance for this run
    model = Model("UrbanRedevelopmentModel")

    # --- A. DEFINE VARIABLES ---
    # x[i, j]: Continuous variable (0 to 1) representing the percentage of area 
    #          in story 'i' allocated to use-case 'j'.
    # z[i, j]: Binary variable (0 or 1) indicating if use-case 'j' is present 
    #          (z=1) or not (z=0) in story 'i'.
    x, z = {}, {}
    for i in range(params['m']):
        for j in range(params['n']):
            x[i, j] = model.addVar(vtype='C', name=f"x_{i}-{j}", lb=0, ub=1)
            z[i, j] = model.addVar(vtype='B', name=f"z_{i}-{j}")

    # --- B. DEFINE CONSTRAINTS ---
    
    # -- Core Structural Constraints --
    for i in range(params['m']):
        # 1. Full Area Utilization: The sum of area percentages in each story must be 100%.
        model.addCons(quicksum([x[i, j] for j in range(params['n'])]) == 1, 
                      name=f"Use100PercentAreaInStory_{i}")
        
        # 2. Limit on Distinct Use-Cases: At most U[i] different use-cases are allowed per story.
        model.addCons(quicksum([z[i, j] for j in range(params['n'])]) <= params['U'][i], 
                      name=f"AtMost_{params['U'][i]}_UsesInStory_{i}")
        
        # 3. Implication: If area is used (x > 0), the corresponding binary must be 1 (z=1).
        for j in range(params['n']):
            model.addCons(x[i, j] <= z[i, j], name=f"Implication_{i}_{j}")
    
    # -- Custom Business Logic / Realistic Constraints --
    # These constraints are based on the specific project requirements.
    # The indices [story, use_case] are hardcoded as they represent specific floors/uses.

    # Variable [0,5] is eliminated as this use-case does not exist on the ground floor.
    model.addCons(z[0, 5] == 0, name="Eliminate_Use_5_in_Story_0")
    
    # Local supply is allowed either in the basement (story 0) or ground floor (story 1), but not both.
    model.addCons(z[0, 4] + z[1, 4] <= 1, name="Exclusive_Local_Supply_Location")
    
    # Sport/Culture can be in story 3 OR story 5, but not both.
    model.addCons(z[3, 4] + z[5, 3] <= 1, name="Exclusive_Sport-Culture_Location")
    
    # Restaurants can be in story 3 OR story 5, but not both.
    model.addCons(z[3, 5] + z[5, 4] <= 1, name="Exclusive_Restaurant_Location")
    
    # If a story contains apartments, no other use-case is allowed in that story.
    # The "(n-1)" ensures if z_apt=1, the sum of other z's must be 0.
    num_other_uses = params['n'] - 1
    # Story 3 (GS3): Apartments are use-case 3
    model.addCons(z[3, 0] + z[3, 1] + z[3, 2] + z[3, 4] + z[3, 5] <= num_other_uses * (1 - z[3, 3]), name="Exclusive_Apartments_GS3")
    # Story 4 (GS4): Apartments are use-case 2
    model.addCons(z[4, 0] + z[4, 1] + z[4, 3] + z[4, 4] + z[4, 5] <= num_other_uses * (1 - z[4, 2]), name="Exclusive_Apartments_GS4")
    # Story 5 (GS5): Apartments are use-case 2
    model.addCons(z[5, 0] + z[5, 1] + z[5, 3] + z[5, 4] + z[5, 5] <= num_other_uses * (1 - z[5, 2]), name="Exclusive_Apartments_GS5")
    
    # Continuity of Living: Apartments in story 3 and 5 are only allowed if story 4 also has apartments.
    # This prevents a "sandwich" of non-residential between residential floors.
    model.addCons(z[3, 3] + z[5, 2] <= 1 + z[4, 2], name="Continuity_of_Living")
    
    # Local Supply Dependency: Local supply in story 3 requires some form of local supply in story 2.
    model.addCons(z[3, 0] <= z[2, 0] + z[2, 1] + z[2, 2] + z[2, 3], name="Local_Supply_Dependency_GS3_on_GS2")
    
    # Exclusive Use Dependency for Rooftop: Exclusive use on story 6 requires office/apartments on story 5.
    model.addCons(z[6, 4] <= z[5, 0] + z[5, 1] + z[5, 2], name="Exclusive_Use_Dependency_GS6_on_GS5")

    # --- C. DEFINE OBJECTIVE FUNCTION ---
    # The objective is to minimize a weighted sum of two terms:
    # 1. Negative Rental Income (i.e., maximize rent).
    # 2. Quadratic deviation from citizen preferences.

    # Create a helper variable for the nonlinear preference term.
    obj_pref_term = model.addVar(vtype='C', name="obj_preference_term")
    
    # The vector of area variables x, excluding the non-existent one at [0,5].
    x_vector = [x[i, j] for i in range(params['m']) for j in range(params['n']) if not (i == 0 and j == 5)]

    # Define the preference term via a constraint.
    # This term calculates the weighted sum of squared differences between the
    # solution's area allocation (x) and the citizens' preferred allocation (A).
    model.addCons(quicksum([params['w_story'][i] * quicksum([(x[i, j] - params['A'][i][j])**2 for j in range(params['n'])]) 
                            for i in range(params['m'])]) == obj_pref_term,
                  name="Define_Preference_Objective")
    
    # Combine both terms into the final objective function.
    model.setObjective(
        params['c_pref'] * obj_pref_term - 
        params['c_rent'] * quicksum([params['M_flat'][k] * x_vector[k] for k in range(len(x_vector))]),
        "minimize"
    )

    # --- D. SOLVE THE MODEL ---
    # Set solver time limit.
    model.setParam('limits/time', params['time_limit'])
    
    # Optionally write model to file for debugging.
    # model.writeProblem(filename=f"{params['debug_prefix']}_{params['run_k']}.cip")

    model.optimize()

    # --- E. EXTRACT SOLUTION ---
    if model.getStatus() in ["optimal", "timelimit", "userinterrupt"]:
        if model.getNSols() > 0:
            solution = {}
            best_sol = model.getBestSol()
            for i in range(params['m']):
                for j in range(params['n']):
                    solution[i,j] = model.getSolVal(best_sol, x[i,j])
            return solution
    
    # If no feasible solution was found
    return None


# =============================================================================
# 5. RESULTS EXPORT
# =============================================================================

def format_and_save_results(solutions_dict, sensitivity_labels, m, n, filename):
    """
    Formats the collected solutions into a pandas DataFrame and saves it to Excel.

    Args:
        solutions_dict (dict): Dictionary containing the solution values for each run.
        sensitivity_labels (list): List of tuples with the weights for each run.
        m (int): Number of stories.
        n (int): Number of use-cases.
        filename (str): Path for the output Excel file.
    """
    print(f"\n--- Formatting results and saving to {filename} ---")
    
    # Define the desired order of rows (variables) in the final table.
    index_order = [f"[{i},{j}]" for i in range(m) for j in range(n)]

    # Create a DataFrame from the solutions dictionary.
    # Each column represents a sensitivity run, each row a variable x[i,j].
    df_results = pd.DataFrame(solutions_dict, index=range(len(sensitivity_labels))).T
    
    # Create a MultiIndex for the columns to clearly label each sensitivity run.
    multi_columns = pd.MultiIndex.from_tuples(sensitivity_labels, names=["weight_preference", "weight_rent"])
    df_results.columns = multi_columns
    
    # Reorder the rows to have a consistent [0,0], [0,1], ... order.
    df_results = df_results.loc[index_order]

    # Save the DataFrame to an Excel file.
    df_results.to_excel(filename)
    print("--- Results saved successfully ---")


# =============================================================================
# 6. MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    
    # --- Step 1: Load and prepare data ---
    rents_flat, preferences_matrix = load_and_preprocess_data(
        filename=INPUT_DATA_FILE,
        start_row=DATA_START_ROW,
        rents_col=RENTS_COL_IDX,
        pref_col=PREFERENCES_COL_IDX,
        m=NUM_STORIES,
        n=NUM_USE_CASES
    )
    
    # --- Step 2: Initialize structures for sensitivity analysis ---
    all_run_solutions = {}
    for i in range(NUM_STORIES):
        for j in range(NUM_USE_CASES):
            label = f"[{i},{j}]"
            all_run_solutions[label] = []

    sensitivity_run_labels = list(zip(WEIGHT_PREFERENCES_LIST, WEIGHT_RENTS_LIST))
    
    # Determine if this is a single run or a sensitivity analysis to set time limit
    is_sensitivity_run = len(WEIGHT_PREFERENCES_LIST) > 1
    time_limit = SOLVER_TIME_LIMIT_SENSITIVITY if is_sensitivity_run else SOLVER_TIME_LIMIT_FULL

    # --- Step 3: Run the optimization for each set of weights ---
    for k, (c_pref, c_rent) in enumerate(sensitivity_run_labels):
        print(f"--- Running Sensitivity Analysis: Run {k+1}/{len(sensitivity_run_labels)} ---")
        print(f"    Weight Preference (c_pref): {c_pref}, Weight Rent (c_rent): {c_rent}")
        
        # Package all parameters for the solver function
        model_params = {
            'm': NUM_STORIES,
            'n': NUM_USE_CASES,
            'A': preferences_matrix,
            'M_flat': rents_flat,
            'w_story': STORY_IMPORTANCE_WEIGHTS,
            'U': MAX_DISTINCT_USES_PER_STORY,
            'c_pref': c_pref,
            'c_rent': c_rent,
            'time_limit': time_limit,
            'debug_prefix': MODEL_DEBUG_FILE_PREFIX,
            'run_k': k
        }
        
        # Build and solve the model
        solution = build_and_solve_model(model_params)
        
        # Store the results of the run
        for i in range(NUM_STORIES):
            for j in range(NUM_USE_CASES):
                label = f"[{i},{j}]"
                if solution:
                    sol_val = solution.get((i, j), np.nan)
                    all_run_solutions[label].append(sol_val)
                else:
                    # Append NaN if the model was infeasible or no solution was found
                    all_run_solutions[label].append(np.nan)
                    
    # --- Step 4: Save the collected results to an Excel file ---
    format_and_save_results(
        solutions_dict=all_run_solutions,
        sensitivity_labels=sensitivity_run_labels,
        m=NUM_STORIES,
        n=NUM_USE_CASES,
        filename=OUTPUT_SOLUTION_FILE
    )

