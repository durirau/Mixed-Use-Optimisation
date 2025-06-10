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
@version: 1.1 (Variant A: rents weight = 1 - preference weight)
"""

# =============================================================================
# 1. IMPORTS - Libraries required for the script
# =============================================================================
import pandas as pd  # Used for reading and handling data from Excel files.
import numpy as np   # Used for numerical operations, especially array/matrix manipulation.
from pyscipopt import Model, quicksum  # Core components from the SCIP solver interface.

# =============================================================================
# 2. CONFIGURATION - All parameters and settings for the model
# =============================================================================

# --- File Paths ---
INPUT_DATA_FILE             = 'preferences-and-rents.xlsx'
DATA_SHEET                  = 'PreferencesRents'
OUTPUT_SOLUTION_FILE        = 'Solution_Analysis.xlsx'
MODEL_DEBUG_FILE_PREFIX     = 'model_Kaufhaus'

# --- Model Dimensions ---
NUM_STORIES                 = 7       # Total number of stories in the building (indexed 0 to 6).
NUM_USE_CASES               = 6       # Number of different use-cases per story (indexed 0 to 5).

# --- Data Reading Parameters ---
DATA_START_ROW              = 1       # Number of lines to skip before data (Excel row 2 header, data from row 3)
PREFERENCES_COL_IDX         = 2       # Column C (0-based index = 2)
RENTS_COL_IDX               = 6       # Column G (0-based index = 6)

# --- Model Parameters ---
STORY_IMPORTANCE_WEIGHTS    = [0.1467, 0.162, 0.138, 0.1226, 0.1158, 0.135, 0.1821]
MAX_DISTINCT_USES_PER_STORY = NUM_STORIES * [4]

# --- Sensitivity Analysis Parameters (Variant A) ---
WEIGHT_PREFERENCES_LIST     = [0.3,  0.35, 0.4,  0.45, 0.5,  0.55, 0.6, 0.65, 0.7, 1.0]
WEIGHT_RENTS_LIST           = [1 - w for w in WEIGHT_PREFERENCES_LIST]

# --- Solver Settings ---
SOLVER_TIME_LIMIT_FULL      = 64800   # 18 hours
SOLVER_TIME_LIMIT_SENSITIVITY = 1000  # ~16 minutes

# =============================================================================
# 3. DATA LOADING AND PREPROCESSING
# =============================================================================
def load_and_preprocess_data(filename, sheet_name,
                             start_row, rents_col, pref_col,
                             m, n):
    """
    Lädt Präferenzen und Mieten aus dem Excel-Sheet und 
    bereitet sie für das Modell auf.

    Args:
        filename (str): Pfad zur Excel-Datei.
        sheet_name (str): Name des Worksheets mit den Daten.
        start_row (int): Anzahl der zu überspringenden Zeilen vor den Daten.
        rents_col (int): Spaltenindex für die Mieten.
        pref_col (int): Spaltenindex für die Präferenzen.
        m (int): Anzahl der Stockwerke.
        n (int): Anzahl der Use-Cases pro Stockwerk.

    Returns:
        tuple: (normierte Mieten, normierte Präferenzmatrix)
    """
    print("--- Loading and preprocessing data ---")
    df_rents = pd.read_excel(
        filename,
        sheet_name=sheet_name,
        skiprows=start_row,
        usecols=[rents_col]
    )
    rents = df_rents.values.astype(float)
    normalized_rents = rents / rents.sum()
    df_prefs = pd.read_excel(
        filename,
        sheet_name=sheet_name,
        skiprows=start_row,
        usecols=[pref_col]
    )
    prefs = df_prefs.values.astype(float)
    raw_A = np.zeros((m, n))
    raw_A[0, :n-1] = prefs[:n-1, 0]
    for i in range(1, m):
        start = (n-1) + (i-1)*n
        raw_A[i] = prefs[start:start+n, 0]
    A = np.array([row/row.sum() if row.sum()>0 else row for row in raw_A])
    print("--- Data processing complete ---\n")
    return normalized_rents.flatten(), A

# =============================================================================
# 4. OPTIMIZATION MODEL DEFINITION AND SOLVING
# =============================================================================
def build_and_solve_model(params):
    model = Model("UrbanRedevelopmentModel")
    x, z = {}, {}
    for i in range(params['m']):
        for j in range(params['n']):
            x[i, j] = model.addVar(vtype='C', name=f"x_{i}-{j}", lb=0, ub=1)
            z[i, j] = model.addVar(vtype='B', name=f"z_{i}-{j}")
    for i in range(params['m']):
        model.addCons(quicksum(x[i, j] for j in range(params['n'])) == 1, name=f"Use100PercentAreaInStory_{i}")
        model.addCons(quicksum(z[i, j] for j in range(params['n'])) <= params['U'][i], name=f"AtMost_{params['U'][i]}_UsesInStory_{i}")
        for j in range(params['n']): model.addCons(x[i, j] <= z[i, j], name=f"Implication_{i}_{j}")
    model.addCons(z[0, 5] == 0, name="Eliminate_Use_5_in_Story_0")
    model.addCons(z[0, 4] + z[1, 4] <= 1, name="Exclusive_Local_Supply_Location")
    model.addCons(z[3, 4] + z[5, 3] <= 1, name="Exclusive_Sport-Culture_Location")
    model.addCons(z[3, 5] + z[5, 4] <= 1, name="Exclusive_Restaurant_Location")
    num_other_uses = params['n'] - 1
    model.addCons(z[3, 0] + z[3, 1] + z[3, 2] + z[3, 4] + z[3, 5] <= num_other_uses * (1 - z[3, 3]), name="Exclusive_Apartments_GS3")
    model.addCons(z[4, 0] + z[4, 1] + z[4, 3] + z[4, 4] + z[4, 5] <= num_other_uses * (1 - z[4, 2]), name="Exclusive_Apartments_GS4")
    model.addCons(z[5, 0] + z[5, 1] + z[5, 3] + z[5, 4] + z[5, 5] <= num_other_uses * (1 - z[5, 2]), name="Exclusive_Apartments_GS5")
    model.addCons(z[3, 3] + z[5, 2] <= 1 + z[4, 2], name="Continuity_of_Living")
    model.addCons(z[3, 0] <= z[2, 0] + z[2, 1] + z[2, 2] + z[2, 3], name="Local_Supply_Dependency_GS3_on_GS2")
    model.addCons(z[6, 4] <= z[5, 0] + z[5, 1] + z[5, 2], name="Exclusive_Use_Dependency_GS6_on_GS5")
    obj_pref_term = model.addVar(vtype='C', name="obj_preference_term")
    x_vector = [x[i, j] for i in range(params['m']) for j in range(params['n']) if not (i == 0 and j == 5)]
    model.addCons(quicksum(params['w_story'][i] * quicksum((x[i, j] - params['A'][i][j])**2 for j in range(params['n'])) for i in range(params['m'])) == obj_pref_term, name="Define_Preference_Objective")
    model.setObjective(params['c_pref'] * obj_pref_term - params['c_rent'] * quicksum(params['M_flat'][k] * x_vector[k] for k in range(len(x_vector))), "minimize")
    model.setParam('limits/time', params['time_limit'])
    model.optimize()
    if model.getStatus() in ["optimal", "timelimit", "userinterrupt"] and model.getNSols() > 0:
        solution = {}
        best_sol = model.getBestSol()
        for i in range(params['m']):
            for j in range(params['n']): solution[i, j] = model.getSolVal(best_sol, x[i, j])
        return solution
    return None

# =============================================================================
# 5. RESULTS EXPORT
# =============================================================================
def format_and_save_results(solutions_dict, sensitivity_labels,
                             valid_combos, floors, usecases,
                             filename):
    print(f"\n--- Formatting results and saving to {filename} ---")
    df = pd.DataFrame(solutions_dict, index=range(len(sensitivity_labels))).T
    multi_cols = pd.MultiIndex.from_tuples(sensitivity_labels, names=["weight_preference","weight_rent"])
    df.columns = multi_cols
    labels = [f"{floors[idx]}: {usecases[idx]}" for idx in range(len(valid_combos))]
    df.index = labels
    df.to_excel(filename)
    print("--- Results saved successfully ---")

# =============================================================================
# 6. MAIN EXECUTION BLOCK
# =============================================================================
if __name__ == "__main__":
    rents_flat, preferences_matrix = load_and_preprocess_data(
        filename=INPUT_DATA_FILE,
        sheet_name=DATA_SHEET,
        start_row=DATA_START_ROW,
        rents_col=RENTS_COL_IDX,
        pref_col=PREFERENCES_COL_IDX,
        m=NUM_STORIES,
        n=NUM_USE_CASES
    )
    df_meta = pd.read_excel(
        INPUT_DATA_FILE,
        sheet_name=DATA_SHEET,
        skiprows=DATA_START_ROW,
        usecols=[0, 1],
        header=None,
        names=["Floor","Useclass"]
    )
    floors   = df_meta["Floor"].astype(str).tolist()
    usecases = df_meta["Useclass"].tolist()
    valid_combos = [(i, j) for i in range(NUM_STORIES) for j in range(NUM_USE_CASES) if not (i == 0 and j == NUM_USE_CASES-1)]
    all_run_solutions = { f"[{i},{j}]": [] for (i,j) in valid_combos }
    sensitivity_run_labels = list(zip(WEIGHT_PREFERENCES_LIST, WEIGHT_RENTS_LIST))
    time_limit = SOLVER_TIME_LIMIT_SENSITIVITY if len(sensitivity_run_labels) > 1 else SOLVER_TIME_LIMIT_FULL
    for k, (c_pref, c_rent) in enumerate(sensitivity_run_labels):
        sol = build_and_solve_model({
            'm': NUM_STORIES, 'n': NUM_USE_CASES,
            'A': preferences_matrix, 'M_flat': rents_flat,
            'w_story': STORY_IMPORTANCE_WEIGHTS, 'U': MAX_DISTINCT_USES_PER_STORY,
            'c_pref': c_pref, 'c_rent': c_rent,
            'time_limit': time_limit,
            'debug_prefix': MODEL_DEBUG_FILE_PREFIX, 'run_k': k
        })
        for (i,j) in valid_combos:
            label = f"[{i},{j}]"
            all_run_solutions[label].append(sol.get((i,j), np.nan) if sol else np.nan)
    format_and_save_results(
        solutions_dict     = all_run_solutions,
        sensitivity_labels = sensitivity_run_labels,
        valid_combos       = valid_combos,
        floors             = floors,
        usecases           = usecases,
        filename           = OUTPUT_SOLUTION_FILE
    )
