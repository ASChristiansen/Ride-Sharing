import gurobipy as gp
from gurobipy import GRB
import math
import pandas as pd
import time
import os

# Import data extraction function
from Data_from_txt import extract_sets

# Base directory
base_path = r"C:\Users\Amand\OneDrive\Dokumenter\Københavns Universitet\3. år\Bachelorprojekt\Large instances"

# Variable for the specific instance name (Modify this when needed)
instance_name = "B-n39-k12"

# Construct full file path
filename = os.path.join(base_path, f"{instance_name}.txt")

# Extract variables from the dataset
K, d, P, T_U, Q, A, Q_max, vehicle_coords, customer_coords = extract_sets(filename)

T_k = 0  # Vehicle availability time

# Start timing the optimization
start_time = time.time()

# Distance calculation function
def euclidean_d(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

def get_coord(i):
    return vehicle_coords[i] if i in vehicle_coords else customer_coords[i]

# Distance and time matrices
# Define parameters
C = {
    (i, j): math.floor(euclidean_d(get_coord(i), customer_coords[j]))
    for i in K + P
    for j in P + [d]
}

T = {(i, j): C[i, j] for (i, j) in C}

r = {i: 2 * C[i, d] for i in P}  # Revenue at each node

# Gurobi model
model = gp.Model("Minimize_cost")

# Decision variables
x = model.addVars(K, K + P, P + [d], vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="x")
q = model.addVars(P + K, vtype=GRB.CONTINUOUS, lb=0.0, name="q")

# Objective function
OF = gp.quicksum(C[i, j] * x[k, i, j] for k in K for i in K + P for j in P + [d] if i != j) \
     - gp.quicksum(r[i] * x[k, i, j] for k in K for i in P for j in P + [d])
model.setObjective(OF, GRB.MINIMIZE)

# Constraints (as defined earlier)

## 4c
for k in K:
    model.addConstr(
        gp.quicksum(x[k, k, j] for j in P + [d]) <= 1,
        name=f"4c_{k}"
    )

## 4d
for k in K:
    model.addConstr(
        gp.quicksum(x[k, k, j] for j in P + [d]) ==
        gp.quicksum(x[k, i, d] for i in [k] + P),
        name=f"4d_{k}"
    )

## 4e
for j in P:
    for k in K:
         model.addConstr(
             gp.quicksum(x[k, i, j] for i in [k] + P) ==
             gp.quicksum(x[k, j, i] for i in P + [d]),
             name=f"4e_{j}_{k}"
         )

## 4f
for i in K + P:
    for j in P:
        if i != j:
            model.addConstr(
                q[j] >= q[i] - (len(P) + 1)*(1 - gp.quicksum(x[k, i, j] for k in K)) + 1,
                name=f"4f_{i}_{j}"
            )

## 4g
for i in P:
    model.addConstr(
        gp.quicksum(x[k, i, j] for k in K for j in P + [d]) <= 1,
        name=f"4g_{i}"
    )

## 4h
for k in K:
    model.addConstr(
         gp.quicksum(x[k, i, j] for i in P for j in P + [d]) <= Q_max - Q[k],
         name=f"4h_{k}"
    )

## 4i
for k in K:
    model.addConstr(
         Q[k] <= gp.quicksum(Q[k] * x[k, k, j] for j in P + [d]),
         name=f"4i_{k}"
    )

## 4j
for i in P:
    for k in K:
        model.addConstr(
            T_k + gp.quicksum(T[i, j] * x[k, i, j] for i in [k] + P for j in P + [d])
            <= min(A[i], A[k]) + (A[k] - min(A[i], A[k])) * (1 - gp.quicksum(x[k, j, i] for j in P + [k])),
           name=f"4j_{i}_{k}"
        )

## 4k
#x = model.addVars(K, K + P, P + [d], vtype=GRB.BINARY, name="4k_x")

## 4l
for i in K + P:
    model.addConstr(
        q[i] >= 0,
        name=f"4l_lower_{i}"
    )
    model.addConstr(
        q[i] <= min(len(P), max(Q_max - Q[k] for k in K)),
        name=f"4l_upper_{i}"
    )

## 4m
for i in P:  # i is a customer (present in both origin and destination sets)
    for k in K:
         model.addConstr(x[k, i, i] == 0, name=f"no_self_{i}_{k}")

# Solving time limit
model.setParam('TimeLimit', 1800)

# Optimize
model.optimize()

# Calculate solving time
solving_time = time.time() - start_time

# Extract solution details
solution_summary = []
routes = []

if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
    best_obj = model.objVal
    best_bound = model.ObjBound
#    gap = model.MIPGap * 100  # Convert to percentage

    if model.status == GRB.OPTIMAL:
        solution_summary.append(["Optimal solution found", ""])
        print("Optimal solution found!")
    elif model.status == GRB.TIME_LIMIT:
        solution_summary.append(["Time limit reached", ""])
        print("Time limit reached – best feasible solution found!")

    # Store summary info
    solution_summary.append(["Best Objective", best_obj])
    solution_summary.append(["Best Bound", best_bound])
#    solution_summary.append(["Gap (%)", f"{gap:.4f}%"])
    solution_summary.append(["Solving Time (s)", f"{solving_time:.2f}"])

    print(f"Best objective {best_obj:.12e}, best bound {best_bound:.12e}")

    # Collect only active routes
    for k in K:
        for i in K + P:
            for j in P + [d]:
                if (i, j) in C and x[k, i, j].x > 0.5:
                    routes.append([f"{k}", f"{k}" if i in K else f"{i}", f"{j}" if j in P else f"d", round(C[i, j], 2)])
                    print(f"Vehicle {k} travels from {i} to {j} with cost {C[i, j]:.2f}")
else:
    print("No feasible solution found.")
    solution_summary.append(["No feasible solution", ""])

# ✅ Convert to DataFrame
df_solution = pd.DataFrame(solution_summary, columns=["Info", "Value"])  # Solution summary
df_routes = pd.DataFrame(routes, columns=["Vehicle", "From", "To", "Travel Cost"])  # Vehicle routes

# ✅ Reset index to prevent duplicate index errors
df_solution.reset_index(drop=True, inplace=True)
df_routes.reset_index(drop=True, inplace=True)

# ✅ Add a blank row for spacing
spacing_row = pd.DataFrame([["", "", "", ""]], columns=df_routes.columns)

# ✅ Combine solution summary and routes in one sheet
df_combined = pd.concat([df_solution, spacing_row, df_routes], ignore_index=True)

# Define output file location
output_file = r"C:\Users\Amand\OneDrive\Dokumenter\Københavns Universitet\3. år\Bachelorprojekt\LP_Vehicle_Flow.xlsx"

# Load existing Excel file if it exists, otherwise create a new one
try:
    with pd.ExcelWriter(output_file, engine="openpyxl", mode="a", if_sheet_exists="new") as writer:
        df_combined.to_excel(writer, sheet_name=instance_name, index=False)
except FileNotFoundError:
    with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
        df_combined.to_excel(writer, sheet_name=instance_name, index=False)

print(f"Results for {instance_name} exported successfully to {output_file} (Sheet: {instance_name})")
