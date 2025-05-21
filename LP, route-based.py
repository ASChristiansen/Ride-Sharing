import gurobipy as gp
from gurobipy import GRB
import math
import pandas as pd
import time
import os
import itertools

# Import data extraction function
from Data_from_txt import extract_sets

# Base directory
base_path = r"C:\Users\Amand\OneDrive\Dokumenter\Københavns Universitet\3. år\Bachelorprojekt\Large instances"

# Variable for the specific instance name (Modify this when needed)
instance_name = "A-n39-k13"

# Construct full file path
filename = os.path.join(base_path, f"{instance_name}.txt")

# Extract variables from the dataset
K, d, P, T_U, Q, A, Q_max, vehicle_coords, customer_coords = extract_sets(filename)

T_k = 0  # Vehicle availability time

# Start timing the optimization and impose limit
start_time = time.time()

# Distance calculation function
def euclidean_d(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

def get_coord(i):
    return vehicle_coords[i] if i in vehicle_coords else customer_coords[i]

C = {
    (i, j): math.floor(euclidean_d(get_coord(i), customer_coords[j]))
    for i in K + P
    for j in P + [d]
}

# We define all subsets of P, i.e. the powerset
from itertools import permutations, chain, combinations

# Helper: powerset (all subsets)
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

def generate_all_routes(P, d, Q_max):
    all_routes = []

    all_routes.append([]) # 1. Stay put
    all_routes.append([d]) # 2. Go directly to depot

    # 3. Pickup subsets (1 or more) → permutations + depot
    for subset in powerset(P):
        if len(subset) <= Q_max:
            for pickup_order in permutations(subset):
                route = list(pickup_order) + [d]
                all_routes.append(route)

    return all_routes

# Computing route costs
def route_only_cost(r, C):
    if not r or r == [d]:
        return 0  # stay put
    route_cost = 0
    for j in range(len(r) - 1):
        route_cost += C[(r[j], r[j + 1])]
    return route_cost

# Route + vehicle costs
def cost(k, r, C, d):
    if not r:
        return 0  # stay put
    if r == [d]:
        return C[(k, d)]  # go straight to depot

    return C[(k, r[0])] + route_only_cost(r, C)

# Time feasible routes
def time_feasible(k, r, route_cost, A, d):
    if not r or r == [d]:
        return route_cost <= A[k]  # only vehicle's own constraint applies

    latest_allowed = min([A[k]] + [A[node] for node in r if node != d])
    return route_cost <= latest_allowed

feasible_routes = {}

# Generate common routes once
all_possible_routes = generate_all_routes(P, d, Q_max)

for k in K:
    feasible_routes[k] = []
    for r in all_possible_routes:
        if len([n for n in r if n != d]) <= Q_max - Q[k]:  # enforce vehicle-specific residual capacity
            route_cost = cost(k, r, C, d)
            if time_feasible(k, r, route_cost, A, d):
                feasible_routes[k].append(r)

    print(f"\nFeasible (time + capacity) routes for vehicle {k}:")
    for r in feasible_routes[k]:
        full_route = [k] + r  # prepend the vehicle to the route
        route_str = " → ".join(full_route)
        route_cost = cost(k, r, C, d)
        print(f"  Route: {route_str} | Cost: {route_cost}")

# Implementing the model

# Gurobi model
model = gp.Model("Route_Assignment")

# Decision variables
y = {}  # (k, route as tuple) → variable

for k in K:
    for r in feasible_routes[k]:
        r_tuple = tuple(r)
        y[k, r_tuple] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"y_{k}_{'_'.join(r_tuple) if r else 'stay'}")

# Calculating revenue
def rev(r, C, d):
    if not r or r == [d]:
        return 0

    customers = [i for i in r if i != d]
    return 2 * sum(C[(i, d)] for i in customers)

route_revenue = {
    (k, tuple(r)): rev(r, C, d)
    for k in K
    for r in feasible_routes[k]
}

# Objective function
OF = gp.quicksum(cost(k, list(r), C, d) * y[k, r] for (k, r) in y) \
     - gp.quicksum(rev(list(r), C, d) * y[k, r] for (k, r) in y)

model.setObjective(OF, GRB.MINIMIZE)

# Constraints
## Vehicle capacity is respected
for (k, r) in y:
    pickups = len([i for i in r if i != d])
    if pickups > Q_max - Q[k]:
        model.addConstr(y[k, r] == 0,
        name=f"capacity_violation_{k}_{'_'.join(r)}"
    )

## Each vehicle follows at most one route
for k in K:
    model.addConstr(
        gp.quicksum(y[k, r] for (vk, r) in y if vk == k) <= 1,
        name=f"one_route_{k}"
    )

## Each customer is picked up at most once
for p in P:
    model.addConstr(
        gp.quicksum(
            y[k, r] for (k, r) in y if p in r and p != d
        ) <= 1,
        name=f"pickup_once_{p}"
    )

## If a vehicle has passengers onboard at time 0, it must follow a route
for k in K:
    if Q[k] > 0:
        model.addConstr(
            gp.quicksum(y[k, r] for (vk, r) in y if vk == k and r !=()) >= 1,
            name=f"must_use_if_loaded_{k}"
        )

# Solving time limit
model.setParam('TimeLimit', 1800)

solver_start_time = time.time()
# Optimize
model.optimize()

# Calculate solving time
solver_time = time.time() - solver_start_time
computation_time = solver_start_time - start_time
total_runtime = time.time() - start_time

# Solution details
solution_summary = []
routes = []

# Capture objective info if available
if model.SolCount > 0:
    best_obj = model.objVal
    best_bound = model.ObjBound
#    gap = model.MIPGap * 100  # %

    solution_summary.append(["Solution status", model.Status])
    solution_summary.append(["Best Objective", best_obj])
    solution_summary.append(["Best Bound", best_bound])
#    solution_summary.append(["Gap (%)", f"{gap:.4f}%"])
    solution_summary.append(["Gurobi Solver Time (s)", f"{solver_time:.2f}"])
    solution_summary.append(["Computation Time (s)", f"{computation_time:.2f}"])
    solution_summary.append(["Total Runtime (s)", f"{total_runtime:.2f}"])

    print(f"\nModel solved. Status: {model.Status}")
    print(f"Best objective: {best_obj:.2f}, Best bound: {best_bound:.2f}")

    # Extract active routes
    for (k, r), var in y.items():
        if var.X > 0.5:
            route_list = [k] + list(r)
            route_str = " → ".join(route_list)
            c = cost(k, list(r), C, d)
            revenue = rev(list(r), C, d)
            profit = revenue - c
            routes.append([k, route_str, c, revenue, profit])
else:
    # If no feasible solution found
    solution_summary.append(["Solution status", model.Status])
    solution_summary.append(["Best Objective", "N/A"])
    solution_summary.append(["Best Bound", model.ObjBound])
#    solution_summary.append(["Gap (%)", "N/A"])
    solution_summary.append(["Gurobi Solver Time (s)", f"{solver_time:.2f}"])
    solution_summary.append(["Total Runtime (s)", f"{total_runtime:.2f}"])

    print(f"\n⚠️ No feasible solution found. Status: {model.Status}")

# ✅ Convert to DataFrames
df_solution = pd.DataFrame(solution_summary, columns=["Info", "Value"])
df_routes = pd.DataFrame(routes, columns=["Vehicle", "Route", "Cost", "Revenue", "Profit"])

# ✅ Reset index
df_solution.reset_index(drop=True, inplace=True)
df_routes.reset_index(drop=True, inplace=True)

# ✅ Add blank row for spacing
spacing_row = pd.DataFrame([["", "", "", "", ""]], columns=df_routes.columns)

# ✅ Combine summary and routes into one sheet
df_combined = pd.concat([df_solution, spacing_row, df_routes], ignore_index=True)

# ✅ Define Excel file path and sheet name
output_file = r"C:\Users\Amand\OneDrive\Dokumenter\Københavns Universitet\3. år\Bachelorprojekt\LP_Route_Based.xlsx"

# ✅ Export to Excel (append or create)
try:
    with pd.ExcelWriter(output_file, engine="openpyxl", mode="a", if_sheet_exists="new") as writer:
        df_combined.to_excel(writer, sheet_name=instance_name, index=False)
except FileNotFoundError:
    with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
        df_combined.to_excel(writer, sheet_name=instance_name, index=False)

print(f"Results for {instance_name} exported successfully to {output_file} (Sheet: {instance_name})")



