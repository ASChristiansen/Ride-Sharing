import os
# Base directory
base_path = r"C:\Users\Amand\OneDrive\Dokumenter\Københavns Universitet\3. år\Bachelorprojekt\Large instances"

# Variable for the specific instance name (Modify this when needed)
instance_name = "P-n15-k5"

# Construct full file path
filename = os.path.join(base_path, f"{instance_name}.txt")

def extract_sets(filename):
    vehicle_count = None
    customer_count = None
    T_U = None

    # First pass: extract VEHICLES, CUSTOMERS, and TIME UB values
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("VEHICLES:"):
                parts = line.split()
                if len(parts) > 1:
                    vehicle_count = int(parts[1])
            elif line.startswith("CUSTOMERS:"):
                parts = line.split()
                if len(parts) > 1:
                    customer_count = int(parts[1])
            elif line.startswith("TIME UB:"):
                parts = line.split()
                if len(parts) > 1:
                    T_U = int(parts[-1])
            if vehicle_count is not None and customer_count is not None and T_U is not None:
                break

    # Define basic sets:
    K = [f"v{i}" for i in range(vehicle_count)]
    d = "c0"
    P = [f"c{i}" for i in range(1, customer_count + 1)]

    # Read all lines for further processing
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Second pass: process the VEHICLE SECTION
    Q = {}  # Vehicle occupancy: {'v0': occupancy, ...}
    A_v = {}  # Vehicle arrival times: {'v0': arrival_time, ...}
    vehicle_coords = {}  # Vehicle coordinates: {'v0': (x, y), ...}
    Q_max = 0  # Maximum vehicle capacity

    vehicle_section_index = None
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("VEHICLE SECTION"):
            vehicle_section_index = i
            break

    if vehicle_section_index is not None:
        # Assuming the header is the next line, data starts two lines after the section title
        vehicle_data_start = vehicle_section_index + 2
        for j in range(vehicle_count):
            data_line = lines[vehicle_data_start + j].strip()
            tokens = data_line.split()
            # Expected tokens: [vehicle_no, X_COORD, Y_COORD, CAPACITY, OCCUPANCY, Arr. T]
            x_coord = int(tokens[1])
            y_coord = int(tokens[2])
            capacity = int(tokens[3])
            occupancy = int(tokens[4])
            arrival_time_vehicle = int(tokens[5])
            Q[f"v{j}"] = occupancy
            A_v[f"v{j}"] = arrival_time_vehicle
            vehicle_coords[f"v{j}"] = (x_coord, y_coord)
            if capacity > Q_max:
                Q_max = capacity

    # Third pass: process the CUSTOMER SECTION (includes depot and customers)
    A_c = {}  # Customer arrival times: {'c0': arrival_time, ...}
    customer_coords = {}  # Customer coordinates: {'c0': (x, y), ...}

    customer_section_index = None
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("CUSTOMER SECTION"):
            customer_section_index = i
            break

    if customer_section_index is not None:
        # Assuming the header is the next line, data starts two lines after the section title
        customer_data_start = customer_section_index + 2
        # There are (customer_count + 1) lines: depot (customer 0) plus the customers
        num_customer_lines = customer_count + 1
        for j in range(num_customer_lines):
            data_line = lines[customer_data_start + j].strip()
            tokens = data_line.split()
            # Expected tokens: [CUST. NO., X_COORD, Y_COORD, DEMAND, Arr. T]
            cust_no = int(tokens[0])
            x_coord = int(tokens[1])
            y_coord = int(tokens[2])
            arrival_time_customer = int(tokens[4])
            A_c[f"c{cust_no}"] = arrival_time_customer
            customer_coords[f"c{cust_no}"] = (x_coord, y_coord)

    # Combine the vehicle and customer arrival times into a single dictionary A:
    A = {}
    A.update(A_v)
    A.update(A_c)

    # Print the extracted sets for verification:
    print("K (Vehicles):", K)
    print("d (Depot):", d)
    print("P (Customers):", P)
    print("T_U (Time UB):", T_U)
    print("Q (Vehicle Occupancy):", Q)
    print("A (Latest Arrival Times):", A)
    print("Q_max (Max Vehicle Capacity):", Q_max)
    print("vehicle_coords (Vehicle Coordinates):", vehicle_coords)
    print("customer_coords (Customer Coordinates):", customer_coords)

    return K, d, P, T_U, Q, A, Q_max, vehicle_coords, customer_coords

if __name__ == "__main__":
    extract_sets(filename)
