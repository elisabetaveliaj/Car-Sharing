import smopy
from elasticsearch import Elasticsearch
from gurobipy import Model, GRB
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import numpy as np
import json


# Function to check compatibility of a car with a given request based on walking distance
def is_compatible(car_index, request_index, walking_distance, distances):
    if distances[car_index, request_index] <= walking_distance:
        return True
    else:
        return False


# Function to create the optimization model
def optimization(cars_data, requests_data, walking_distance):
    # Initialize model
    m = Model("Car_sharing")

    # Decision variables
    n_cars = len(cars_data)
    n_requests = len(requests_data)
    x = m.addVars(n_cars, n_requests, vtype=GRB.BINARY, name="assign")

    # Objective function to maximize total profit
    m.setObjective(sum(x[c, j] * requests_data[j]['profit'] for c in range(n_cars) for j in range(n_requests)),
                       GRB.MAXIMIZE)

    # Constraints
    # each request is assigned to most one car
    for j in range(n_requests):
        m.addConstr(sum(x[c, j] for c in range(n_cars)) <= 1)

    # each car is assigned to most one request
    for c in range(n_cars):
        m.addConstr(sum(x[c, j] for j in range(n_requests)) <= 1)

    # Compatibility constraint:
    # A request is only assigned to a car if the car's position is within the walking distance from the request's origin
    for j in range(n_requests):
        for c in range(n_cars):
            if not is_compatible(c, j, walking_distance, distances):
                m.addConstr(x[c, j] == 0)

    return m, x


# Initialize Elasticsearch client
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
index_name = "car_requests"
sample_size = 2455

# Load car data
cars_data = []
with open("car_locations.json", 'r') as file:
    for line in file:
        car = json.loads(line)

        # Swap longitude and latitude for the first 200 data points
        if len(cars_data) < 200:
            car['start_location'] = [car['start_location'][1], car['start_location'][0]]
        cars_data.append(car)

# querying data from ES
query = {
    "size": sample_size,
    "query": {
        "match_all": {}
    }
}

response = es.search(index=index_name, body=query)
requests_data = []

for hit in response['hits']['hits']:
    request = hit['_source']
    origin_location = tuple(request['origin_location'])
    destination_location = tuple(request['destination_location'])

    requests_data.append({
        "request_id": hit['_id'],
        "origin_location": origin_location,
        "destination_location": destination_location,
        "walking_distance": 0.4,
        "profit": 1
    })

# Average speed in km/h and revenue rate in € per minute
average_speed_kmh = 50
revenue_rate = 0.19

# Calculate profit for each request
for request in requests_data:
    distance_km = geodesic(request['origin_location'], request['destination_location']).kilometers
    travel_time_hours = distance_km / average_speed_kmh
    travel_time_minutes = np.ceil(travel_time_hours * 60)
    request['profit'] = travel_time_minutes * revenue_rate

# Decision variables
n_cars = len(cars_data)
n_requests = len(requests_data)

# Calculate distances between cars and requests
distances = np.zeros((n_cars, n_requests))
for c, car in enumerate(cars_data):
    for j, request in enumerate(requests_data):
        distance = geodesic(car['start_location'], request['origin_location']).kilometers
        distances[c, j] = distance

# Initialize model
m = Model("Car_sharing")

# set the walking distance
walking_distance = 0.4

m, x = optimization(cars_data, requests_data, walking_distance)

# Solve the model
m.optimize()

# Extract and print the solution
matched_assignments = []  # List to store matched assignments
for j in range(n_requests):
    for c in range(n_cars):
        if x[c, j].X > 0.5:  # If a request-car assignment is chosen
            matched_assignments.append((c, j))  # Store the matched assignment
            print(f"Request {j} assigned to car {c}")

# lists to store latitude and longitude values
lats = []
lons = []

# Iterate over car locations to collect latitude and longitude values
for car in cars_data:
    lats.append(car['start_location'][0])
    lons.append(car['start_location'][1])

# Create map using min-max values
map = smopy.Map((min(lats), min(lons), max(lats), max(lons)), z=12)

# Plot the map with cars and requests
ax = map.show_mpl(figsize=(10, 10))

# Plot the cars on the map
for car in cars_data:
    x, y = map.to_pixels(car['start_location'][0], car['start_location'][1])
    ax.plot(x, y, 'r^', markersize=10, markeredgecolor='k', label='Cars')

# Plot unmatched request origins
for j, request in enumerate(requests_data):
    if j not in [assignment[1] for assignment in matched_assignments]:  # Check if the request is unmatched
        ox, oy = map.to_pixels(request['origin_location'][0], request['origin_location'][1])
        ax.plot(ox, oy, 'o', markersize=5, markeredgecolor='k', color='orange', label='Unmatched Request Origins')

# Plot matched request origins and draw lines to their assigned cars
for car_index, request_index in matched_assignments:
    request = requests_data[request_index]
    car = cars_data[car_index]
    ox, oy = map.to_pixels(request['origin_location'][0], request['origin_location'][1])
    cx, cy = map.to_pixels(car['start_location'][0], car['start_location'][1])

    # Plot matched request origins
    ax.plot(ox, oy, 'go', markersize=5, markeredgecolor='k', label='Matched Request Origins')

    # Draw lines from request to car
    ax.plot([ox, cx], [oy, cy], '-', color='k', alpha=0.9)

# plot matched request destinations
for car_index, request_index in matched_assignments:
    request = requests_data[request_index]
    car = cars_data[car_index]
    dx, dy = map.to_pixels(request['destination_location'][0], request['destination_location'][1])
    # Car start location coordinates (specific to this car)
    cx, cy = map.to_pixels(car['start_location'][0], car['start_location'][1])
    ax.plot(dx, dy, 'bo', markersize=5, markeredgecolor='k', label='Matched Request Destinations')

    # Plot the matched request destination with a blue dot
    ax.plot(dx, dy, 'bo', markersize=5, markeredgecolor='k', label='Matched Request Destinations')

    # Draw a purple line from matched car to matched destination // commented out to visualizing first graph[5]
    # ax.plot([cx, dx], [cy, dy], "-", color="purple", alpha=0.9)

# Add legend and show plot
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())

plt.show()

# PART 5 #

# lists to store walking distances and profits
walking_distances = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2]
profits = []

# Loop over walking distances
for walking_distance in walking_distances:
    # Update walking distance in requests data
    for request in requests_data:
        request['walking_distance'] = walking_distance

    # Recreate the optimization model with the updated walking distance constraint
    m, x = optimization(cars_data, requests_data, walking_distance)

    m.optimize()

    # Check if optimization was successful
    if m.status == GRB.OPTIMAL:

        # Calculate total profit
        total_profit = sum(
            m.getVarByName(f"assign[{c},{j}]").x * requests_data[j]['profit'] for j in range(n_requests) for c in
            range(n_cars))
        profits.append(total_profit)
    else:
        print(f"Optimization failed for walking distance {walking_distance}")

# plot the impact of maximum walking distance to profit
plt.figure(figsize=(14, 7))

# setting up the plot
plt.plot(walking_distances, profits, marker='o')
plt.title('Impact of Maximum Walking Distance on Profit')
plt.xlabel('Maximum Walking Distance (km)')
plt.ylabel('Total Profit (€)')
plt.grid(True)

plt.show()
