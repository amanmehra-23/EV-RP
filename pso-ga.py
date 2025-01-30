import time
import random
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from pyswarm import pso
from deap import base, creator, tools

import folium
from folium import plugins
from branca.element import Template, MacroElement

import geopandas as gpd
from shapely.geometry import Point

# ----------------------------------------------------------------------
# 1. Data and Threshold
# ----------------------------------------------------------------------

# Replace this with your actual station data
# Example:
df = pd.read_csv('existing_stations.csv')
# Ensure columns: ['Station Name','Latitude','Longitude']

existing_stations = df[['Station Name', 'Latitude', 'Longitude']]
existing_locations = [(row['Latitude'], row['Longitude']) for _, row in existing_stations.iterrows()]

def calculate_distance(point1, point2):
    """Geodesic distance (km)."""
    return geodesic(point1, point2).kilometers

def calculate_threshold_distance(locations):
    """
    Calculate a threshold distance based on each existing station's average
    distance to its 3 closest neighbors, and then averaging those values.
    """
    avg_distances = []
    for i, station in enumerate(locations):
        distances = []
        for j, other_station in enumerate(locations):
            if i != j:
                distances.append(calculate_distance(station, other_station))
        distances.sort()
        closest_3 = distances[:3] if len(distances) >= 3 else distances
        avg_distances.append(sum(closest_3)/len(closest_3))
    return sum(avg_distances)/len(avg_distances)

threshold_distance = calculate_threshold_distance(existing_locations)
print(f"Threshold distance: {threshold_distance:.2f} km")

# Number of new stations
N_NEW_STATIONS = 10
# Tight bounding box for California
lat_min, lat_max = 32.5343, 42.0095
lon_min, lon_max = -124.4096, -114.1312

# ----------------------------------------------------------------------
# 2. Load California Boundary
# ----------------------------------------------------------------------

# URL to the GeoJSON file containing U.S. states
usa_states_url = 'https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json'

# Load the GeoJSON data into a GeoDataFrame
usa_states = gpd.read_file(usa_states_url)

# Inspect the columns to ensure the state name column is correct
print("USA States GeoDataFrame Columns:", usa_states.columns)

# Filter for California
california = usa_states[usa_states['name'] == 'California']

# Check if California was found
if california.empty:
    raise ValueError("California not found in the dataset. Please check the state name or dataset.")

# Combine all geometries into a single polygon (useful if California has multiple polygons)
california_polygon = california.geometry.unary_union

def is_within_california(lat, lon, california_polygon):
    """
    Checks if a given latitude and longitude are within California's boundaries.

    Args:
        lat (float): Latitude of the point.
        lon (float): Longitude of the point.
        california_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of California.

    Returns:
        bool: True if the point is within California, False otherwise.
    """
    point = Point(lon, lat)  # Note: Point takes (longitude, latitude)
    return california_polygon.contains(point)

# ----------------------------------------------------------------------
# 3. PSO Setup with California Constraints
# ----------------------------------------------------------------------
lb = [lat_min, lon_min] * N_NEW_STATIONS
ub = [lat_max, lon_max] * N_NEW_STATIONS

def fitness_function_pso(new_station_locations, california_polygon):
    """
    PSO Fitness Function that minimizes the average distance to the nearest existing station,
    while ensuring all new stations are within California and respect the threshold distance.

    Args:
        new_station_locations (list): Flat list of [lat1, lon1, lat2, lon2, ...].
        california_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of California.

    Returns:
        float: The fitness value (lower is better).
    """
    penalty_outside = 1e8  # Higher penalty for being outside California
    penalty_threshold = 1e6  # Penalty for violating threshold distance
    total_distance = 0
    num_new_stations = len(new_station_locations) // 2

    for i in range(num_new_stations):
        lat = new_station_locations[2 * i]
        lon = new_station_locations[2 * i + 1]
        new_coord = (lat, lon)

        # Check if within California
        if not is_within_california(lat, lon, california_polygon):
            return penalty_outside  # Constraint violation

        # Calculate distance to existing stations
        distances_to_existing = [calculate_distance(new_coord, e) for e in existing_locations]
        min_dist = min(distances_to_existing)

        if min_dist < threshold_distance:
            return penalty_threshold  # Constraint violation

        total_distance += min_dist

    return total_distance / num_new_stations

def run_pso_optimization(california_polygon):
    """
    Runs PSO optimization to find initial station locations within California.

    Args:
        california_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of California.

    Returns:
        tuple: (best_coords, best_fitness, duration)
    """
    start = time.time()
    # Define a wrapper for the fitness function to include california_polygon
    def fitness_wrapper(new_station_locations):
        return fitness_function_pso(new_station_locations, california_polygon)

    best_locations, best_value = pso(
        fitness_wrapper,
        lb=lb,
        ub=ub,
        swarmsize=50,
        maxiter=50
    )
    end = time.time()
    duration = end - start

    # Convert best_locations to list of tuples
    best_coords = [(best_locations[2 * i], best_locations[2 * i + 1]) for i in range(N_NEW_STATIONS)]

    return best_coords, best_value, duration

# ----------------------------------------------------------------------
# 4. GA (DEAP) Setup with California Constraints
# ----------------------------------------------------------------------
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

def generate_location(california_polygon):
    """
    Generates a valid [lat, lon] pair within California and respecting the threshold distance.

    Args:
        california_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of California.

    Returns:
        list: A list containing [lat, lon].
    """
    attempts = 0
    max_attempts = 1000  # Prevent infinite loops

    while attempts < max_attempts:
        lat = random.uniform(lat_min, lat_max)
        lon = random.uniform(lon_min, lon_max)
        if is_within_california(lat, lon, california_polygon):
            distances = [calculate_distance((lat, lon), e) for e in existing_locations]
            if min(distances) >= threshold_distance:
                return [lat, lon]
        attempts += 1

    raise ValueError("Exceeded maximum attempts to generate a valid location within California.")

# GA Individual Initialization
def init_individual(california_polygon):
    """
    Initializes an individual with N_NEW_STATIONS within California.

    Args:
        california_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of California.

    Returns:
        Individual: A DEAP Individual.
    """
    coords = []
    for _ in range(N_NEW_STATIONS):
        coords.extend(generate_location(california_polygon))
    return creator.Individual(coords)

# Register GA components with california_polygon
toolbox.register("individual", init_individual, california_polygon=california_polygon)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def fitness_function_ga(individual, california_polygon):
    """
    GA Fitness Function that minimizes the average distance to the nearest existing station,
    while ensuring all new stations are within California and respect the threshold distance.

    Args:
        individual (list): Flat list of [lat1, lon1, lat2, lon2, ...].
        california_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of California.

    Returns:
        tuple: The fitness value.
    """
    penalty_outside = 1e8  # Higher penalty for being outside California
    penalty_threshold = 1e6  # Penalty for violating threshold distance
    total_distance = 0
    num_new_stations = len(individual) // 2

    for i in range(num_new_stations):
        lat = individual[2 * i]
        lon = individual[2 * i + 1]
        new_coord = (lat, lon)

        # Check if within California
        if not is_within_california(lat, lon, california_polygon):
            return (penalty_outside,)

        # Calculate distance to existing stations
        distances_to_existing = [calculate_distance(new_coord, e) for e in existing_locations]
        min_dist = min(distances_to_existing)

        if min_dist < threshold_distance:
            return (penalty_threshold,)

        total_distance += min_dist

    return (total_distance / num_new_stations,)

toolbox.register("evaluate", fitness_function_ga, california_polygon=california_polygon)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxTwoPoint)

def constrained_mutate(individual, california_polygon):
    """
    Mutates an individual while ensuring constraints:
    - New locations are within California.
    - New locations respect the threshold distance from existing stations.

    Args:
        individual (list): Flat list of [lat1, lon1, lat2, lon2, ...].
        california_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of California.

    Returns:
        tuple: The mutated individual.
    """
    mutation_probability = 0.1  # Probability to mutate each gene pair

    for i in range(0, len(individual), 2):
        if random.random() < mutation_probability:
            # Apply Gaussian mutation
            individual[i] += random.gauss(0, 0.01)      # Mutate latitude
            individual[i + 1] += random.gauss(0, 0.01)  # Mutate longitude

            # Clamp to bounding box
            individual[i] = max(min(individual[i], lat_max), lat_min)
            individual[i + 1] = max(min(individual[i + 1], lon_max), lon_min)

            # Check if within California
            if not is_within_california(individual[i], individual[i + 1], california_polygon):
                # Reinitialize if out of bounds
                new_coord = generate_location(california_polygon)
                individual[i], individual[i + 1] = new_coord
            else:
                # Check distance constraint
                distances = [calculate_distance((individual[i], individual[i + 1]), e) for e in existing_locations]
                if min(distances) < threshold_distance:
                    # Reinitialize if distance constraint violated
                    new_coord = generate_location(california_polygon)
                    individual[i], individual[i + 1] = new_coord

    return (individual,)

toolbox.register("mutate", constrained_mutate, california_polygon=california_polygon)

def initialize_population_from_pso(pso_coords):
    """
    Builds a GA Individual from PSO solution.

    Args:
        pso_coords (list of tuples): List of (lat, lon) from PSO.

    Returns:
        Individual: A DEAP Individual.
    """
    ind_data = []
    for (lat, lon) in pso_coords:
        ind_data.append(lat)
        ind_data.append(lon)
    return creator.Individual(ind_data)

# ----------------------------------------------------------------------
# 5. Combined Routine: PSO -> GA
# ----------------------------------------------------------------------
def run_pso_then_ga(california_polygon):
    """
    1) Runs PSO to find an initial solution within California.
    2) Uses that solution as part of the GA initial population.
    3) Returns the GA final best solution and times.

    Args:
        california_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of California.

    Returns:
        tuple: (pso_coords, pso_fitness, ga_coords, ga_fitness, pso_time, ga_time, total_time)
    """
    # 1) PSO Optimization
    pso_start = time.time()
    pso_best_locations, pso_best_value, pso_time = run_pso_optimization(california_polygon)
    pso_end = time.time()

    # Convert PSO solution to list of tuples
    pso_coords = pso_best_locations

    # 2) GA Optimization
    ga_start = time.time()
    pop_size = 50
    n_gen = 50
    cx_prob = 0.5
    mut_prob = 0.2

    # Initialize GA population
    population = toolbox.population(n=pop_size - 1)
    # Add PSO solution as one individual
    pso_individual = initialize_population_from_pso(pso_coords)
    population.append(pso_individual)

    # Evaluate all individuals
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    # Evolutionary loop
    for gen in range(n_gen):
        # Selection
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate invalid individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind)

        # Replace population
        population[:] = offspring

    ga_end = time.time()
    ga_time = ga_end - ga_start

    # Best GA individual
    best_ind = min(population, key=lambda ind: ind.fitness.values[0])
    ga_fitness = best_ind.fitness.values[0]
    ga_coords = [(best_ind[2 * i], best_ind[2 * i + 1]) for i in range(N_NEW_STATIONS)]

    total_time = pso_time + ga_time

    return pso_coords, pso_best_value, ga_coords, ga_fitness, pso_time, ga_time, total_time

# ----------------------------------------------------------------------
# 6. Main Function
# ----------------------------------------------------------------------
def main():
    """
    Main function to run combined PSO and GA optimizations and visualize results.
    """
    results = []
    for run_idx in range(1, 11):
        print(f"\n--- Combined PSO+GA Simulation #{run_idx} ---")
        try:
            (pso_coords, pso_fitness,
             ga_coords, ga_fitness,
             pso_time, ga_time, total_time) = run_pso_then_ga(california_polygon)

            print(f"PSO Fit: {pso_fitness:.4f} | GA Fit: {ga_fitness:.4f}")
            print(f"Times => PSO: {pso_time:.2f}s, GA: {ga_time:.2f}s, Total: {total_time:.2f}s")

            results.append({
                'Run': run_idx,
                'PSO Fitness': pso_fitness,
                'GA Fitness': ga_fitness,
                'PSO Time': pso_time,
                'GA Time': ga_time,
                'Total Time': total_time,
                'PSO Coordinates': pso_coords,
                'GA Refined Coordinates': ga_coords
            })
        except ValueError as ve:
            print(f"Run #{run_idx} failed: {ve}")
            results.append({
                'Run': run_idx,
                'PSO Fitness': None,
                'GA Fitness': None,
                'PSO Time': None,
                'GA Time': None,
                'Total Time': None,
                'PSO Coordinates': None,
                'GA Refined Coordinates': None
            })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    print("\nAll Results:\n", results_df)
    results_df.to_csv("pso-ga.csv", index=False)

    # Visualization with Folium
    final_run = results[-1]
    pso_coords = final_run['PSO Coordinates']
    ga_coords = final_run['GA Refined Coordinates']

    # Function to find the nearest existing station
    def find_nearest(existing_stations, new_station):
        dists = [geodesic(new_station, st).kilometers for st in existing_locations]
        min_dist = min(dists)
        nearest_station = existing_stations[dists.index(min_dist)]
        return nearest_station, min_dist

    # Calculate the map center
    all_coords = existing_locations + pso_coords + ga_coords
    all_lats = [c[0] for c in all_coords]
    all_lons = [c[1] for c in all_coords]
    map_center = [np.mean(all_lats), np.mean(all_lons)]

    # Initialize Folium map
    m = folium.Map(location=map_center, zoom_start=6)

    # Plot existing stations (Blue)
    for lat, lon in existing_locations:
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6,
            popup='Existing Station'
        ).add_to(m)

    # Plot PSO coordinates (Orange)
    for lat, lon in pso_coords:
        nearest, dist = find_nearest(existing_locations, (lat, lon))
        folium.Marker(
            location=[lat, lon],
            popup=f'PSO Station<br>Nearest existing: {dist:.2f} km',
            icon=folium.Icon(color='orange', icon='info-sign')
        ).add_to(m)
        folium.Circle(
            location=[lat, lon],
            radius=threshold_distance * 1000,  # Convert km to meters
            color='orange',
            fill=False,
            weight=1,
            opacity=0.5,
            tooltip=f'Threshold: {threshold_distance:.2f} km'
        ).add_to(m)

    # Plot GA refined coordinates (Red)
    for lat, lon in ga_coords:
        nearest, dist = find_nearest(existing_locations, (lat, lon))
        folium.Marker(
            location=[lat, lon],
            popup=f'GA Refined Station<br>Nearest existing: {dist:.2f} km',
            icon=folium.Icon(color='red', icon='star')
        ).add_to(m)
        folium.Circle(
            location=[lat, lon],
            radius=threshold_distance * 1000,  # Convert km to meters
            color='red',
            fill=False,
            weight=1,
            opacity=0.5,
            tooltip=f'Threshold: {threshold_distance:.2f} km'
        ).add_to(m)

    # Highlight California Boundary
    folium.GeoJson(
        california.__geo_interface__,
        name="California",
        style_function=lambda x: {
            'fillColor': 'none',
            'color': 'green',
            'weight': 2
        }
    ).add_to(m)

    # Add a legend
    legend_html = '''
    {% macro html(this, kwargs) %}
    <div style="
        position: fixed;
        bottom: 50px;
        left: 50px;
        width: 230px;
        height: 170px;
        z-index:9999;
        font-size:14px;
        background-color: white;
        padding: 10px;
        border: 2px solid grey;
        border-radius: 5px;
    ">
        <b>Legend</b><br>
        <i class="fa fa-circle" style="color:blue"></i>
          &nbsp;Existing Stations<br>
        <i class="fa fa-info-circle" style="color:orange"></i>
          &nbsp;PSO Locations<br>
        <i class="fa fa-star" style="color:red"></i>
          &nbsp;GA Refined Locations<br>
        <i class="fa fa-circle-thin" style="color:grey"></i>
          &nbsp;Threshold Distance<br>
        <i class="fa fa-square" style="color:green"></i>
          &nbsp;California Boundary
    </div>
    {% endmacro %}
    '''
    legend = MacroElement()
    legend._template = Template(legend_html)
    m.get_root().add_child(legend)

    # Save the map
    m.save("combined_pso_ga_map.html")
    print("Folium map saved to 'combined_pso_ga_map.html'.")

if __name__ == "__main__":
    main()
