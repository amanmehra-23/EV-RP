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

# ----------------------------------------------------------------------
# 1. Data and Threshold
# ----------------------------------------------------------------------

# Replace this with your actual station data
# Example:
df = pd.read_csv('existing_chargers.csv')
# Ensure columns: ['Station Name','Latitude','Longitude']

# For demo, suppose df is already defined in your environment:
# data = {
#     'Station Name': ['StationA', 'StationB', 'StationC'],
#     'Latitude': [34.0522, 37.7749, 40.7128],
#     'Longitude': [-118.2437, -122.4194, -74.0060]
# }
# df = pd.DataFrame(data)

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
# Bounds (U.S. example):
lat_min, lat_max = 24.396308, 49.384358
lon_min, lon_max = -125.0, -66.93457

# ----------------------------------------------------------------------
# 2. PSO
# ----------------------------------------------------------------------
lb = [lat_min, lon_min] * N_NEW_STATIONS
ub = [lat_max, lon_max] * N_NEW_STATIONS

def fitness_function_pso(new_station_locations):
    """
    Minimize average distance to the nearest existing station,
    subject to a constraint: each new station must be >= threshold_distance
    from all existing stations.
    """
    penalty = 1e6
    total_distance = 0
    num_new_stations = len(new_station_locations)//2

    for i in range(num_new_stations):
        lat = new_station_locations[2*i]
        lon = new_station_locations[2*i + 1]
        new_coord = (lat, lon)

        distances_to_existing = [calculate_distance(new_coord, e) for e in existing_locations]
        min_dist = min(distances_to_existing)

        if min_dist < threshold_distance:
            return penalty  # constraint violation

        total_distance += min_dist

    return total_distance / num_new_stations

# ----------------------------------------------------------------------
# 3. GA (DEAP)
# ----------------------------------------------------------------------
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

# Generate a valid (lat, lon)
def generate_location():
    while True:
        lat = random.uniform(lat_min, lat_max)
        lon = random.uniform(lon_min, lon_max)
        dist_existing = [calculate_distance((lat, lon), e) for e in existing_locations]
        if min(dist_existing) >= threshold_distance:
            return [lat, lon]

# GA individual initialization
def init_individual():
    coords = []
    for _ in range(N_NEW_STATIONS):
        coords.extend(generate_location())
    return creator.Individual(coords)

toolbox.register("individual", init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# GA fitness function
def fitness_function_ga(individual):
    penalty = 1e6
    total_distance = 0
    num_new = len(individual)//2

    for i in range(num_new):
        lat = individual[2*i]
        lon = individual[2*i + 1]
        dists_to_existing = [calculate_distance((lat, lon), e) for e in existing_locations]
        min_dist = min(dists_to_existing)

        if min_dist < threshold_distance:
            return (penalty,)

        total_distance += min_dist

    return (total_distance / num_new,)

toolbox.register("evaluate", fitness_function_ga)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxTwoPoint)

def constrained_mutate(individual):
    mutpb = 0.1
    for i in range(0, len(individual), 2):
        if random.random() < mutpb:
            # mutate lat
            individual[i] += random.gauss(0, 0.01)
            # mutate lon
            individual[i+1] += random.gauss(0, 0.01)

            # clamp to bounds
            individual[i]   = max(min(individual[i],   lat_max), lat_min)
            individual[i+1] = max(min(individual[i+1], lon_max), lon_min)

            # re-check constraint
            if min([calculate_distance((individual[i], individual[i+1]), e)
                    for e in existing_locations]) < threshold_distance:
                # revert to a valid random coordinate
                new_coord = generate_location()
                individual[i], individual[i+1] = new_coord

    return (individual,)

toolbox.register("mutate", constrained_mutate)

def initialize_population_from_pso(pso_coords):
    """Build a GA Individual from PSO solution."""
    ind_data = []
    for (lat, lon) in pso_coords:
        ind_data.append(lat)
        ind_data.append(lon)
    return creator.Individual(ind_data)

# ----------------------------------------------------------------------
# 4. Combined Routine: PSO -> GA
# ----------------------------------------------------------------------
def run_pso_then_ga():
    """
    1) Runs PSO to find an initial solution.
    2) Uses that solution as part of the GA initial population.
    3) Returns the GA final best solution and times.
    """
    # 1) PSO
    pso_start = time.time()
    pso_best_locations, pso_best_value = pso(
        fitness_function_pso,
        lb=lb,
        ub=ub,
        swarmsize=50,
        maxiter=50
    )
    pso_end = time.time()
    pso_time = pso_end - pso_start

    # Convert PSO solution to (lat, lon)
    pso_coords = []
    for i in range(N_NEW_STATIONS):
        lat = pso_best_locations[2*i]
        lon = pso_best_locations[2*i+1]
        pso_coords.append((lat, lon))

    # 2) GA
    ga_start = time.time()
    pop_size = 50
    n_gen = 50
    cx_prob = 0.5
    mut_prob = 0.2

    population = []
    # fill population with random individuals
    population.extend(toolbox.population(n=pop_size - 1))
    # add 1 from PSO
    population.append(initialize_population_from_pso(pso_coords))

    # Evaluate
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    for g in range(n_gen):
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

        # Evaluate invalid
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind)

        population[:] = offspring

    ga_end = time.time()
    ga_time = ga_end - ga_start

    # best from GA
    best_ind = min(population, key=lambda ind: ind.fitness.values[0])
    best_fitness = best_ind.fitness.values[0]

    refined_coords = []
    for i in range(N_NEW_STATIONS):
        lat = best_ind[2*i]
        lon = best_ind[2*i+1]
        refined_coords.append((lat, lon))

    total_time = pso_time + ga_time

    return pso_coords, pso_best_value, refined_coords, best_fitness, pso_time, ga_time, total_time

def main():
    results = []
    for run_idx in range(1, 11):
        print(f"\n--- Combined PSO+GA Simulation #{run_idx} ---")
        (pso_coords, pso_fitness,
         ga_coords, ga_fitness,
         pso_time, ga_time, total_time) = run_pso_then_ga()

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

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    print("\nAll Results:\n", results_df)

    # Example: visualize the final run's stations on Folium map
    # ---------------------------------------------------------------------------------
    final_run = results[-1]
    optimal_station_coords = final_run['PSO Coordinates']      # from PSO
    refined_station_coords = final_run['GA Refined Coordinates'] # from GA

    # BUILD FOLIUM MAP
    def find_nearest(existing_stations, new_station):
        dists = [geodesic(new_station, st).kilometers for st in existing_stations]
        min_d = min(dists)
        return existing_stations[dists.index(min_d)], min_d

    # Calculate the map center
    # We'll combine existing, PSO, GA coords
    all_coords = existing_locations + optimal_station_coords + refined_station_coords
    all_lats = [c[0] for c in all_coords]
    all_lons = [c[1] for c in all_coords]
    map_center = [np.mean(all_lats), np.mean(all_lons)]

    m = folium.Map(location=map_center, zoom_start=5)

    # Plot existing stations (blue)
    for lat, lon in existing_locations:
        folium.CircleMarker(
            location=[lat, lon],
            radius=2,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6,
            popup='Existing Station'
        ).add_to(m)

    # Plot PSO coords (orange)
    for lat, lon in optimal_station_coords:
        _, dist = find_nearest(existing_locations, (lat, lon))
        folium.Marker(
            location=[lat, lon],
            popup=f'PSO Station<br>Nearest existing: {dist:.2f} km',
            icon=folium.Icon(color='orange', icon='info-sign')
        ).add_to(m)
        folium.Circle(
            location=[lat, lon],
            radius=threshold_distance * 1000,
            color='orange',
            fill=False,
            weight=1,
            opacity=0.5,
            tooltip=f'Threshold: {threshold_distance:.2f} km'
        ).add_to(m)

    # Plot GA refined coords (red)
    for lat, lon in refined_station_coords:
        _, dist = find_nearest(existing_locations, (lat, lon))
        folium.Marker(
            location=[lat, lon],
            popup=f'GA Refined Station<br>Nearest existing: {dist:.2f} km',
            icon=folium.Icon(color='red', icon='star')
        ).add_to(m)
        folium.Circle(
            location=[lat, lon],
            radius=threshold_distance * 1000,
            color='red',
            fill=False,
            weight=1,
            opacity=0.5,
            tooltip=f'Threshold: {threshold_distance:.2f} km'
        ).add_to(m)

    # Add a legend
    legend_html = '''
    {% macro html(this, kwargs) %}
    <div style="
        position: fixed;
        bottom: 50px;
        left: 50px;
        width: 230px;
        height: 160px;
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
          &nbsp;Threshold Distance
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
