import time
import random
import numpy as np
import pandas as pd
import folium
from folium import plugins
from branca.element import Template, MacroElement
from geopy.distance import geodesic
from deap import base, creator, tools

# ----------------------------------------------------------------------
# 1. Data and Threshold
# ----------------------------------------------------------------------

# Replace with your actual data loading
# Example:
df = pd.read_csv('/Users/amanmehra/Desktop/RP-EV/existing_chargers.csv')

existing_stations = df[['Station Name', 'Latitude', 'Longitude']]
existing_locations = [(row['Latitude'], row['Longitude']) for _, row in existing_stations.iterrows()]

def calculate_distance(p1, p2):
    return geodesic(p1, p2).kilometers

def calculate_threshold_distance(locations):
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

# ----------------------------------------------------------------------
# 2. GA Setup
# ----------------------------------------------------------------------
N_NEW_STATIONS = 10
lat_min, lat_max = 24.396308, 49.384358
lon_min, lon_max = -125.0, -66.93457

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

def generate_location():
    while True:
        lat = random.uniform(lat_min, lat_max)
        lon = random.uniform(lon_min, lon_max)
        dists = [calculate_distance((lat, lon), e) for e in existing_locations]
        if min(dists) >= threshold_distance:
            return [lat, lon]

def init_individual():
    coords = []
    for _ in range(N_NEW_STATIONS):
        coords.extend(generate_location())
    return creator.Individual(coords)

toolbox.register("individual", init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def fitness_function(individual):
    penalty = 1e6
    total_dist = 0
    num_new = len(individual)//2

    for i in range(num_new):
        lat = individual[2*i]
        lon = individual[2*i+1]
        dist_existing = [calculate_distance((lat, lon), e) for e in existing_locations]
        min_dist = min(dist_existing)

        if min_dist < threshold_distance:
            return (penalty,)

        total_dist += min_dist

    return (total_dist / num_new,)

toolbox.register("evaluate", fitness_function)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxTwoPoint)

def constrained_mutate(individual):
    mutpb = 0.1
    for i in range(0, len(individual), 2):
        if random.random() < mutpb:
            individual[i]   += random.gauss(0, 0.01)
            individual[i+1] += random.gauss(0, 0.01)

            # clamp
            individual[i]   = max(min(individual[i],   lat_max), lat_min)
            individual[i+1] = max(min(individual[i+1], lon_max), lon_min)

            # re-check constraint
            dists = [calculate_distance((individual[i], individual[i+1]), e)
                     for e in existing_locations]
            if min(dists) < threshold_distance:
                # revert to new random
                valid_loc = generate_location()
                individual[i], individual[i+1] = valid_loc
    return (individual,)

toolbox.register("mutate", constrained_mutate)

def run_ga_optimization():
    pop_size = 50
    n_gen = 50
    cx_prob = 0.5
    mut_prob = 0.2

    # init population
    population = toolbox.population(n=pop_size)
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    start = time.time()
    for gen in range(n_gen):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # mutate
        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # evaluate
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind)

        population[:] = offspring
    end = time.time()
    duration = end - start

    # best
    best_ind = min(population, key=lambda ind: ind.fitness.values[0])
    best_fitness = best_ind.fitness.values[0]

    best_coords = []
    for i in range(N_NEW_STATIONS):
        lat = best_ind[2*i]
        lon = best_ind[2*i+1]
        best_coords.append((lat, lon))

    return best_coords, best_fitness, duration

def main():
    results = []
    for run_idx in range(1, 11):
        print(f"\n--- GA-Only Simulation #{run_idx} ---")
        coords, fitness_val, comp_time = run_ga_optimization()
        print(f"Best Fitness: {fitness_val:.4f} | Time: {comp_time:.2f}s")
        results.append({
            'Run': run_idx,
            'Best Fitness': fitness_val,
            'Time (s)': comp_time,
            'Coordinates': coords
        })

    results_df = pd.DataFrame(results)
    print("\nAll Results:\n", results_df)

    # Visualize final run with Folium
    final_run = results[-1]
    ga_coords = final_run['Coordinates']

    def find_nearest(existing_stations, new_station):
        dists = [calculate_distance(new_station, st) for st in existing_stations]
        min_d = min(dists)
        return existing_stations[dists.index(min_d)], min_d

    all_coords = existing_locations + ga_coords
    all_lats = [c[0] for c in all_coords]
    all_lons = [c[1] for c in all_coords]
    map_center = [np.mean(all_lats), np.mean(all_lons)]

    m = folium.Map(location=map_center, zoom_start=5)

    # Existing
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

    # GA (Red)
    for lat, lon in ga_coords:
        _, dist = find_nearest(existing_locations, (lat, lon))
        folium.Marker(
            location=[lat, lon],
            popup=f'GA Station<br>Nearest existing: {dist:.2f} km',
            icon=folium.Icon(color='red', icon='star')
        ).add_to(m)
        folium.Circle(
            location=[lat, lon],
            radius=threshold_distance*1000,
            color='red',
            fill=False,
            weight=1,
            opacity=0.5,
            tooltip=f'Threshold: {threshold_distance:.2f} km'
        ).add_to(m)

    # Legend
    legend_html = '''
    {% macro html(this, kwargs) %}
    <div style="
        position: fixed;
        bottom: 50px;
        left: 50px;
        width: 200px;
        height: 130px;
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
        <i class="fa fa-star" style="color:red"></i>
          &nbsp;GA Locations<br>
        <i class="fa fa-circle-thin" style="color:grey"></i>
          &nbsp;Threshold Distance
    </div>
    {% endmacro %}
    '''
    legend = MacroElement()
    legend._template = Template(legend_html)
    m.get_root().add_child(legend)

    m.save("ga_only_map.html")
    print("Folium map saved to 'ga_only_map.html'.")

if __name__ == "__main__":
    main()
