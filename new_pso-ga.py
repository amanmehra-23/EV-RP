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

import warnings
import sys

warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    df = pd.read_csv('existing_stations.csv')
except FileNotFoundError:
    sys.exit(1)

required_columns = ['Station Name', 'Latitude', 'Longitude']
if not all(column in df.columns for column in required_columns):
    sys.exit(1)

existing_stations = df[['Station Name', 'Latitude', 'Longitude']]
existing_locations = [(row['Latitude'], row['Longitude']) for _, row in existing_stations.iterrows()]

def calculate_distance(point1, point2):
    return geodesic(point1, point2).kilometers

def calculate_threshold_distance(locations):
    avg_distances = []
    for i, station in enumerate(locations):
        distances = []
        for j, other_station in enumerate(locations):
            if i != j:
                distances.append(calculate_distance(station, other_station))
        distances.sort()
        closest_3 = distances[:3] if len(distances) >= 3 else distances
        avg_distances.append(sum(closest_3) / len(closest_3))
    threshold = sum(avg_distances) / len(avg_distances)
    return threshold

threshold_distance = calculate_threshold_distance(existing_locations)

N_NEW_STATIONS = 10

lat_min, lat_max = 32.5343, 42.0095
lon_min, lon_max = -124.4096, -114.1312

usa_states_url = 'https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json'

try:
    usa_states = gpd.read_file(usa_states_url)
except:
    sys.exit(1)

california = usa_states[usa_states['name'] == 'California']

if california.empty:
    sys.exit(1)

if hasattr(california.geometry, 'union_all'):
    california_polygon = california.geometry.union_all()
else:
    california_polygon = california.geometry.unary_union

land_geojson_path = 'ne_10m_land.geojson'

try:
    land = gpd.read_file(land_geojson_path)
except FileNotFoundError:
    sys.exit(1)
except:
    sys.exit(1)

if land.crs != usa_states.crs:
    land = land.to_crs(usa_states.crs)

if hasattr(land.geometry, 'union_all'):
    land_polygon = land.geometry.union_all()
else:
    land_polygon = land.geometry.unary_union

def is_within_california_and_land(lat, lon, california_polygon, land_polygon):
    point = Point(lon, lat)
    return california_polygon.contains(point) and land_polygon.contains(point)

lb = [lat_min, lon_min] * N_NEW_STATIONS
ub = [lat_max, lon_max] * N_NEW_STATIONS

def fitness_function_pso(new_station_locations, california_polygon, land_polygon):
    penalty_outside = 1e8
    penalty_water = 1e7
    penalty_threshold = 1e6
    total_distance = 0
    num_new_stations = len(new_station_locations) // 2

    for i in range(num_new_stations):
        lat = new_station_locations[2 * i]
        lon = new_station_locations[2 * i + 1]
        new_coord = (lat, lon)

        if not is_within_california_and_land(lat, lon, california_polygon, land_polygon):
            point = Point(lon, lat)
            if not california_polygon.contains(point):
                return penalty_outside
            elif not land_polygon.contains(point):
                return penalty_water

        distances_to_existing = [calculate_distance(new_coord, e) for e in existing_locations]
        min_dist = min(distances_to_existing)

        if min_dist < threshold_distance:
            return penalty_threshold

        total_distance += min_dist

    average_distance = total_distance / num_new_stations
    return average_distance

def run_pso_optimization(california_polygon, land_polygon):
    start = time.time()

    def fitness_wrapper(new_station_locations):
        return fitness_function_pso(new_station_locations, california_polygon, land_polygon)

    try:
        best_locations, best_value = pso(
            fitness_wrapper,
            lb=lb,
            ub=ub,
            swarmsize=10,
            maxiter=50,
            debug=False  # Disabled debug prints
        )
        end = time.time()
        duration = end - start
    except:
        sys.exit(1)

    best_coords = [(best_locations[2 * i], best_locations[2 * i + 1]) for i in range(N_NEW_STATIONS)]
    return best_coords, best_value, duration

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

def generate_location(california_polygon, land_polygon):
    attempts = 0
    max_attempts = 50

    while attempts < max_attempts:
        lat = random.uniform(lat_min, lat_max)
        lon = random.uniform(lon_min, lon_max)
        if is_within_california_and_land(lat, lon, california_polygon, land_polygon):
            distances = [calculate_distance((lat, lon), e) for e in existing_locations]
            if min(distances) >= threshold_distance:
                return [lat, lon]
        attempts += 1
    raise ValueError("Exceeded maximum attempts to generate a valid location within California on land.")

def init_individual(california_polygon, land_polygon):
    coords = []
    for _ in range(N_NEW_STATIONS):
        coords.extend(generate_location(california_polygon, land_polygon))
    return creator.Individual(coords)

toolbox.register("individual", init_individual, california_polygon=california_polygon, land_polygon=land_polygon)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def fitness_function_ga(individual, california_polygon, land_polygon):
    penalty_outside = 1e8
    penalty_water = 1e7
    penalty_threshold = 1e6
    total_distance = 0
    num_new_stations = len(individual) // 2

    for i in range(num_new_stations):
        lat = individual[2 * i]
        lon = individual[2 * i + 1]
        new_coord = (lat, lon)

        if not is_within_california_and_land(lat, lon, california_polygon, land_polygon):
            point = Point(lon, lat)
            if not california_polygon.contains(point):
                return (penalty_outside,)
            elif not land_polygon.contains(point):
                return (penalty_water,)

        distances_to_existing = [calculate_distance(new_coord, e) for e in existing_locations]
        min_dist = min(distances_to_existing)

        if min_dist < threshold_distance:
            return (penalty_threshold,)

        total_distance += min_dist

    average_distance = total_distance / num_new_stations
    return (average_distance,)

toolbox.register("evaluate", fitness_function_ga, california_polygon=california_polygon, land_polygon=land_polygon)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxTwoPoint)

def constrained_mutate(individual, california_polygon, land_polygon):
    mutation_probability = 0.1

    for i in range(0, len(individual), 2):
        if random.random() < mutation_probability:
            original_lat = individual[i]
            original_lon = individual[i + 1]
            individual[i] += random.gauss(0, 0.01)
            individual[i + 1] += random.gauss(0, 0.01)
            individual[i] = max(min(individual[i], lat_max), lat_min)
            individual[i + 1] = max(min(individual[i + 1], lon_max), lon_min)

            if not is_within_california_and_land(individual[i], individual[i + 1], california_polygon, land_polygon):
                try:
                    new_coord = generate_location(california_polygon, land_polygon)
                    individual[i], individual[i + 1] = new_coord
                except:
                    pass
            else:
                distances = [calculate_distance((individual[i], individual[i + 1]), e) for e in existing_locations]
                if min(distances) < threshold_distance:
                    try:
                        new_coord = generate_location(california_polygon, land_polygon)
                        individual[i], individual[i + 1] = new_coord
                    except:
                        pass
    return (individual,)

toolbox.register("mutate", constrained_mutate, california_polygon=california_polygon, land_polygon=land_polygon)

def initialize_population_from_pso(pso_coords):
    ind_data = []
    for (lat, lon) in pso_coords:
        ind_data.append(lat)
        ind_data.append(lon)
    return creator.Individual(ind_data)

def run_pso_then_ga(california_polygon, land_polygon):
    pso_coords, pso_fitness, pso_time = run_pso_optimization(california_polygon, land_polygon)

    ga_start = time.time()
    pop_size = 30
    n_gen = 30
    cx_prob = 0.5
    mut_prob = 0.2

    try:
        population = toolbox.population(n=pop_size - 1)
    except:
        sys.exit(1)

    try:
        pso_individual = initialize_population_from_pso(pso_coords)
        population.append(pso_individual)
    except:
        pass

    for ind in population:
        try:
            ind.fitness.values = toolbox.evaluate(ind)
        except:
            ind.fitness.values = (1e12,)

    for _ in range(n_gen):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            try:
                ind.fitness.values = toolbox.evaluate(ind)
            except:
                ind.fitness.values = (1e12,)

        population[:] = offspring

    ga_end = time.time()
    ga_time = ga_end - ga_start

    try:
        best_ind = min(population, key=lambda ind: ind.fitness.values[0])
        ga_fitness = best_ind.fitness.values[0]
        ga_coords = [(best_ind[2 * i], best_ind[2 * i + 1]) for i in range(N_NEW_STATIONS)]
    except ValueError:
        ga_fitness = None
        ga_coords = None

    total_time = pso_time + ga_time

    return pso_coords, pso_fitness, ga_coords, ga_fitness, pso_time, ga_time, total_time

def main():
    results = []
    for run_idx in range(1, 2):
        try:
            (
                pso_coords,
                pso_fitness,
                ga_coords,
                ga_fitness,
                pso_time,
                ga_time,
                total_time
            ) = run_pso_then_ga(california_polygon, land_polygon)

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

    results_df = pd.DataFrame(results)
    results_df.to_csv("pso-ga.csv", index=False)

    final_run = results[-1]
    if final_run['PSO Coordinates'] is not None and final_run['GA Refined Coordinates'] is not None:
        pso_coords = final_run['PSO Coordinates']
        ga_coords = final_run['GA Refined Coordinates']

        def find_nearest(existing_stations, new_station):
            dists = [geodesic(new_station, st).kilometers for st in existing_locations]
            min_d = min(dists)
            nearest_station = existing_stations[dists.index(min_d)]
            return nearest_station, min_d

        all_coords = existing_locations + pso_coords + ga_coords
        all_lats = [c[0] for c in all_coords]
        all_lons = [c[1] for c in all_coords]
        map_center = [np.mean(all_lats), np.mean(all_lons)]

        m = folium.Map(location=map_center, zoom_start=6)

        for lat, lon in existing_locations:
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.6
            ).add_to(m)

        for idx, (lat, lon) in enumerate(pso_coords, 1):
            nearest, dist = find_nearest(existing_locations, (lat, lon))
            folium.Marker(
                location=[lat, lon],
                icon=folium.Icon(color='orange', icon='info-sign')
            ).add_to(m)
            folium.Circle(
                location=[lat, lon],
                radius=threshold_distance * 1000,
                color='orange',
                fill=False,
                weight=1,
                opacity=0.5
            ).add_to(m)

        for idx, (lat, lon) in enumerate(ga_coords, 1):
            nearest, dist = find_nearest(existing_locations, (lat, lon))
            folium.Marker(
                location=[lat, lon],
                icon=folium.Icon(color='red', icon='star')
            ).add_to(m)
            folium.Circle(
                location=[lat, lon],
                radius=threshold_distance * 1000,
                color='red',
                fill=False,
                weight=1,
                opacity=0.5
            ).add_to(m)

        folium.GeoJson(
            california.__geo_interface__,
            name="California",
            style_function=lambda x: {
                'fillColor': 'none',
                'color': 'green',
                'weight': 2
            }
        ).add_to(m)

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

        m.save("combined_pso_ga_map.html")

if __name__ == "__main__":
    main()
