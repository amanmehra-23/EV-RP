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
import logging

# ----------------------------------------------------------------------
# Configure Logging
# ----------------------------------------------------------------------
logging.basicConfig(
    filename='pso_ga_debug.log',
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ----------------------------------------------------------------------
# 1. Data and Threshold
# ----------------------------------------------------------------------

# Suppress Deprecation Warnings Temporarily
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Replace this with your actual station data
# Ensure columns: ['Station Name', 'Latitude', 'Longitude']
try:
    df = pd.read_csv('existing_stations.csv')
    logging.info("existing_stations.csv loaded successfully.")
except FileNotFoundError:
    logging.error("The file 'existing_stations.csv' was not found. Please check the file path.")
    sys.exit(1)

required_columns = ['Station Name', 'Latitude', 'Longitude']
if not all(column in df.columns for column in required_columns):
    logging.error(f"CSV file is missing required columns. Required: {required_columns}")
    sys.exit(1)

existing_stations = df[['Station Name', 'Latitude', 'Longitude']]
existing_locations = [(row['Latitude'], row['Longitude']) for _, row in existing_stations.iterrows()]
logging.debug(f"Existing Locations: {existing_locations}")

def calculate_distance(point1, point2):
    """Calculate geodesic distance between two points in kilometers."""
    return geodesic(point1, point2).kilometers

def calculate_threshold_distance(locations):
    """
    Calculate a threshold distance based on each existing station's average
    distance to its 3 closest neighbors, then average those values.
    """
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
    logging.info(f"Calculated threshold distance: {threshold:.2f} km")
    return threshold

threshold_distance = calculate_threshold_distance(existing_locations)

# Number of new stations
N_NEW_STATIONS = 10

# Tight bounding box for California
lat_min, lat_max = 32.5343, 42.0095
lon_min, lon_max = -124.4096, -114.1312

# ----------------------------------------------------------------------
# 2. Load California Boundary and Land Polygon
# ----------------------------------------------------------------------

# URL to the GeoJSON file containing U.S. states
usa_states_url = 'https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json'

# Load the GeoJSON data into a GeoDataFrame
try:
    usa_states = gpd.read_file(usa_states_url)
    logging.info("USA States GeoDataFrame loaded successfully.")
except Exception as e:
    logging.error(f"Error loading USA states GeoJSON: {e}")
    sys.exit(1)

# Inspect the columns to ensure the state name column is correct
logging.debug(f"USA States GeoDataFrame Columns: {usa_states.columns}")

# Filter for California
california = usa_states[usa_states['name'] == 'California']

# Check if California was found
if california.empty:
    logging.error("California not found in the dataset. Please check the state name or dataset.")
    sys.exit(1)

# Combine all geometries into a single polygon (useful if California has multiple polygons)
if hasattr(california.geometry, 'union_all'):
    california_polygon = california.geometry.union_all()
    logging.debug("Used union_all() to combine California geometries.")
else:
    california_polygon = california.geometry.unary_union  # Retain if 'union_all()' is unavailable
    logging.debug("Used unary_union to combine California geometries.")

# ----------------------------------------------------------------------
# 2a. Load Detailed Land Polygon
# ----------------------------------------------------------------------

# Path to the 'ne_10m_land.geojson' file
land_geojson_path = 'ne_10m_land.geojson'  # Update this path if the file is in a different directory

try:
    land = gpd.read_file(land_geojson_path)
    logging.info(f"Land GeoJSON file '{land_geojson_path}' loaded successfully.")
except FileNotFoundError:
    logging.error(f"The land GeoJSON file was not found at '{land_geojson_path}'. Please check the path.")
    sys.exit(1)
except Exception as e:
    logging.error(f"Error loading land GeoJSON file: {e}")
    sys.exit(1)

# Ensure both GeoDataFrames use the same Coordinate Reference System (CRS)
if land.crs != usa_states.crs:
    land = land.to_crs(usa_states.crs)
    logging.info("Land GeoDataFrame reprojected to match USA States CRS.")

# Create a unified land polygon
if hasattr(land.geometry, 'union_all'):
    land_polygon = land.geometry.union_all()
    logging.debug("Used union_all() to combine land geometries.")
else:
    land_polygon = land.geometry.unary_union  # Retain if 'union_all()' is unavailable
    logging.debug("Used unary_union to combine land geometries.")

def is_within_california_and_land(lat, lon, california_polygon, land_polygon):
    """
    Checks if a given latitude and longitude are within California's boundaries and on land.

    Args:
        lat (float): Latitude of the point.
        lon (float): Longitude of the point.
        california_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of California.
        land_polygon (shapely.geometry.Polygon or MultiPolygon):
            The geographic boundary of land.

    Returns:
        bool: True if the point is within California and on land, False otherwise.
    """
    point = Point(lon, lat)
    return california_polygon.contains(point) and land_polygon.contains(point)

# ----------------------------------------------------------------------
# 3. PSO Setup with California and Land Constraints
# ----------------------------------------------------------------------

lb = [lat_min, lon_min] * N_NEW_STATIONS
ub = [lat_max, lon_max] * N_NEW_STATIONS

def fitness_function_pso(new_station_locations, california_polygon, land_polygon):
    """
    PSO Fitness Function that minimizes the average distance to the nearest existing station,
    while ensuring all new stations are within California, on land, and respect the threshold distance.

    Args:
        new_station_locations (list): Flat list of [lat1, lon1, lat2, lon2, ...].
        california_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of California.
        land_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of land.

    Returns:
        float: The fitness value (lower is better).
    """
    penalty_outside = 1e8  # Higher penalty for being outside California
    penalty_water = 1e7    # Penalty for being on water
    penalty_threshold = 1e6  # Penalty for violating threshold distance
    total_distance = 0
    num_new_stations = len(new_station_locations) // 2

    for i in range(num_new_stations):
        lat = new_station_locations[2 * i]
        lon = new_station_locations[2 * i + 1]
        new_coord = (lat, lon)

        # Check if within California and on land
        if not is_within_california_and_land(lat, lon, california_polygon, land_polygon):
            # Determine the type of violation
            point = Point(lon, lat)
            if not california_polygon.contains(point):
                logging.warning(f"Station {i+1} at ({lat}, {lon}) is outside California.")
                return penalty_outside  # Outside California
            elif not land_polygon.contains(point):
                logging.warning(f"Station {i+1} at ({lat}, {lon}) is on water.")
                return penalty_water     # On water

        # Calculate distance to existing stations
        distances_to_existing = [calculate_distance(new_coord, e) for e in existing_locations]
        min_dist = min(distances_to_existing)

        if min_dist < threshold_distance:
            logging.warning(f"Station {i+1} at ({lat}, {lon}) is too close to an existing station.")
            return penalty_threshold  # Too close to existing station

        total_distance += min_dist

    average_distance = total_distance / num_new_stations
    logging.debug(f"Average distance for PSO fitness: {average_distance:.2f} km")
    return average_distance

def run_pso_optimization(california_polygon, land_polygon):
    """
    Runs PSO optimization to find optimal station locations within California and on land.

    Args:
        california_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of California.
        land_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of land.

    Returns:
        tuple: (best_coords, best_fitness, duration)
    """
    start = time.time()
    logging.info("Starting PSO optimization.")

    # Define a wrapper for the fitness function to include california_polygon and land_polygon
    def fitness_wrapper(new_station_locations):
        return fitness_function_pso(new_station_locations, california_polygon, land_polygon)

    try:
        best_locations, best_value = pso(
            fitness_wrapper,
            lb=lb,
            ub=ub,
            swarmsize=10,   # Reduced for initial testing
            maxiter=10,     # Reduced for initial testing
            debug=True      # Enable debug output
        )
        end = time.time()
        duration = end - start
        logging.info(f"PSO optimization completed in {duration:.2f} seconds.")
    except Exception as e:
        logging.error(f"Error during PSO optimization: {e}")
        sys.exit(1)

    # Convert best_locations to list of tuples
    best_coords = [(best_locations[2 * i], best_locations[2 * i + 1]) for i in range(N_NEW_STATIONS)]
    logging.debug(f"Best PSO Coordinates: {best_coords}")
    logging.debug(f"Best PSO Fitness Value: {best_value}")

    return best_coords, best_value, duration

# ----------------------------------------------------------------------
# 4. GA (DEAP) Setup with California Constraints
# ----------------------------------------------------------------------
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

def generate_location(california_polygon, land_polygon):
    """
    Generates a valid [lat, lon] pair within California, on land, and respecting the threshold distance.

    Args:
        california_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of California.
        land_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of land.

    Returns:
        list: A list containing [lat, lon].
    """
    attempts = 0
    max_attempts = 1000  # Prevent infinite loops

    while attempts < max_attempts:
        lat = random.uniform(lat_min, lat_max)
        lon = random.uniform(lon_min, lon_max)
        if is_within_california_and_land(lat, lon, california_polygon, land_polygon):
            distances = [calculate_distance((lat, lon), e) for e in existing_locations]
            if min(distances) >= threshold_distance:
                logging.debug(f"Generated valid location: ({lat}, {lon})")
                return [lat, lon]
        attempts += 1

    logging.warning("generate_location exceeded maximum attempts.")
    raise ValueError("Exceeded maximum attempts to generate a valid location within California on land.")

# GA Individual Initialization
def init_individual(california_polygon, land_polygon):
    """
    Initializes an individual with N_NEW_STATIONS within California and on land.

    Args:
        california_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of California.
        land_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of land.

    Returns:
        Individual: A DEAP Individual.
    """
    coords = []
    for _ in range(N_NEW_STATIONS):
        coords.extend(generate_location(california_polygon, land_polygon))
    logging.debug(f"Initialized GA Individual: {coords}")
    return creator.Individual(coords)

# Register GA components with california_polygon and land_polygon
toolbox.register("individual", init_individual, california_polygon=california_polygon, land_polygon=land_polygon)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def fitness_function_ga(individual, california_polygon, land_polygon):
    """
    GA Fitness Function that minimizes the average distance to the nearest existing station,
    while ensuring all new stations are within California, on land, and respect the threshold distance.

    Args:
        individual (list): Flat list of [lat1, lon1, lat2, lon2, ...].
        california_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of California.
        land_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of land.

    Returns:
        tuple: The fitness value.
    """
    penalty_outside = 1e8  # Higher penalty for being outside California
    penalty_water = 1e7    # Penalty for being on water
    penalty_threshold = 1e6  # Penalty for violating threshold distance
    total_distance = 0
    num_new_stations = len(individual) // 2

    for i in range(num_new_stations):
        lat = individual[2 * i]
        lon = individual[2 * i + 1]
        new_coord = (lat, lon)

        # Check if within California and on land
        if not is_within_california_and_land(lat, lon, california_polygon, land_polygon):
            # Determine the type of violation
            point = Point(lon, lat)
            if not california_polygon.contains(point):
                logging.warning(f"GA Station {i+1} at ({lat}, {lon}) is outside California.")
                return (penalty_outside,)
            elif not land_polygon.contains(point):
                logging.warning(f"GA Station {i+1} at ({lat}, {lon}) is on water.")
                return (penalty_water,)

        # Calculate distance to existing stations
        distances_to_existing = [calculate_distance(new_coord, e) for e in existing_locations]
        min_dist = min(distances_to_existing)

        if min_dist < threshold_distance:
            logging.warning(f"GA Station {i+1} at ({lat}, {lon}) is too close to an existing station.")
            return (penalty_threshold,)

        total_distance += min_dist

    average_distance = total_distance / num_new_stations
    logging.debug(f"Average GA fitness: {average_distance:.2f} km")
    return (average_distance,)

toolbox.register("evaluate", fitness_function_ga, california_polygon=california_polygon, land_polygon=land_polygon)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxTwoPoint)

def constrained_mutate(individual, california_polygon, land_polygon):
    """
    Mutates an individual while ensuring constraints:
    - New locations are within California and on land.
    - New locations respect the threshold distance from existing stations.

    Args:
        individual (list): Flat list of [lat1, lon1, lat2, lon2, ...].
        california_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of California.
        land_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of land.

    Returns:
        tuple: The mutated individual.
    """
    mutation_probability = 0.1  # Probability to mutate each gene pair

    for i in range(0, len(individual), 2):
        if random.random() < mutation_probability:
            # Apply Gaussian mutation
            original_lat = individual[i]
            original_lon = individual[i + 1]
            individual[i] += random.gauss(0, 0.01)      # Mutate latitude
            individual[i + 1] += random.gauss(0, 0.01)  # Mutate longitude
            logging.debug(f"Mutated station {i//2 + 1} from ({original_lat}, {original_lon}) to ({individual[i]}, {individual[i + 1]})")

            # Clamp to bounding box
            individual[i] = max(min(individual[i], lat_max), lat_min)
            individual[i + 1] = max(min(individual[i + 1], lon_max), lon_min)

            # Check if within California and on land
            if not is_within_california_and_land(individual[i], individual[i + 1], california_polygon, land_polygon):
                # Reinitialize if out of bounds or on water
                try:
                    new_coord = generate_location(california_polygon, land_polygon)
                    individual[i], individual[i + 1] = new_coord
                    logging.debug(f"Reinitialized station {i//2 + 1} to ({new_coord[0]}, {new_coord[1]})")
                except ValueError:
                    # If unable to generate a new valid location, keep the mutated value
                    logging.error(f"Failed to reinitialize station {i//2 + 1} after mutation.")
            else:
                # Check distance constraint
                distances = [calculate_distance((individual[i], individual[i + 1]), e) for e in existing_locations]
                if min(distances) < threshold_distance:
                    # Reinitialize if distance constraint violated
                    try:
                        new_coord = generate_location(california_polygon, land_polygon)
                        individual[i], individual[i + 1] = new_coord
                        logging.debug(f"Reinitialized station {i//2 + 1} due to distance violation to ({new_coord[0]}, {new_coord[1]})")
                    except ValueError:
                        # If unable to generate a new valid location, keep the mutated value
                        logging.error(f"Failed to reinitialize station {i//2 + 1} due to distance violation.")

    return (individual,)

toolbox.register("mutate", constrained_mutate, california_polygon=california_polygon, land_polygon=land_polygon)

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
    logging.debug(f"Initialized GA Individual from PSO: {ind_data}")
    return creator.Individual(ind_data)

# ----------------------------------------------------------------------
# 5. Combined Routine: PSO -> GA
# ----------------------------------------------------------------------
def run_pso_then_ga(california_polygon, land_polygon):
    """
    1) Runs PSO to find an initial solution within California and on land.
    2) Uses that solution as part of the GA initial population.
    3) Returns the GA final best solution and times.

    Args:
        california_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of California.
        land_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of land.

    Returns:
        tuple: (pso_coords, pso_fitness, ga_coords, ga_fitness, pso_time, ga_time, total_time)
    """
    # 1) PSO Optimization
    pso_coords, pso_fitness, pso_time = run_pso_optimization(california_polygon, land_polygon)

    # 2) GA Optimization
    ga_start = time.time()
    pop_size = 50
    n_gen = 50
    cx_prob = 0.5
    mut_prob = 0.2

    # Initialize GA population
    try:
        population = toolbox.population(n=pop_size - 1)
        logging.info(f"Initialized GA population with size {pop_size - 1}.")
    except Exception as e:
        logging.error(f"Error initializing GA population: {e}")
        sys.exit(1)

    # Add PSO solution as one individual
    try:
        pso_individual = initialize_population_from_pso(pso_coords)
        population.append(pso_individual)
        logging.info("Added PSO solution to GA population.")
    except ValueError as ve:
        logging.error(f"Error initializing GA individual from PSO solution: {ve}")
        # Proceed without adding PSO solution
        pass

    # Evaluate all individuals
    for ind in population:
        try:
            ind.fitness.values = toolbox.evaluate(ind)
            logging.debug(f"Evaluated individual with fitness {ind.fitness.values}.")
        except Exception as e:
            logging.error(f"Error evaluating individual: {e}")
            ind.fitness.values = (1e12,)  # Assign a high penalty

    # Evolutionary loop
    for gen in range(n_gen):
        logging.info(f"GA Generation {gen + 1}/{n_gen}")
        # Selection
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
                logging.debug("Performed crossover between two individuals.")

        # Mutation
        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values
                logging.debug("Performed mutation on an individual.")

        # Evaluate invalid individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            try:
                ind.fitness.values = toolbox.evaluate(ind)
                logging.debug(f"Evaluated mutated individual with fitness {ind.fitness.values}.")
            except Exception as e:
                logging.error(f"Error evaluating mutated individual: {e}")
                ind.fitness.values = (1e12,)  # Assign a high penalty

        # Replace population
        population[:] = offspring
        logging.debug("Replaced old population with new offspring.")

    ga_end = time.time()
    ga_time = ga_end - ga_start
    logging.info(f"GA optimization completed in {ga_time:.2f} seconds.")

    # Best GA individual
    try:
        best_ind = min(population, key=lambda ind: ind.fitness.values[0])
        ga_fitness = best_ind.fitness.values[0]
        ga_coords = [(best_ind[2 * i], best_ind[2 * i + 1]) for i in range(N_NEW_STATIONS)]
        logging.info(f"GA found a best individual with fitness {ga_fitness:.4f}.")
    except ValueError:
        logging.error("No valid individuals found in GA population.")
        ga_fitness = None
        ga_coords = None

    total_time = pso_time + ga_time
    logging.info(f"Total optimization time: {total_time:.2f} seconds.")

    return pso_coords, pso_fitness, ga_coords, ga_fitness, pso_time, ga_time, total_time

# ----------------------------------------------------------------------
# 6. Main Function
# ----------------------------------------------------------------------
def main():
    """
    Main function to run combined PSO and GA optimizations and visualize results.
    """
    results = []
    for run_idx in range(1, 2):  # Changed to run once for initial testing
        logging.info(f"--- Combined PSO+GA Simulation #{run_idx} ---")
        try:
            pso_coords, pso_fitness, ga_coords, ga_fitness, pso_time, ga_time, total_time = run_pso_then_ga(california_polygon, land_polygon)

            print(f"\n--- Combined PSO+GA Simulation #{run_idx} ---")
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
            logging.error(f"Run #{run_idx} failed: {ve}")
            print(f"\n--- Combined PSO+GA Simulation #{run_idx} ---")
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
    logging.info("All results saved to pso-ga.csv.")

    # Visualization with Folium for the final run
    final_run = results[-1]
    if final_run['PSO Coordinates'] is not None and final_run['GA Refined Coordinates'] is not None:
        pso_coords = final_run['PSO Coordinates']
        ga_coords = final_run['GA Refined Coordinates']

        # Function to find the nearest existing station
        def find_nearest(existing_stations, new_station):
            dists = [geodesic(new_station, st).kilometers for st in existing_locations]
            min_d = min(dists)
            nearest_station = existing_stations[dists.index(min_d)]
            return nearest_station, min_d

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
        for idx, (lat, lon) in enumerate(pso_coords, 1):
            nearest, dist = find_nearest(existing_locations, (lat, lon))
            folium.Marker(
                location=[lat, lon],
                popup=f'PSO Station {idx}<br>Nearest existing: {dist:.2f} km',
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
        for idx, (lat, lon) in enumerate(ga_coords, 1):
            nearest, dist = find_nearest(existing_locations, (lat, lon))
            folium.Marker(
                location=[lat, lon],
                popup=f'GA Refined Station {idx}<br>Nearest existing: {dist:.2f} km',
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

        # Highlight California Boundary (Green)
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
        logging.info("Folium map saved to 'combined_pso_ga_map.html'.")
    else:
        print("No valid coordinates to visualize.")
        logging.warning("No valid coordinates to visualize.")

if __name__ == "__main__":
    main()
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
import logging

# ----------------------------------------------------------------------
# Configure Logging
# ----------------------------------------------------------------------
logging.basicConfig(
    filename='pso_ga_debug.log',
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ----------------------------------------------------------------------
# 1. Data and Threshold
# ----------------------------------------------------------------------

# Suppress Deprecation Warnings Temporarily
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Replace this with your actual station data
# Ensure columns: ['Station Name', 'Latitude', 'Longitude']
try:
    df = pd.read_csv('existing_stations.csv')
    logging.info("existing_stations.csv loaded successfully.")
except FileNotFoundError:
    logging.error("The file 'existing_stations.csv' was not found. Please check the file path.")
    sys.exit(1)

required_columns = ['Station Name', 'Latitude', 'Longitude']
if not all(column in df.columns for column in required_columns):
    logging.error(f"CSV file is missing required columns. Required: {required_columns}")
    sys.exit(1)

existing_stations = df[['Station Name', 'Latitude', 'Longitude']]
existing_locations = [(row['Latitude'], row['Longitude']) for _, row in existing_stations.iterrows()]
logging.debug(f"Existing Locations: {existing_locations}")

def calculate_distance(point1, point2):
    """Calculate geodesic distance between two points in kilometers."""
    return geodesic(point1, point2).kilometers

def calculate_threshold_distance(locations):
    """
    Calculate a threshold distance based on each existing station's average
    distance to its 3 closest neighbors, then average those values.
    """
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
    logging.info(f"Calculated threshold distance: {threshold:.2f} km")
    return threshold

threshold_distance = calculate_threshold_distance(existing_locations)

# Number of new stations
N_NEW_STATIONS = 10

# Tight bounding box for California
lat_min, lat_max = 32.5343, 42.0095
lon_min, lon_max = -124.4096, -114.1312

# ----------------------------------------------------------------------
# 2. Load California Boundary and Land Polygon
# ----------------------------------------------------------------------

# URL to the GeoJSON file containing U.S. states
usa_states_url = 'https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json'

# Load the GeoJSON data into a GeoDataFrame
try:
    usa_states = gpd.read_file(usa_states_url)
    logging.info("USA States GeoDataFrame loaded successfully.")
except Exception as e:
    logging.error(f"Error loading USA states GeoJSON: {e}")
    sys.exit(1)

# Inspect the columns to ensure the state name column is correct
logging.debug(f"USA States GeoDataFrame Columns: {usa_states.columns}")

# Filter for California
california = usa_states[usa_states['name'] == 'California']

# Check if California was found
if california.empty:
    logging.error("California not found in the dataset. Please check the state name or dataset.")
    sys.exit(1)

# Combine all geometries into a single polygon (useful if California has multiple polygons)
if hasattr(california.geometry, 'union_all'):
    california_polygon = california.geometry.union_all()
    logging.debug("Used union_all() to combine California geometries.")
else:
    california_polygon = california.geometry.unary_union  # Retain if 'union_all()' is unavailable
    logging.debug("Used unary_union to combine California geometries.")

# ----------------------------------------------------------------------
# 2a. Load Detailed Land Polygon
# ----------------------------------------------------------------------

# Path to the 'ne_10m_land.geojson' file
land_geojson_path = 'ne_10m_land.geojson'  # Update this path if the file is in a different directory

try:
    land = gpd.read_file(land_geojson_path)
    logging.info(f"Land GeoJSON file '{land_geojson_path}' loaded successfully.")
except FileNotFoundError:
    logging.error(f"The land GeoJSON file was not found at '{land_geojson_path}'. Please check the path.")
    sys.exit(1)
except Exception as e:
    logging.error(f"Error loading land GeoJSON file: {e}")
    sys.exit(1)

# Ensure both GeoDataFrames use the same Coordinate Reference System (CRS)
if land.crs != usa_states.crs:
    land = land.to_crs(usa_states.crs)
    logging.info("Land GeoDataFrame reprojected to match USA States CRS.")

# Create a unified land polygon
if hasattr(land.geometry, 'union_all'):
    land_polygon = land.geometry.union_all()
    logging.debug("Used union_all() to combine land geometries.")
else:
    land_polygon = land.geometry.unary_union  # Retain if 'union_all()' is unavailable
    logging.debug("Used unary_union to combine land geometries.")

def is_within_california_and_land(lat, lon, california_polygon, land_polygon):
    """
    Checks if a given latitude and longitude are within California's boundaries and on land.

    Args:
        lat (float): Latitude of the point.
        lon (float): Longitude of the point.
        california_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of California.
        land_polygon (shapely.geometry.Polygon or MultiPolygon):
            The geographic boundary of land.

    Returns:
        bool: True if the point is within California and on land, False otherwise.
    """
    point = Point(lon, lat)
    return california_polygon.contains(point) and land_polygon.contains(point)

# ----------------------------------------------------------------------
# 3. PSO Setup with California and Land Constraints
# ----------------------------------------------------------------------

lb = [lat_min, lon_min] * N_NEW_STATIONS
ub = [lat_max, lon_max] * N_NEW_STATIONS

def fitness_function_pso(new_station_locations, california_polygon, land_polygon):
    """
    PSO Fitness Function that minimizes the average distance to the nearest existing station,
    while ensuring all new stations are within California, on land, and respect the threshold distance.

    Args:
        new_station_locations (list): Flat list of [lat1, lon1, lat2, lon2, ...].
        california_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of California.
        land_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of land.

    Returns:
        float: The fitness value (lower is better).
    """
    penalty_outside = 1e8  # Higher penalty for being outside California
    penalty_water = 1e7    # Penalty for being on water
    penalty_threshold = 1e6  # Penalty for violating threshold distance
    total_distance = 0
    num_new_stations = len(new_station_locations) // 2

    for i in range(num_new_stations):
        lat = new_station_locations[2 * i]
        lon = new_station_locations[2 * i + 1]
        new_coord = (lat, lon)

        # Check if within California and on land
        if not is_within_california_and_land(lat, lon, california_polygon, land_polygon):
            # Determine the type of violation
            point = Point(lon, lat)
            if not california_polygon.contains(point):
                logging.warning(f"Station {i+1} at ({lat}, {lon}) is outside California.")
                return penalty_outside  # Outside California
            elif not land_polygon.contains(point):
                logging.warning(f"Station {i+1} at ({lat}, {lon}) is on water.")
                return penalty_water     # On water

        # Calculate distance to existing stations
        distances_to_existing = [calculate_distance(new_coord, e) for e in existing_locations]
        min_dist = min(distances_to_existing)

        if min_dist < threshold_distance:
            logging.warning(f"Station {i+1} at ({lat}, {lon}) is too close to an existing station.")
            return penalty_threshold  # Too close to existing station

        total_distance += min_dist

    average_distance = total_distance / num_new_stations
    logging.debug(f"Average distance for PSO fitness: {average_distance:.2f} km")
    return average_distance

def run_pso_optimization(california_polygon, land_polygon):
    """
    Runs PSO optimization to find optimal station locations within California and on land.

    Args:
        california_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of California.
        land_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of land.

    Returns:
        tuple: (best_coords, best_fitness, duration)
    """
    start = time.time()
    logging.info("Starting PSO optimization.")

    # Define a wrapper for the fitness function to include california_polygon and land_polygon
    def fitness_wrapper(new_station_locations):
        return fitness_function_pso(new_station_locations, california_polygon, land_polygon)

    try:
        best_locations, best_value = pso(
            fitness_wrapper,
            lb=lb,
            ub=ub,
            swarmsize=10,   # Reduced for initial testing
            maxiter=10,     # Reduced for initial testing
            debug=True      # Enable debug output
        )
        end = time.time()
        duration = end - start
        logging.info(f"PSO optimization completed in {duration:.2f} seconds.")
    except Exception as e:
        logging.error(f"Error during PSO optimization: {e}")
        sys.exit(1)

    # Convert best_locations to list of tuples
    best_coords = [(best_locations[2 * i], best_locations[2 * i + 1]) for i in range(N_NEW_STATIONS)]
    logging.debug(f"Best PSO Coordinates: {best_coords}")
    logging.debug(f"Best PSO Fitness Value: {best_value}")

    return best_coords, best_value, duration

# ----------------------------------------------------------------------
# 4. GA (DEAP) Setup with California Constraints
# ----------------------------------------------------------------------
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

def generate_location(california_polygon, land_polygon):
    """
    Generates a valid [lat, lon] pair within California, on land, and respecting the threshold distance.

    Args:
        california_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of California.
        land_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of land.

    Returns:
        list: A list containing [lat, lon].
    """
    attempts = 0
    max_attempts = 1000  # Prevent infinite loops

    while attempts < max_attempts:
        lat = random.uniform(lat_min, lat_max)
        lon = random.uniform(lon_min, lon_max)
        if is_within_california_and_land(lat, lon, california_polygon, land_polygon):
            distances = [calculate_distance((lat, lon), e) for e in existing_locations]
            if min(distances) >= threshold_distance:
                logging.debug(f"Generated valid location: ({lat}, {lon})")
                return [lat, lon]
        attempts += 1

    logging.warning("generate_location exceeded maximum attempts.")
    raise ValueError("Exceeded maximum attempts to generate a valid location within California on land.")

# GA Individual Initialization
def init_individual(california_polygon, land_polygon):
    """
    Initializes an individual with N_NEW_STATIONS within California and on land.

    Args:
        california_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of California.
        land_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of land.

    Returns:
        Individual: A DEAP Individual.
    """
    coords = []
    for _ in range(N_NEW_STATIONS):
        coords.extend(generate_location(california_polygon, land_polygon))
    logging.debug(f"Initialized GA Individual: {coords}")
    return creator.Individual(coords)

# Register GA components with california_polygon and land_polygon
toolbox.register("individual", init_individual, california_polygon=california_polygon, land_polygon=land_polygon)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def fitness_function_ga(individual, california_polygon, land_polygon):
    """
    GA Fitness Function that minimizes the average distance to the nearest existing station,
    while ensuring all new stations are within California, on land, and respect the threshold distance.

    Args:
        individual (list): Flat list of [lat1, lon1, lat2, lon2, ...].
        california_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of California.
        land_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of land.

    Returns:
        tuple: The fitness value.
    """
    penalty_outside = 1e8  # Higher penalty for being outside California
    penalty_water = 1e7    # Penalty for being on water
    penalty_threshold = 1e6  # Penalty for violating threshold distance
    total_distance = 0
    num_new_stations = len(individual) // 2

    for i in range(num_new_stations):
        lat = individual[2 * i]
        lon = individual[2 * i + 1]
        new_coord = (lat, lon)

        # Check if within California and on land
        if not is_within_california_and_land(lat, lon, california_polygon, land_polygon):
            # Determine the type of violation
            point = Point(lon, lat)
            if not california_polygon.contains(point):
                logging.warning(f"GA Station {i+1} at ({lat}, {lon}) is outside California.")
                return (penalty_outside,)
            elif not land_polygon.contains(point):
                logging.warning(f"GA Station {i+1} at ({lat}, {lon}) is on water.")
                return (penalty_water,)

        # Calculate distance to existing stations
        distances_to_existing = [calculate_distance(new_coord, e) for e in existing_locations]
        min_dist = min(distances_to_existing)

        if min_dist < threshold_distance:
            logging.warning(f"GA Station {i+1} at ({lat}, {lon}) is too close to an existing station.")
            return (penalty_threshold,)

        total_distance += min_dist

    average_distance = total_distance / num_new_stations
    logging.debug(f"Average GA fitness: {average_distance:.2f} km")
    return (average_distance,)

toolbox.register("evaluate", fitness_function_ga, california_polygon=california_polygon, land_polygon=land_polygon)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxTwoPoint)

def constrained_mutate(individual, california_polygon, land_polygon):
    """
    Mutates an individual while ensuring constraints:
    - New locations are within California and on land.
    - New locations respect the threshold distance from existing stations.

    Args:
        individual (list): Flat list of [lat1, lon1, lat2, lon2, ...].
        california_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of California.
        land_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of land.

    Returns:
        tuple: The mutated individual.
    """
    mutation_probability = 0.1  # Probability to mutate each gene pair

    for i in range(0, len(individual), 2):
        if random.random() < mutation_probability:
            # Apply Gaussian mutation
            original_lat = individual[i]
            original_lon = individual[i + 1]
            individual[i] += random.gauss(0, 0.01)      # Mutate latitude
            individual[i + 1] += random.gauss(0, 0.01)  # Mutate longitude
            logging.debug(f"Mutated station {i//2 + 1} from ({original_lat}, {original_lon}) to ({individual[i]}, {individual[i + 1]})")

            # Clamp to bounding box
            individual[i] = max(min(individual[i], lat_max), lat_min)
            individual[i + 1] = max(min(individual[i + 1], lon_max), lon_min)

            # Check if within California and on land
            if not is_within_california_and_land(individual[i], individual[i + 1], california_polygon, land_polygon):
                # Reinitialize if out of bounds or on water
                try:
                    new_coord = generate_location(california_polygon, land_polygon)
                    individual[i], individual[i + 1] = new_coord
                    logging.debug(f"Reinitialized station {i//2 + 1} to ({new_coord[0]}, {new_coord[1]})")
                except ValueError:
                    # If unable to generate a new valid location, keep the mutated value
                    logging.error(f"Failed to reinitialize station {i//2 + 1} after mutation.")
            else:
                # Check distance constraint
                distances = [calculate_distance((individual[i], individual[i + 1]), e) for e in existing_locations]
                if min(distances) < threshold_distance:
                    # Reinitialize if distance constraint violated
                    try:
                        new_coord = generate_location(california_polygon, land_polygon)
                        individual[i], individual[i + 1] = new_coord
                        logging.debug(f"Reinitialized station {i//2 + 1} due to distance violation to ({new_coord[0]}, {new_coord[1]})")
                    except ValueError:
                        # If unable to generate a new valid location, keep the mutated value
                        logging.error(f"Failed to reinitialize station {i//2 + 1} due to distance violation.")

    return (individual,)

toolbox.register("mutate", constrained_mutate, california_polygon=california_polygon, land_polygon=land_polygon)

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
    logging.debug(f"Initialized GA Individual from PSO: {ind_data}")
    return creator.Individual(ind_data)

# ----------------------------------------------------------------------
# 5. Combined Routine: PSO -> GA
# ----------------------------------------------------------------------
def run_pso_then_ga(california_polygon, land_polygon):
    """
    1) Runs PSO to find an initial solution within California and on land.
    2) Uses that solution as part of the GA initial population.
    3) Returns the GA final best solution and times.

    Args:
        california_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of California.
        land_polygon (shapely.geometry.Polygon or MultiPolygon): 
            The geographic boundary of land.

    Returns:
        tuple: (pso_coords, pso_fitness, ga_coords, ga_fitness, pso_time, ga_time, total_time)
    """
    # 1) PSO Optimization
    pso_coords, pso_fitness, pso_time = run_pso_optimization(california_polygon, land_polygon)

    # 2) GA Optimization
    ga_start = time.time()
    pop_size = 50
    n_gen = 50
    cx_prob = 0.5
    mut_prob = 0.2

    # Initialize GA population
    try:
        population = toolbox.population(n=pop_size - 1)
        logging.info(f"Initialized GA population with size {pop_size - 1}.")
    except Exception as e:
        logging.error(f"Error initializing GA population: {e}")
        sys.exit(1)

    # Add PSO solution as one individual
    try:
        pso_individual = initialize_population_from_pso(pso_coords)
        population.append(pso_individual)
        logging.info("Added PSO solution to GA population.")
    except ValueError as ve:
        logging.error(f"Error initializing GA individual from PSO solution: {ve}")
        # Proceed without adding PSO solution
        pass

    # Evaluate all individuals
    for ind in population:
        try:
            ind.fitness.values = toolbox.evaluate(ind)
            logging.debug(f"Evaluated individual with fitness {ind.fitness.values}.")
        except Exception as e:
            logging.error(f"Error evaluating individual: {e}")
            ind.fitness.values = (1e12,)  # Assign a high penalty

    # Evolutionary loop
    for gen in range(n_gen):
        logging.info(f"GA Generation {gen + 1}/{n_gen}")
        # Selection
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
                logging.debug("Performed crossover between two individuals.")

        # Mutation
        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values
                logging.debug("Performed mutation on an individual.")

        # Evaluate invalid individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            try:
                ind.fitness.values = toolbox.evaluate(ind)
                logging.debug(f"Evaluated mutated individual with fitness {ind.fitness.values}.")
            except Exception as e:
                logging.error(f"Error evaluating mutated individual: {e}")
                ind.fitness.values = (1e12,)  # Assign a high penalty

        # Replace population
        population[:] = offspring
        logging.debug("Replaced old population with new offspring.")

    ga_end = time.time()
    ga_time = ga_end - ga_start
    logging.info(f"GA optimization completed in {ga_time:.2f} seconds.")

    # Best GA individual
    try:
        best_ind = min(population, key=lambda ind: ind.fitness.values[0])
        ga_fitness = best_ind.fitness.values[0]
        ga_coords = [(best_ind[2 * i], best_ind[2 * i + 1]) for i in range(N_NEW_STATIONS)]
        logging.info(f"GA found a best individual with fitness {ga_fitness:.4f}.")
    except ValueError:
        logging.error("No valid individuals found in GA population.")
        ga_fitness = None
        ga_coords = None

    total_time = pso_time + ga_time
    logging.info(f"Total optimization time: {total_time:.2f} seconds.")

    return pso_coords, pso_fitness, ga_coords, ga_fitness, pso_time, ga_time, total_time

# ----------------------------------------------------------------------
# 6. Main Function
# ----------------------------------------------------------------------
def main():
    """
    Main function to run combined PSO and GA optimizations and visualize results.
    """
    results = []
    for run_idx in range(1, 2):  # Changed to run once for initial testing
        logging.info(f"--- Combined PSO+GA Simulation #{run_idx} ---")
        try:
            pso_coords, pso_fitness, ga_coords, ga_fitness, pso_time, ga_time, total_time = run_pso_then_ga(california_polygon, land_polygon)

            print(f"\n--- Combined PSO+GA Simulation #{run_idx} ---")
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
            logging.error(f"Run #{run_idx} failed: {ve}")
            print(f"\n--- Combined PSO+GA Simulation #{run_idx} ---")
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
    logging.info("All results saved to pso-ga.csv.")

    # Visualization with Folium for the final run
    final_run = results[-1]
    if final_run['PSO Coordinates'] is not None and final_run['GA Refined Coordinates'] is not None:
        pso_coords = final_run['PSO Coordinates']
        ga_coords = final_run['GA Refined Coordinates']

        # Function to find the nearest existing station
        def find_nearest(existing_stations, new_station):
            dists = [geodesic(new_station, st).kilometers for st in existing_locations]
            min_d = min(dists)
            nearest_station = existing_stations[dists.index(min_d)]
            return nearest_station, min_d

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
        for idx, (lat, lon) in enumerate(pso_coords, 1):
            nearest, dist = find_nearest(existing_locations, (lat, lon))
            folium.Marker(
                location=[lat, lon],
                popup=f'PSO Station {idx}<br>Nearest existing: {dist:.2f} km',
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
        for idx, (lat, lon) in enumerate(ga_coords, 1):
            nearest, dist = find_nearest(existing_locations, (lat, lon))
            folium.Marker(
                location=[lat, lon],
                popup=f'GA Refined Station {idx}<br>Nearest existing: {dist:.2f} km',
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

        # Highlight California Boundary (Green)
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
        logging.info("Folium map saved to 'combined_pso_ga_map.html'.")
    else:
        print("No valid coordinates to visualize.")
        logging.warning("No valid coordinates to visualize.")

if __name__ == "__main__":
    main()
