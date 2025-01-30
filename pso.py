import time
import random
import numpy as np
import pandas as pd
import folium
from folium import plugins
from branca.element import Template, MacroElement
from geopy.distance import geodesic
from pyswarm import pso
import geopandas as gpd
from shapely.geometry import Point

# ----------------------------------------------------------------------
# 1. Data and Threshold
# ----------------------------------------------------------------------

# Replace this with your actual station data
# Example:
df = pd.read_csv('/Users/amanmehra/Desktop/RP-EV/existing_stations.csv')
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
N_NEW_STATIONS = 10

# Tight bounding box for California
lat_min, lat_max = 32.5343, 42.0095
lon_min, lon_max = -124.4096, -114.1312

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
# 4. Main Function
# ----------------------------------------------------------------------

def main():
    # Ensure California polygon is in the correct format
    california_polygon = california.geometry.unary_union

    # Run 10 times
    results = []
    for run_idx in range(1, 11):
        print(f"\n--- PSO-Only Simulation #{run_idx} ---")
        try:
            coords, fitness_val, comp_time = run_pso_optimization(california_polygon)
            print(f"Best Fitness: {fitness_val:.4f} | Time: {comp_time:.2f}s")
            results.append({
                'Run': run_idx,
                'Best Fitness': fitness_val,
                'Time (s)': comp_time,
                'Coordinates': coords
            })
        except Exception as e:
            print(f"Run #{run_idx} failed: {e}")
            results.append({
                'Run': run_idx,
                'Best Fitness': None,
                'Time (s)': None,
                'Coordinates': None
            })

    results_df = pd.DataFrame(results)
    print("\nAll Results:\n", results_df)
    results_df.to_csv("pso.csv", index=False)

    # Example: visualize the final run’s solution with Folium
    final_run = results[-1]
    if final_run['Coordinates'] is not None:
        optimal_station_coords = final_run['Coordinates']

        # Build the map
        def find_nearest(existing_stations, new_station):
            dists = [geodesic(new_station, st).kilometers for st in existing_stations]
            min_d = min(dists)
            return existing_stations[dists.index(min_d)], min_d

        # Combine existing + final run coords for map center
        all_coords = existing_locations + optimal_station_coords
        all_lats = [c[0] for c in all_coords]
        all_lons = [c[1] for c in all_coords]
        map_center = [np.mean(all_lats), np.mean(all_lons)]

        m = folium.Map(location=map_center, zoom_start=6)

        # Existing (Blue)
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

        # PSO (Orange)
        for lat, lon in optimal_station_coords:
            nearest, dist = find_nearest(existing_locations, (lat, lon))
            folium.Marker(
                location=[lat, lon],
                popup=f'PSO Station<br>Nearest existing: {dist:.2f} km',
                icon=folium.Icon(color='orange', icon='info-sign')
            ).add_to(m)
            # Threshold circle
            folium.Circle(
                location=[lat, lon],
                radius=threshold_distance * 1000,  # Convert km to meters
                color='orange',
                fill=False,
                weight=1,
                opacity=0.5,
                tooltip=f'Threshold: {threshold_distance:.2f} km'
            ).add_to(m)

        # Highlight California boundary
        folium.GeoJson(
            california.__geo_interface__,
            name="California",
            style_function=lambda x: {
                'fillColor': 'none',
                'color': 'green',
                'weight': 2
            }
        ).add_to(m)

        # Legend
        legend_html = '''
        {% macro html(this, kwargs) %}
        <div style="
            position: fixed;
            bottom: 50px;
            left: 50px;
            width: 220px;
            height: 150px;
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
        m.save("pso_only_map.html")
        print("Folium map saved to 'pso_only_map.html'.")
    else:
        print("No valid coordinates to visualize.")

if __name__ == "__main__":
    main()
