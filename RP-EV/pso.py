import time
import random
import numpy as np
import pandas as pd
import folium
from folium import plugins
from branca.element import Template, MacroElement
from geopy.distance import geodesic
from pyswarm import pso

# ----------------------------------------------------------------------
# 1. Data and Threshold
# ----------------------------------------------------------------------

# Replace with your actual loading code
# Example:
df = pd.read_csv('/Users/amanmehra/Desktop/RP-EV/existing_chargers.csv')

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
        avg_distances.append(sum(closest_3)/len(closest_3))
    return sum(avg_distances)/len(avg_distances)

threshold_distance = calculate_threshold_distance(existing_locations)
print(f"Threshold distance: {threshold_distance:.2f} km")

# ----------------------------------------------------------------------
# 2. PSO
# ----------------------------------------------------------------------
N_NEW_STATIONS = 10
lat_min, lat_max = 24.396308, 49.384358
lon_min, lon_max = -125.0, -66.93457

lb = [lat_min, lon_min]*N_NEW_STATIONS
ub = [lat_max, lon_max]*N_NEW_STATIONS

def fitness_function_pso(new_station_locations):
    penalty = 1e6
    total_distance = 0
    num_new = len(new_station_locations)//2

    for i in range(num_new):
        lat = new_station_locations[2*i]
        lon = new_station_locations[2*i+1]
        new_coord = (lat, lon)

        dists_to_existing = [calculate_distance(new_coord, e) for e in existing_locations]
        min_dist = min(dists_to_existing)
        if min_dist < threshold_distance:
            return penalty
        total_distance += min_dist

    return total_distance/num_new

def run_pso_optimization():
    start = time.time()
    best_locations, best_value = pso(
        fitness_function_pso,
        lb=lb,
        ub=ub,
        swarmsize=50,
        maxiter=50
    )
    end = time.time()
    duration = end - start

    # Convert best to (lat, lon)
    best_coords = []
    for i in range(N_NEW_STATIONS):
        lat = best_locations[2*i]
        lon = best_locations[2*i+1]
        best_coords.append((lat, lon))

    return best_coords, best_value, duration

def main():
    # Run 10 times
    results = []
    for run_idx in range(1, 11):
        print(f"\n--- PSO-Only Simulation #{run_idx} ---")
        coords, fitness_val, comp_time = run_pso_optimization()
        print(f"Best Fitness: {fitness_val:.4f} | Time: {comp_time:.2f}s")
        results.append({
            'Run': run_idx,
            'Best Fitness': fitness_val,
            'Time (s)': comp_time,
            'Coordinates': coords
        })

    results_df = pd.DataFrame(results)
    print("\nAll Results:\n", results_df)

    # Example: visualize the final run’s solution with Folium
    final_run = results[-1]
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

    m = folium.Map(location=map_center, zoom_start=5)

    # Existing (Blue)
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

    # PSO (Orange)
    for lat, lon in optimal_station_coords:
        _, dist = find_nearest(existing_locations, (lat, lon))
        folium.Marker(
            location=[lat, lon],
            popup=f'PSO Station<br>Nearest existing: {dist:.2f} km',
            icon=folium.Icon(color='orange', icon='info-sign')
        ).add_to(m)
        # threshold circle
        folium.Circle(
            location=[lat, lon],
            radius=threshold_distance*1000,
            color='orange',
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
        width: 220px;
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
        <i class="fa fa-info-circle" style="color:orange"></i>
          &nbsp;PSO Locations<br>
        <i class="fa fa-circle-thin" style="color:grey"></i>
          &nbsp;Threshold Distance
    </div>
    {% endmacro %}
    '''
    legend = MacroElement()
    legend._template = Template(legend_html)
    m.get_root().add_child(legend)

    # Save the map
    m.save("pso_only_map.html")
    print("Folium map saved to 'pso_only_map.html'.")

if __name__ == "__main__":
    main()
