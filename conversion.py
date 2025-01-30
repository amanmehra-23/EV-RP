import geopandas as gpd

# Path to the extracted Shapefile
shapefile_path = '/Users/amanmehra/Desktop/ne_10m_land/ne_10m_land.shp'  # Update if your path differs

# Load the Shapefile
land_gdf = gpd.read_file(shapefile_path)

# Save as GeoJSON
land_gdf.to_file('ne_10m_land.geojson', driver='GeoJSON')

print("Conversion to GeoJSON completed successfully.")
