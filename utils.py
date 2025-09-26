import os
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

from shapely.ops import unary_union, linemerge
from shapely.geometry import Point, MultiLineString, LineString

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(filepath, **kwargs):
    """
    Load CSV or geospatial data (GPKG, SHP, GeoJSON) with error handling.
    Returns either a pandas.DataFrame or geopandas.GeoDataFrame.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return None

    try:
        # CSV or TXT → pandas
        if filepath.suffix.lower() in [".csv", ".txt"]:
            return pd.read_csv(filepath, **kwargs)

        # GeoPackage, Shapefile, GeoJSON → geopandas
        elif filepath.suffix.lower() in [".gpkg", ".shp", ".geojson"]:
            return gpd.read_file(filepath, **kwargs)

        else:
            print(f"⚠️ Unsupported file type: {filepath.suffix}")
            return None

    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return None

def load_gps_from_csv(filepath, **kwargs):
    """
    Load GPS data from CSV and convert to GeoDataFrame.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to the CSV file (e.g., '../data/zg11_20250915.csv')
    **kwargs : dict
        Additional arguments for pd.read_csv()
    
    Returns:
    --------
    gpd.GeoDataFrame
        GeoDataFrame with renamed columns and geometry
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return None
    
    try:
        # Load data
        gps_df = pd.read_csv(filepath, **kwargs)
        
        # Rename columns to English
        column_mapping = {
            '時刻': 'timestamp',
            '緯度': 'lat', 
            '経度': 'lon',
            '速度(km/h)': 'speed_kmh',
            '方向': 'direction'
        }
        gps_df = gps_df.rename(columns=column_mapping)
        
        # Convert to GeoDataFrame
        gps_gdf = gpd.GeoDataFrame(
            gps_df,
            geometry=gpd.points_from_xy(gps_df.lon, gps_df.lat),
            crs="EPSG:4326"
        )
        
        return gps_gdf
        
    except Exception as e:
        print(f"❌ Error loading GPS data from {filepath}: {e}")
        return None
    
def load_gtfs_stops(gtfs_folder, utm_crs="EPSG:32736"):
    """
    Load GTFS stops.txt from a folder as a GeoDataFrame.

    Parameters
    ----------
    gtfs_folder : str or Path
        Path to the folder containing stops.txt
    utm_crs : str
        CRS to project stops into (meters, e.g., UTM)

    Returns
    -------
    stops_gdf : gpd.GeoDataFrame
    """
    gtfs_folder = Path(gtfs_folder)
    stops_file = gtfs_folder / "stops.txt"

    if not stops_file.exists():
        print(f"❌ File not found: {stops_file}")
        return None

    try:
        stops_df = pd.read_csv(stops_file)

        required_cols = ["stop_id", "stop_name", "stop_lat", "stop_lon"]
        for col in required_cols:
            if col not in stops_df.columns:
                raise ValueError(f"{col} not found in stops.txt")

        # Convert to GeoDataFrame in WGS84
        stops_gdf = gpd.GeoDataFrame(
            stops_df,
            geometry=[Point(xy) for xy in zip(stops_df.stop_lon, stops_df.stop_lat)],
            crs="EPSG:4326"
        )

        # Reproject to UTM
        stops_gdf = stops_gdf.to_crs(utm_crs)
        return stops_gdf

    except Exception as e:
        print(f"❌ Error loading {stops_file}: {e}")
        return None


# ============================================================================
# GPS LINE DISTANCE ANALYSIS
# ============================================================================

def calculate_line_distance(gps_gdf, lines_gdf, line_id, utm_crs="EPSG:32736", plot=True, flip_origin=False):
    """
    Calculate distances along reference line for GPS points, restricted to one line_id.
    
    Parameters:
    -----------
    gps_gdf : gpd.GeoDataFrame
        GPS points GeoDataFrame
    lines_gdf : gpd.GeoDataFrame
        Lines GeoDataFrame with line_id column
    line_id : str
        Specific line ID to analyze (e.g., "L5")
    utm_crs : str, default "EPSG:32736"
        UTM coordinate reference system for distance calculations
    plot : bool, default True
        Whether to create visualization plots
    flip_origin : bool, default False
        Whether to reverse the start/end of the reference line
    
    Returns:
    --------
    tuple
        (result_gdf, ref_line) - GeoDataFrame with distance calculations and reference line geometry
    """
    
    # --- 1. Filter to one line ---
    line_gdf = lines_gdf[lines_gdf["line_id"] == line_id].copy()
    if line_gdf.empty:
        raise ValueError(f"No geometries found for line_id={line_id}")
    
    print(f"📍 Analyzing line {line_id} with {len(gps_gdf)} GPS points")

    # --- 2. Project to UTM ---
    gps_utm = gps_gdf.to_crs(utm_crs)
    line_utm = line_gdf.to_crs(utm_crs)

    # --- 3. Merge into a single geometry ---
    merged = unary_union(line_utm.geometry)

    if isinstance(merged, LineString):
        ref_line = merged
    else:
        ref_line = linemerge(merged)

    # If still MultiLineString, pick the longest
    if isinstance(ref_line, MultiLineString):
        ref_line = max(ref_line.geoms, key=lambda g: g.length)
        print(f"⚠️ MultiLineString detected, using longest segment ({ref_line.length/1000:.2f} km)")

    # --- Flip origin if requested ---
    if flip_origin:
        ref_line = LineString(list(ref_line.coords)[::-1])
        print("🔄 Reference line origin has been flipped")

    print(f"📏 Reference line length: {ref_line.length/1000:.2f} km")

    # --- 4. Calculate distances ---
    def get_line_distance(point):
        dist_along_line = ref_line.project(point)  # meters
        nearest_point = ref_line.interpolate(dist_along_line)
        offset_dist = point.distance(nearest_point)
        return pd.Series({
            "distance_along_line_m": dist_along_line,
            "distance_along_line_km": dist_along_line / 1000,
            "offset_distance_m": offset_dist,
            "nearest_point": nearest_point
        })

    distance_df = gps_utm.geometry.apply(get_line_distance)

    result_gdf = gps_gdf.copy()
    result_gdf = result_gdf.join(distance_df)

    # --- 5. Print summary statistics ---
    print("\n📊 Distance Statistics:")
    stats = result_gdf[['distance_along_line_m', 'offset_distance_m']].describe()
    print(f"Distance along line: {stats.loc['min', 'distance_along_line_m']:.0f}m to {stats.loc['max', 'distance_along_line_m']:.0f}m")
    print(f"Average offset from line: {stats.loc['mean', 'offset_distance_m']:.1f}m")
    print(f"Max offset from line: {stats.loc['max', 'offset_distance_m']:.1f}m")

    # --- 6. Create plots if requested ---
    if plot:
        # Plot 1: Map view
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Map visualization
        line_utm.plot(ax=ax1, color="black", linewidth=3, label=f"Line {line_id}")
        gps_utm.plot(ax=ax1, color="red", markersize=1, alpha=0.7, label="GPS Points")
        ax1.set_title(f"GPS Points vs Line {line_id} (UTM)")
        ax1.legend()
        ax1.set_aspect('equal')
        
        # Distance scatter plot
        ax2.scatter(result_gdf['distance_along_line_km'], 
                    result_gdf['offset_distance_m'],
                    color="red",
                    alpha=0.5, s=1)
        ax2.set_xlabel('Distance Along Line (km)')
        ax2.set_ylabel('Offset Distance (m)')
        ax2.set_title(f'GPS Points Distance from Line {line_id}')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    return result_gdf, ref_line

def plot_distance_analysis(result_gdf, line_id):
    """
    Create additional plots for distance analysis.
    
    Parameters:
    -----------
    result_gdf : gpd.GeoDataFrame
        Result from calculate_line_distance()
    line_id : str
        Line ID for plot titles
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Distance along line histogram
    axes[0,0].hist(result_gdf['distance_along_line_km'], bins=50, color="red", alpha=0.7, edgecolor='black')
    axes[0,0].set_xlabel('Distance Along Line (km)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title(f'Distribution: Distance Along Line {line_id}')
    axes[0,0].grid(True, alpha=0.3)
    
    # Offset distance histogram
    axes[0,1].hist(result_gdf['offset_distance_m'], bins=50, color="red", alpha=0.7, edgecolor='black')
    axes[0,1].set_xlabel('Offset Distance (m)')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title(f'Distribution: Offset from Line {line_id}')
    axes[0,1].grid(True, alpha=0.3)
    
    # Speed vs distance along line (if speed column exists)
    if 'speed_kmh' in result_gdf.columns:
        axes[1,0].scatter(result_gdf['distance_along_line_km'], 
                          result_gdf['speed_kmh'], color="red", alpha=0.5, s=1)
        axes[1,0].set_xlabel('Distance Along Line (km)')
        axes[1,0].set_ylabel('Speed (km/h)')
        axes[1,0].set_title(f'Speed vs Distance Along Line {line_id}')
        axes[1,0].grid(True, alpha=0.3)
    else:
        axes[1,0].text(0.5, 0.5, 'Speed data not available', 
                       ha='center', va='center', transform=axes[1,0].transAxes)
    
    # Time series if timestamp exists
    if 'timestamp' in result_gdf.columns:
        try:
            # Convert timestamp to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(result_gdf['timestamp']):
                result_gdf['timestamp'] = pd.to_datetime(result_gdf['timestamp'])
            
            axes[1,1].plot(result_gdf['timestamp'], result_gdf['distance_along_line_km'], 
                           linewidth=0.5, color="red", alpha=0.7)
            axes[1,1].set_xlabel('Time')
            axes[1,1].set_ylabel('Distance Along Line (km)')
            axes[1,1].set_title(f'Progress Along Line {line_id} Over Time')
            axes[1,1].tick_params(axis='x', rotation=45)
            axes[1,1].grid(True, alpha=0.3)
        except Exception:
            axes[1,1].text(0.5, 0.5, 'Timestamp parsing failed', 
                           ha='center', va='center', transform=axes[1,1].transAxes)
    else:
        axes[1,1].text(0.5, 0.5, 'Timestamp data not available', 
                       ha='center', va='center', transform=axes[1,1].transAxes)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# CREATING SCHEDULED TRIPS FROM GTFS
# ============================================================================

def get_routes_for_line(line_id, gtfs_dir="./data/gtfs", line_id_file="./data/line_id.csv"):
    import os
    import pandas as pd

    def parse_array(s):
        if pd.isna(s):
            return []
        return [x.strip() for x in s.strip("{}").split(",")]

    # Load line reference
    line_id_ref = pd.read_csv(
        line_id_file,
        converters={"carrier_ids": parse_array, "route_ids": parse_array}
    )

    # Explode route_ids and filter by line_id first
    line_routes = line_id_ref.explode("route_ids")
    line_routes = line_routes[line_routes["line_id"] == line_id]

    # If no routes found, return empty
    if line_routes.empty:
        return pd.DataFrame(columns=["line_id", "route_id", "direction_id"])

    # Load trips
    trips_df = pd.read_csv(os.path.join(gtfs_dir, "trips.txt"))

    # Ensure both are strings
    trips_df["route_id"] = trips_df["route_id"].astype(str)
    line_routes["route_ids"] = line_routes["route_ids"].astype(str)

    # Merge and return requested columns
    filtered = trips_df.merge(
        line_routes,
        left_on="route_id",
        right_on="route_ids",
        how="inner"
    )

    return filtered[["line_id", "route_id", "direction_id"]].drop_duplicates()

def get_timetable_for_routes(route_ids, gtfs_dir="./data/gtfs"):
    # --- Ensure route_ids is a list ---
    if isinstance(route_ids, str):
        route_ids = [route_ids]

    # --- Load GTFS files ---
    trips = pd.read_csv(os.path.join(gtfs_dir, "trips.txt"))
    stop_times = pd.read_csv(os.path.join(gtfs_dir, "stop_times.txt"))
    stops = pd.read_csv(os.path.join(gtfs_dir, "stops.txt"))

    # --- Get all trips for the chosen route_ids ---
    trips_filtered = trips[trips["route_id"].isin(route_ids)]

    # --- Filter stop_times for these trips and keep route_id + direction_id ---
    timetable = stop_times.merge(
        trips_filtered[["trip_id", "route_id", "direction_id"]],
        on="trip_id",
        how="inner"
    )

    # --- Join with stops to get stop names ---
    timetable = timetable.merge(
        stops[["stop_id", "stop_name"]],
        on="stop_id",
        how="left"
    )
    
    # === FIX: Convert GTFS time strings to timedelta to handle hours > 23 ===
    timetable['arrival_time'] = pd.to_timedelta(timetable['arrival_time'])
    timetable['departure_time'] = pd.to_timedelta(timetable['departure_time'])
    # =======================================================================

    # --- Sort by trip_id and stop_sequence ---
    timetable = timetable.sort_values(["trip_id", "stop_sequence"])

    # --- Select relevant columns ---
    timetable = timetable[["trip_id", "route_id", "direction_id", "stop_sequence", "stop_id", "stop_name", "arrival_time", "departure_time"]]

    return timetable

def create_scheduled_trips_from_gtfs(route_ids, gtfs_dir, lines_gdf, line_id,
                                      utm_crs="EPSG:32736", plot=False, flip_origin=False):
    """
    Create a scheduled trips GeoDataFrame with distances along route for given route_id(s)
    and optionally plot reference line with stops and distance-time diagram.

    Returns
    -------
    scheduled_trips_gdf : gpd.GeoDataFrame
        Scheduled stops with geometry, distance along line, and offsets
    ref_line : shapely LineString
        Reference line geometry in UTM CRS
    """

    # --- 1. Load GTFS timetable for given route_ids ---
    timetable_df = get_timetable_for_routes(route_ids, gtfs_dir=gtfs_dir)
    if timetable_df.empty:
        return gpd.GeoDataFrame(), None

    # --- 2. Load GTFS stops and convert to GeoDataFrame in UTM CRS ---
    stops_gdf = load_gtfs_stops(gtfs_dir, utm_crs=utm_crs)

    # --- 3. Build independent reference line from lines_gdf ---
    line_gdf = lines_gdf[lines_gdf["line_id"] == line_id].copy()
    if line_gdf.empty:
        raise ValueError(f"No geometries found for line_id={line_id}")

    line_utm = line_gdf.to_crs(utm_crs)
    merged = unary_union(line_utm.geometry)

    if isinstance(merged, LineString):
        ref_line = merged
    else:
        ref_line = linemerge(merged)

    if isinstance(ref_line, MultiLineString):
        ref_line = max(ref_line.geoms, key=lambda g: g.length)

    # --- Flip origin if requested ---
    if flip_origin:
        ref_line = LineString(list(ref_line.coords)[::-1])
        print("🔄 Reference line origin has been flipped")

    print(f"📏 Reference line length: {ref_line.length/1000:.2f} km")

    # --- 4. Merge timetable with stops ---
    scheduled_df = pd.merge(
        timetable_df,
        stops_gdf[['stop_id', 'geometry']],
        on='stop_id',
        how='left'
    )
    scheduled_gdf = gpd.GeoDataFrame(scheduled_df, geometry='geometry', crs=utm_crs)

    # --- 5. Calculate distances along reference line ---
    def get_line_distance(point):
        dist_along_line = ref_line.project(point)
        nearest_point = ref_line.interpolate(dist_along_line)
        offset_dist = point.distance(nearest_point)
        return pd.Series({
            "distance_along_line_m": dist_along_line,
            "distance_along_line_km": dist_along_line / 1000,
            "offset_distance_m": offset_dist
        })

    distance_df = scheduled_gdf.geometry.apply(get_line_distance)
    scheduled_trips_gdf = scheduled_gdf.join(distance_df)

    # --- 6. Optional plot ---
    if plot:
        # --- 6a. Plot reference line + stops ---
        base = line_utm.plot(color='black', linewidth=2)
        scheduled_gdf.plot(ax=base, color='red', markersize=40)
        plt.title(f"Scheduled Stops and Reference Line for {line_id}")
        plt.show()

        # --- 6b. Plot distance along line vs scheduled time ---
        scheduled_trips_gdf = scheduled_trips_gdf.sort_values('arrival_time')

        # Create a plottable time-of-day axis, similar to Tab 4
        dummy_date = pd.Timestamp("1900-01-01")
        scheduled_trips_gdf["time_of_day"] = dummy_date + scheduled_trips_gdf["arrival_time"]

        plt.figure(figsize=(12,6))
        plt.plot(
            scheduled_trips_gdf['time_of_day'], # Use the new time_of_day column
            scheduled_trips_gdf['distance_along_line_km'],
            marker='o',
            linestyle='none',
            color='blue',
            alpha=0.7
        )
        
        # Add the time formatter to show HH:MM
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
        
        plt.xlabel("Time of Day") # Update label
        plt.ylabel("Distance from Origin (km)")
        plt.title(f"Scheduled Stops Distance-Time Diagram for {line_id}")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return scheduled_trips_gdf, ref_line

# ============================================================================
# DAILY BUS OPERATION VISUALIZATION
# ============================================================================

def plot_gps_vs_scheduled(gps_gdf, scheduled_trips_gdf,
                          gps_time_col="timestamp",
                          sched_time_col="arrival_time",
                          distance_col="distance_along_line_km"):
    """
    Plot GPS trajectory and scheduled stops with time-of-day on x-axis.
    """
    # --- Prepare GPS Data ---
    gps_df = gps_gdf.copy()
    gps_df[gps_time_col] = pd.to_datetime(gps_df[gps_time_col])

    # --- Prepare Schedule Data ---
    sched_df = scheduled_trips_gdf.copy()
    # The sched_time_col is already a timedelta from the get_timetable_for_routes function

    # --- Create a common, plottable "time_of_day" axis ---
    dummy_date = pd.Timestamp("1900-01-01")

    # For GPS, get timedelta from start of its day and add to the dummy date
    gps_time_delta = gps_df[gps_time_col] - gps_df[gps_time_col].dt.normalize()
    gps_df["time_of_day"] = dummy_date + gps_time_delta

    # For schedule, add the timedelta column directly to the dummy date
    sched_df["time_of_day"] = dummy_date + sched_df[sched_time_col]

    # --- Plot ---
    plt.figure(figsize=(14,6))

    # GPS points
    plt.plot(gps_df["time_of_day"], gps_df[distance_col],
             marker='.', linestyle='-', color='red', alpha=0.6, label="GPS (Realized)")

    # Scheduled stops
    plt.scatter(sched_df["time_of_day"], sched_df[distance_col],
                marker='o', color='blue', s=40, label="Scheduled Stops")

    # Format x-axis as HH:MM
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2)) # Set major ticks every 2 hours

    plt.xlabel("Time of Day")
    plt.ylabel("Distance from Origin (km)")
    plt.title("Bus Distance Along Route Over Time (Time-of-Day X-axis)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()