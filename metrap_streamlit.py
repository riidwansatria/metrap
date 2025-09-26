import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io
import sys
from contextlib import redirect_stdout

# Import utils module
try:
    import utils
except ImportError:
    st.error("Utils module not found. Please ensure utils.py is in the same directory or in Python path.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="METRAP Bus Operations Analysis",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("METRAP Bus Operations Analysis")
st.markdown("**Using Bus GPS traces and GTFS data**")

# Sidebar for configuration
st.sidebar.header("Configuration")

# File upload for GPS data only
st.sidebar.subheader("Upload GPS Data")
uploaded_gps = st.sidebar.file_uploader(
    "Upload GPS CSV File",
    type=['csv'],
    help="Upload your GPS tracking data in CSV format"
)

# Fixed paths for codebase data
LINES_FILE_PATH = "./data/lines_gdf.gpkg"
GTFS_DIR_PATH = "./data/gtfs"
SAMPLE_GPS_PATH = "./data/zg11_20250915.csv"

@st.cache_data
def load_lines_data(lines_path):
    """Load lines data from codebase"""
    try:
        return utils.load_data(lines_path)
    except Exception as e:
        st.error(f"Error loading lines data: {str(e)}")
        return None

@st.cache_data
def process_gps_data(gps_file=None):
    """Process GPS data from upload or sample file"""
    try:
        if gps_file is not None:
            # Process uploaded file
            return pd.read_csv(gps_file)
        else:
            # Use sample data
            return utils.load_gps_from_csv(SAMPLE_GPS_PATH)
    except Exception as e:
        st.error(f"Error processing GPS data: {str(e)}")
        return None

@st.cache_data
def create_map_data(_lines_gdf, _gps_gdf):
    """Process map data for Plotly visualization"""
    try:
        from shapely.geometry import LineString

        # Process lines data
        lines_data = []
        for i, row in enumerate(_lines_gdf.itertuples()):
            if hasattr(row, 'geometry') and row.geometry is not None:
                if isinstance(row.geometry, LineString):
                    geometries = [row.geometry]
                else:
                    geometries = row.geometry.geoms

                for geom in geometries:
                    coords = list(geom.coords)
                    lats, lons = zip(*[(y, x) for x, y in coords])
                    lines_data.append({
                        'line_id': getattr(row, 'line_id', f'Line {i}'),
                        'route_label': getattr(row, 'route_label', 'N/A'),
                        'lats': list(lats),
                        'lons': list(lons)
                    })

        # Sample GPS data for performance
        gps_sample = _gps_gdf.sample(min(200, len(_gps_gdf))) if len(_gps_gdf) > 200 else _gps_gdf

        return lines_data, gps_sample

    except Exception as e:
        st.error(f"Error processing map data: {str(e)}")
        return [], pd.DataFrame()

# Load data
data_loaded = False

if uploaded_gps is not None:
    st.sidebar.success(f"GPS file uploaded: {uploaded_gps.name}")
    with st.spinner("Loading data..."):
        lines_gdf = load_lines_data(LINES_FILE_PATH)
        gps_gdf = process_gps_data(uploaded_gps)

        if lines_gdf is not None and gps_gdf is not None:
            st.session_state.lines_gdf = lines_gdf
            st.session_state.gps_gdf = gps_gdf
            st.session_state.data_loaded = True
            data_loaded = True
            st.sidebar.success("Data loaded successfully!")
            st.sidebar.info(f"GPS points loaded: {len(gps_gdf)}")
            st.sidebar.info(f"Bus lines available: {len(lines_gdf)}")

elif st.sidebar.button("Load Sample Data"):
    with st.spinner("Loading sample data..."):
        lines_gdf = load_lines_data(LINES_FILE_PATH)
        gps_gdf = process_gps_data()

        if lines_gdf is not None and gps_gdf is not None:
            st.session_state.lines_gdf = lines_gdf
            st.session_state.gps_gdf = gps_gdf
            st.session_state.data_loaded = True
            data_loaded = True
            st.sidebar.success("Sample data loaded successfully!")
            st.sidebar.info(f"GPS points loaded: {len(gps_gdf)}")
            st.sidebar.info(f"Bus lines available: {len(lines_gdf)}")

# Check if data exists in session state
if 'data_loaded' in st.session_state and st.session_state.data_loaded:
    data_loaded = True
    lines_gdf = st.session_state.lines_gdf
    gps_gdf = st.session_state.gps_gdf

# Show instructions if no data loaded
if not data_loaded:
    st.info("Please upload your GPS CSV file or load sample data to get started.")

    st.subheader("Expected GPS CSV Format")
    st.markdown("""
    Your GPS CSV file should contain the following columns:
    - **timestamp**: Date and time of GPS reading
    - **lat**: Latitude coordinate
    - **lon**: Longitude coordinate
    - **speed_kmh**: Speed in kilometers per hour (optional)
    - **direction**: Direction/heading (optional)

    Example:
    ```
    timestamp,lat,lon,speed_kmh,direction
    2025-09-15 08:00:00,-25.9653,32.5832,45.2,180
    2025-09-15 08:01:00,-25.9663,32.5842,42.1,185
    ```
    """)
    st.stop()

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview & Map",
    "GPS Analysis",
    "Schedule Analysis",
    "Comparative Analysis"
])

# Use a single cached function for fetching routes, accessible by Tab 3 and Tab 4
@st.cache_data
def get_routes_for_line_cached(line_id):
    """Cached function to fetch routes for a given line ID."""
    return utils.get_routes_for_line(line_id)

# Get all available lines and sort them naturally
all_line_ids = lines_gdf['line_id'].unique().tolist()
all_line_ids = sorted(all_line_ids, key=lambda x: int(x[1:]))


# Tab 1: Overview & Map
with tab1:
    st.header("Data Overview & Map Visualization")

    # Interactive Map Section
    st.subheader("Interactive Map")

    lines_data, gps_sample = create_map_data(lines_gdf, gps_gdf)

    if lines_data:
        fig = go.Figure()

        # Add bus lines
        colors = px.colors.qualitative.Set3 + px.colors.qualitative.Dark24
        for i, line in enumerate(lines_data):
            color = colors[i % len(colors)]
            fig.add_trace(go.Scattermap(
                mode="lines",
                lon=line['lons'],
                lat=line['lats'],
                name=f"{line['line_id']}: {line['route_label']}",
                line=dict(width=3, color=color),
                hovertemplate=f"<b>{line['line_id']}</b><br>{line['route_label']}<extra></extra>"
            ))

        # Add GPS points
        if not gps_sample.empty:
            hover_text = []
            for _, row in gps_sample.iterrows():
                text = f"Time: {row.timestamp}<br>"
                if 'speed_kmh' in row and pd.notna(row.speed_kmh):
                    text += f"Speed: {row.speed_kmh} km/h<br>"
                if 'direction' in row and pd.notna(row.direction):
                    text += f"Direction: {row.direction}"
                hover_text.append(text)

            fig.add_trace(go.Scattermap(
                mode="markers",
                lon=gps_sample['lon'],
                lat=gps_sample['lat'],
                name="GPS Points",
                marker=dict(size=5, color='red'),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>"
            ))

        fig.update_layout(
            map_style="carto-positron",
            map=dict(
                center=dict(lat=-25.9653, lon=32.5832),
                zoom=10
            ),
            height=500,
            margin=dict(l=0, r=0, t=0, b=25),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
            )
        )

        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        st.info(f"Showing {len(gps_sample)} GPS points (sampled from {len(gps_gdf)} total points)")

    st.divider()

    # Data Overview Section
    st.subheader("Data Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Bus Lines Data (Codebase)")
        st.write(f"Total lines: {len(lines_gdf)}")
        st.info("Lines data loaded from codebase")

        st.subheader("Available Bus Lines")
        if 'line_id' in lines_gdf.columns and 'route_label' in lines_gdf.columns:
            for _, row in lines_gdf.iterrows():
                st.write(f"**{row['line_id']}**: {row['route_label']}")

        if st.checkbox("Show lines data preview", key="preview_lines"):
            st.dataframe(lines_gdf.head(17))

    with col2:
        data_source = "(Uploaded)" if uploaded_gps else "(Sample)"
        st.subheader(f"GPS Data {data_source}")
        st.write(f"Total GPS points: {len(gps_gdf)}")

        if uploaded_gps:
            st.success(f"Using uploaded file: {uploaded_gps.name}")
        else:
            st.info("Using sample GPS data from codebase")

        st.subheader("GPS Statistics")
        if 'speed_kmh' in gps_gdf.columns:
            st.metric("Average Speed", f"{gps_gdf['speed_kmh'].mean():.1f} km/h")
            st.metric("Max Speed", f"{gps_gdf['speed_kmh'].max():.1f} km/h")

        if 'timestamp' in gps_gdf.columns:
            st.write("**Time Range:**")
            st.write(f"From: {gps_gdf['timestamp'].min()}")
            st.write(f"To: {gps_gdf['timestamp'].max()}")

        if st.checkbox("Show GPS data preview", key="preview_gps"):
            st.dataframe(gps_gdf.head())

# Tab 2: GPS Analysis
with tab2:
    st.header("GPS Analysis")

    selected_line = st.selectbox("Select a Line", all_line_ids, index=4, key="gps_line_select")

    col1, col2 = st.columns(2)

    with col1:
        flip_origin = st.checkbox("Flip Origin", value=False, key="flip_origin_gps")
        analyze_button = st.button("Analyze GPS Data", type="primary")

    with col2:
        show_plots = st.checkbox("Show Analysis Plots", value=True)

    if analyze_button:
        with st.spinner("Analyzing GPS data..."):
            try:
                # Capture print output from utils function
                f = io.StringIO()
                with redirect_stdout(f):
                    gps_result_gdf, ref_line = utils.calculate_line_distance(
                        gps_gdf, lines_gdf, line_id=selected_line,
                        plot=show_plots, flip_origin=flip_origin
                    )

                output = f.getvalue()

                # Store results for other tabs
                st.session_state.gps_result_gdf = gps_result_gdf
                st.session_state.ref_line = ref_line
                st.session_state.selected_line = selected_line

                st.success(f"GPS analysis completed for line {selected_line}")

                if output:
                    st.subheader("Analysis Output")
                    st.text(output)

                if show_plots:
                    plt.ioff()
                    st.subheader("Distance Analysis Visualization")
                    st.pyplot(plt.gcf())
                    plt.close()

                    st.subheader("Detailed Analysis")
                    utils.plot_distance_analysis(gps_result_gdf, line_id=selected_line)
                    st.pyplot(plt.gcf())
                    plt.close()

                if 'distance_along_line_km' in gps_result_gdf.columns:
                    st.subheader("Key Statistics")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Min Distance", f"{gps_result_gdf['distance_along_line_km'].min():.2f} km")
                    with col_b:
                        st.metric("Max Distance", f"{gps_result_gdf['distance_along_line_km'].max():.2f} km")
                    with col_c:
                        if 'offset_distance_m' in gps_result_gdf.columns:
                            st.metric("Avg Offset", f"{gps_result_gdf['offset_distance_m'].mean():.1f} m")

                st.subheader("Analysis Results")
                st.write(f"Processed {len(gps_result_gdf)} GPS points")

                if st.checkbox("Show processed GPS data", key="preview_gps_processed"):
                    st.dataframe(gps_result_gdf.head(10))

            except Exception as e:
                st.error(f"Error during GPS analysis: {str(e)}")

    elif 'gps_result_gdf' in st.session_state:
        st.info("Previous GPS analysis results are available. Click 'Analyze GPS Data' to run a new analysis.")

# Tab 3: Schedule Analysis
with tab3:
    st.header("Schedule Analysis")

    selected_line_sched = st.selectbox("1. Select a Line", all_line_ids, index=4, key="sched_line_select")

    selected_route_ids_sched = []
    if selected_line_sched:
        try:
            available_routes_df = get_routes_for_line_cached(selected_line_sched)
            route_options = available_routes_df['route_id'].tolist()

            if not route_options:
                st.warning(f"No routes found for Line **{selected_line_sched}** in the GTFS data.")
            else:
                selected_route_ids_sched = st.multiselect(
                    "2. Select Routes for this Line",
                    options=route_options,
                    default=route_options,
                    key="sched_route_select"
                )
        except Exception as e:
            st.error(f"Could not fetch routes for Line {selected_line_sched}: {e}")
    
    st.subheader("Options")
    col1, col2 = st.columns(2)
    with col1:
        flip_origin_sched = st.checkbox("Flip Origin", value=False, key="flip_origin_sched")
    with col2:
        show_plots_sched = st.checkbox("Show Analysis Plots", value=True, key="show_plots_sched")
    
    st.divider()

    if st.button("Analyze Schedule Data", type="primary"):
        if not selected_line_sched or not selected_route_ids_sched:
            st.warning("Please select a line and at least one route to proceed.")
        else:
            with st.spinner("Processing schedule data..."):
                try:
                    scheduled_trips_gdf, ref_line = utils.create_scheduled_trips_from_gtfs(
                        route_ids=selected_route_ids_sched,
                        gtfs_dir=GTFS_DIR_PATH,
                        lines_gdf=lines_gdf,
                        line_id=selected_line_sched,
                        plot=show_plots_sched,
                        flip_origin=flip_origin_sched
                    )
                    
                    # Store results for other tabs
                    st.session_state.scheduled_trips_gdf = scheduled_trips_gdf
                    st.session_state.schedule_ref_line = ref_line
                    st.session_state.selected_line_sched = selected_line_sched

                    st.success(f"Schedule analysis completed for routes: {', '.join(selected_route_ids_sched)}")

                    if show_plots_sched:
                        plt.ioff()
                        st.subheader("Schedule Visualization")
                        st.pyplot(plt.gcf())
                        plt.close()

                    st.subheader("Schedule Results")
                    st.write(f"Processed {len(scheduled_trips_gdf)} scheduled stops.")

                    if st.checkbox("Show schedule data preview", key="preview_sched"):
                        st.dataframe(scheduled_trips_gdf.head())

                except Exception as e:
                    st.error(f"Error during schedule analysis: {str(e)}")

# Tab 4: Comparative Analysis
with tab4:
    st.header("On-Demand Comparative Analysis")
    st.info("Select a line and its corresponding routes to generate a comparison plot directly.")

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_line = st.selectbox("1. Select a Line", all_line_ids, index=4, key="compare_line_select")
    
    selected_route_ids = []
    if selected_line:
        try:
            available_routes_df = get_routes_for_line_cached(selected_line)
            route_options = available_routes_df['route_id'].tolist()

            if not route_options:
                st.warning(f"No routes found for Line **{selected_line}** in the GTFS data.")
            else:
                with col1:
                    selected_route_ids = st.multiselect(
                        "2. Select Routes for this Line",
                        options=route_options,
                        default=route_options,
                        key="compare_route_select"
                    )
        except Exception as e:
            st.error(f"Could not fetch routes for Line {selected_line}: {e}")

    with col2:
        st.write("<br>", unsafe_allow_html=True) 
        flip_origin_checkbox = st.checkbox("Flip Origin", value=False, key="flip_origin_comparison")

    st.divider()

    if st.button("Generate Comparison Plot", type="primary", key="generate_comparison"):
        if not selected_line or not selected_route_ids:
            st.warning("Please select a line and at least one route to proceed.")
        else:
            with st.spinner(f"Running full analysis for Line **{selected_line}**..."):
                try:
                    st.write("Step 1/3: Processing GPS data...")
                    gps_result_gdf, _ = utils.calculate_line_distance(
                        gps_gdf, lines_gdf, line_id=selected_line, plot=False,
                        flip_origin=flip_origin_checkbox
                    )

                    st.write("Step 2/3: Processing schedule data...")
                    scheduled_trips_gdf, _ = utils.create_scheduled_trips_from_gtfs(
                        route_ids=selected_route_ids,
                        gtfs_dir=GTFS_DIR_PATH,
                        lines_gdf=lines_gdf,
                        line_id=selected_line,
                        plot=False,
                        flip_origin=flip_origin_checkbox
                    )

                    st.write("Step 3/3: Generating comparison plot...")
                    if gps_result_gdf.empty or scheduled_trips_gdf.empty:
                        st.error("Analysis complete, but no overlapping data was found. The GPS traces may not cover the selected line, or no scheduled trips exist for the given routes.")
                    else:
                        plt.ioff()
                        utils.plot_gps_vs_scheduled(
                            gps_result_gdf,
                            scheduled_trips_gdf,
                            gps_time_col='timestamp',
                            sched_time_col='arrival_time'
                        )
                        st.pyplot(plt.gcf())
                        plt.close()
                        st.success(f"Successfully generated comparison for Line **{selected_line}**.")

                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")

# Footer
st.divider()
st.markdown("**METRAP Bus Operations Analysis** - Built with Streamlit")

# Sidebar utilities
st.sidebar.divider()
st.sidebar.subheader("Utilities")

if st.sidebar.button("Create Outputs Directory"):
    import os
    os.makedirs("./outputs", exist_ok=True)
    st.sidebar.success("Outputs directory created!")

if st.sidebar.button("Clear Session Data"):
    for key in list(st.session_state.keys()):
        if key.startswith(('gps_', 'scheduled_', 'lines_', 'data_loaded', 'available_routes', 'ref_line', 'selected_')):
            del st.session_state[key]
    st.sidebar.success("Session data cleared!")
    st.rerun()