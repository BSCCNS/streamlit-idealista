from streamlit_idealista.config import   INPUT_DATA_PATH, INPUT_OPERATION_TYPES_PATH, INPUT_TYPOLOGY_TYPES_PATH, INPUT_OPERATION_TYPES_PATH, INPUT_SUPERILLES_INTERVENTIONS_GEOJSON, INPUT_DTYPES_COUPLED_JSON_PATH, INPUT_INE_CENSUSTRACT_GEOJSON 
import functions as fc
import streamlit as st
import folium as folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from pathlib import Path
from shapely.geometry import GeometryCollection, shape
from shapely.ops import transform
from typing import Union,Optional,List
import numpy as np
import pandas as pd
from pathlib import Path
import json
import geopandas as gpd
import shapely
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyproj import Transformer
from PIL import Image

# Added this to avoid error when converting to eps 4326. Following
# https://stackoverflow.com/questions/78050786/why-does-geopandas-to-crs-give-inf-inf-the-first-time-and-correct-resul
import pyproj
pyproj.network.set_network_enabled(False)

im = Image.open("assets/favicon.png")
   
st.set_page_config(    
    
    page_title="Idealista Dashboard",
    page_icon= im,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://vCity.tech',
        'Report a bug': "https://vCity.tech",
        'About': "# This is a header."
    }
)

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400&display=swap" rel="stylesheet">
    <style>
    body {
        font-family: 'DM Sans', sans-serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Dashboard Title
st.title("Idealista Data Exploration Dashboard")

# Dashboard Description
st.markdown("""
This dashboard allows users to explore data from the **Idealista dataset** effectively. 

### How to Use:
1. **Draw on the Map**: Select the area of interest by drawing a polygon on the map.
2. **View Time Series**: Observe how rent and sale prices have changed over the years within your drawn area.
3. **Analyze Interventions**: Check for any superblock interventions that may affect the property values in your selected area.

This tool is designed to aid in decision-making and provide insights into the real estate landscape in the region.
""")

# load data
@st.cache_data
def load_dtypes(dtypes_path: Path) -> dict:
    with open(dtypes_path, 'r') as f:
        return json.load(f)
dtypes_coupled_dict = load_dtypes(INPUT_DTYPES_COUPLED_JSON_PATH)

@st.cache_data
def load_main_data(main_data_path: Path) -> pd.DataFrame:
    return pd.read_csv( main_data_path, sep = ';', dtype=dtypes_coupled_dict)
df = load_main_data(INPUT_DATA_PATH)

@st.cache_data
def load_censustract_geojson(censustract_geojson_path: Path) -> gpd.GeoDataFrame:
    gdf_ine = gpd.read_file(censustract_geojson_path)
    gdf_ine['CENSUSTRACT'] = gdf_ine['CENSUSTRACT'].astype(int).astype(str)
    return gdf_ine
gdf_ine = load_censustract_geojson(INPUT_INE_CENSUSTRACT_GEOJSON)

@st.cache_data
def load_operation_types(operation_types_path: Path) -> pd.DataFrame:
    return pd.read_csv(operation_types_path, sep=";", dtype=dtypes_coupled_dict)
operation_types_df = load_operation_types(INPUT_OPERATION_TYPES_PATH)

@st.cache_data
def load_typology_types(typology_types_path: Path) -> pd.DataFrame:
    return pd.read_csv( typology_types_path, sep=";", dtype=dtypes_coupled_dict)
typology_types_df = load_typology_types(INPUT_TYPOLOGY_TYPES_PATH)

@st.cache_data
def load_interventions(interventions_path: Path) -> gpd.GeoDataFrame:
    return gpd.read_file(interventions_path)
interventions_gdf =  load_interventions(INPUT_SUPERILLES_INTERVENTIONS_GEOJSON)

@st.cache_data
def process_df(df: pd.DataFrame) -> pd.DataFrame:
    copy_df = df.copy(deep=True)
    return (
        copy_df
        .astype({'ADOPERATIONID': 'int',
                'ADTYPOLOGYID': 'int'
                })
        .join(operation_types_df.set_index('ID'), on='ADOPERATIONID', how="left", validate="m:1")
        .rename(columns={
            'SHORTNAME': 'ADOPERATION',
                        }
                )
        .astype({'ADOPERATION': 'category',
                'ADOPERATIONID': 'category'
                })
        .drop(columns=("DESCRIPTION"))
        .join(typology_types_df.set_index('ID'), on='ADTYPOLOGYID', how="left", validate="m:1")
        .rename(columns={
            'SHORTNAME': 'ADTYPOLOGY',
                        }
                )
        .astype({'ADTYPOLOGY': 'category',
                'ADTYPOLOGYID': 'category'
                })
        .drop(columns=("DESCRIPTION"))
    )
processed_df = process_df(df)

# Streamlit App Logic
st.title("Map Drawing and Geometry Capture")

left, right = st.columns([1,1])  # You can adjust these numbers to your preference

interventions_gdf = interventions_gdf.to_crs('EPSG:4326')
gdf_ine = gdf_ine.to_crs("EPSG:4326")

# Initialize the toggle in session state
if 'put_new_map_boolean' not in st.session_state:
    st.session_state['put_new_map_boolean'] = False

if "drawn_geometries" not in st.session_state:
    st.session_state["drawn_geometries"] = []

with left:
    st.subheader("Map")
    geometry_selection = st.multiselect(
        "Select Urban Intervention", 
        options=list(interventions_gdf["TITOL_WO"].unique()),
        help="Select one or more geometries to filter data. Leave empty to use the drawn geometry."
    )

    # Create the base map
    m = folium.Map(location=[41.40463, 2.17924], zoom_start=13, tiles="cartodbpositron")


    draw = Draw(
        draw_options={
            'polyline': False,
            'polygon': True,
            'rectangle': False,
            'circle': False,
            'marker': True,
        },
        edit_options={'edit': True},
    )
    draw.add_to(m)
    
    geojson_layer = folium.FeatureGroup(name="Show Urban Interventions")
    
    # Filter interventions based on selection
    filtered_interventions_gdf = interventions_gdf[interventions_gdf["TITOL_WO"].isin(geometry_selection)].copy()
    
    # Compute impacted and district areas
    impacted_gdf = fc.get_impacted_gdf(filtered_interventions_gdf, gdf_ine)
    filtered_interventions_gdf['md'] = filtered_interventions_gdf['CENSUSTRACT'].astype(str).str[0:7].astype(int)
    gdf_ine['md'] = gdf_ine['CENSUSTRACT'].astype(str).str[0:6].astype(int)
    district_gdf = gdf_ine[gdf_ine['md'].isin(filtered_interventions_gdf['md']) & 
                           ~gdf_ine['CENSUSTRACT'].astype(int).isin(filtered_interventions_gdf['CENSUSTRACT'].astype(str).astype(int))]
    
    if st.session_state["drawn_geometries"]:

        for geom in st.session_state["drawn_geometries"]:
            folium.GeoJson(
                geom,
                style_function=lambda x: {
                    "fillColor": "green",
                    "color": "green",
                    "weight": 1,
                    "fillOpacity": 0.3,
                },
            ).add_to(m)
        

    geometry_collection = fc.GeometryCollection(st.session_state["drawn_geometries"])

    # Convert drawn geometries to GeoDataFrame with the correct CRS
    geometry_gdf = gpd.GeoDataFrame(
        {'geometry': [geometry_collection]},
        crs="EPSG:25830"  # Adjust if needed
    )
    geometry_gdf = geometry_gdf.to_crs(gdf_ine.crs)

    # Get impacted census tracts
    my_censustracts = fc.get_impacted_gdf(geometry_gdf, gdf_ine)

    for _, row in my_censustracts.iterrows():
        folium.GeoJson(
            row["geometry"],
            style_function=lambda x: {
                "fillColor": "green",
                "color": "green",
                "weight": 1,
                "fillOpacity": 0.3,
            },
        ).add_to(geojson_layer)    

    # Add geometries to the map
    for _, row in interventions_gdf.iterrows():
        folium.GeoJson(
            row["geometry"],
            name=row["TITOL_WO"],
            tooltip=row["TITOL_WO"],
            style_function=lambda x: {
                "fillColor": "grey",
                "color": "grey",
                "weight": 1,
                "fillOpacity": 0.3,
            },
        ).add_to(geojson_layer)

    for _, row in filtered_interventions_gdf.iterrows():
        folium.GeoJson(
            row["geometry"],
            name=row["TITOL_WO"],
            tooltip=row["TITOL_WO"],
            style_function=lambda x: {
                "fillColor": "red",
                "color": "red",
                "weight": 2,
                "fillOpacity": 0.6,
            },
        ).add_to(geojson_layer)

    for _, row in impacted_gdf.iterrows():
        folium.GeoJson(
            row["geometry"],
            style_function=lambda x: {
                "fillColor": "blue",
                "color": "blue",
                "weight": 1,
                "fillOpacity": 0.4,
            },
        ).add_to(geojson_layer)

    geojson_layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    folium.plugins.Fullscreen(
        position="bottomleft",
        title="Expand me",
        title_cancel="Exit me",
        force_separate_button=True,
    ).add_to(m)

    # Display the appropriate map based on session state
    if st.session_state['put_new_map_boolean']:
        output = st_folium(m, width=600, height=500, key='new_map')
    else:
        output = st_folium(m, width=600, height=500, key='old_map')

    # Process and display the drawn geometries
    geometry_collection = None  # Define a default value
    if output and output["all_drawings"]:
        drawn_geometries = [fc.transform_geometry(geo_json['geometry']) for geo_json in output["all_drawings"]]
        geometry_collection = fc.GeometryCollection(drawn_geometries)

        st.write("Captured Geometries in UTM (EPSG:25831):")
        st.write(geometry_collection)

        # Convert drawn geometries to GeoDataFrame with the correct CRS
        geometry_gdf = gpd.GeoDataFrame(
            {'geometry': [geometry_collection]},
            crs="EPSG:25830"  # Adjust if needed
        )
        geometry_gdf = geometry_gdf.to_crs(gdf_ine.crs)

        # Get impacted census tracts
        my_censustracts = fc.get_impacted_gdf(geometry_gdf, gdf_ine)

        st.session_state["drawn_geometries"] = drawn_geometries
        
        st.session_state['put_new_map_boolean'] = not st.session_state['put_new_map_boolean']  # Toggle the map state
        st.rerun()  # Force rerun to refresh with the new map

    else:
        st.warning("No geometry has been drawn, so no census tracts can be impacted.")
        my_censustracts = []
    
with right:
    # Price type filter
    price_type = 'Both'

    try:
        if st.session_state["drawn_geometries"]:

            geometry_collection = fc.GeometryCollection(st.session_state["drawn_geometries"])
            geometry_gdf = gpd.GeoDataFrame(
                {'geometry': [geometry_collection]},
                crs="EPSG:25830"  # Assuming gdf_ine has the correct CRS (EPSG:4326)
            )

            geometry_gdf = geometry_gdf.to_crs(gdf_ine.crs)
            my_censustracts = fc.get_impacted_gdf(geometry_gdf, gdf_ine)


            processed_df['CENSUSTRACT'] = processed_df['CENSUSTRACT'].astype(str)
            my_censustracts['CENSUSTRACT'] = my_censustracts['CENSUSTRACT'].astype(str)

            # Apply zfill and filtering
            control_gdf = processed_df[
                processed_df['CENSUSTRACT'].str.zfill(10).isin(
                    my_censustracts['CENSUSTRACT'].str.zfill(10)
                )
            ]

            # Use the geometry drawn on the map
            chart = fc.plot_timeseries(
                processed_df,
                interventions_gdf, 
                impacted_gdf,
                gdf_ine,
                price_type=price_type.lower(),
                district = False,
                district_gdf = district_gdf,
                control_polygon = True,
                control_gdf =  control_gdf

            )

            # Display the chart if available
            if chart is not None:
                st.plotly_chart(chart, use_container_width=True)

        else:
            # Use the geometry drawn on the map
            chart = fc.plot_timeseries(
                processed_df,
                interventions_gdf, 
                impacted_gdf,
                gdf_ine,
                price_type=price_type.lower(),
                district = False,
                district_gdf = district_gdf,
            )

            # Display the chart if available
            if chart is not None:
                st.plotly_chart(chart, use_container_width=True)

    except Exception as e:
        # Silently pass or log the error if needed
        st.error(f"An error occurred: {e}")