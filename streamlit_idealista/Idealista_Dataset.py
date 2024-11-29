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

im = Image.open("assets/favicon.png")
   
st.set_page_config(    
    
    page_title="Idealista Dashboard",
    page_icon= im,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://vCity.tech',
        'Report a bug': "https://vCity.tech",
        'About': "# This is a header. This is an *extremely* cool app!"
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


with open(INPUT_DTYPES_COUPLED_JSON_PATH, 'r') as f:
    dtypes_coupled_dict = json.load(f)

# load data
df = pd.read_csv( INPUT_DATA_PATH, sep = ';', dtype=dtypes_coupled_dict)
gdf_ine = gpd.read_file(INPUT_INE_CENSUSTRACT_GEOJSON)
gdf_ine['CENSUSTRACT'] = gdf_ine['CENSUSTRACT'].astype(int).astype(str)

operation_types_df = pd.read_csv( INPUT_OPERATION_TYPES_PATH, sep=";", dtype=dtypes_coupled_dict)
typology_types_df = pd.read_csv( INPUT_TYPOLOGY_TYPES_PATH, sep=";", dtype=dtypes_coupled_dict)

interventions_gdf =  gpd.read_file( INPUT_SUPERILLES_INTERVENTIONS_GEOJSON)

processed_df = (
    df
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
#st.write(df)
# Streamlit App Logic
st.title("Map Drawing and Geometry Capture")

left, right = st.columns([1,1])  # You can adjust these numbers to your preference

with left:
    # First container for the Folium map
    st.subheader("Map")
    
    # Create and display the map
    m = folium.Map(location=[41.40463, 2.17924], zoom_start=13, tiles='cartodbpositron')
    Draw().add_to(m)

    folium.plugins.Fullscreen(
        position="bottomleft",
        title="Expand me",
        title_cancel="Exit me",
        force_separate_button=True,
    ).add_to(m)
    
    output = st_folium(m, width=600,height = 500)  # Adjust the width if necessary

    # Process and display the drawn geometries
    geometry_collection = None  # Define a default value
    if output and output["all_drawings"]:
        drawn_geometries = [fc.transform_geometry(geo_json['geometry']) for geo_json in output["all_drawings"]]
        geometry_collection = fc.GeometryCollection(drawn_geometries)
        st.write("Captured Geometries in UTM (EPSG:25831):")
        st.write(geometry_collection)
    else:
        st.write("No geometries drawn yet.")

    # Handle census tracts based on drawn geometries
    if geometry_collection:
        my_censustracts = fc.get_impacted_censustracts(geometry_collection, gdf_ine)
    else:
        st.warning("No geometry has been drawn, so no census tracts can be impacted.")
        my_censustracts = []

with right:

    # Price type filter
    price_type = st.radio("Price Type", ['Both', 'Sale', 'Rent'], horizontal=True)

    try: 
    # Create and display the chart
        chart = fc.plot_timeseries(
            processed_df, 
            interventions_gdf, 
            my_censustracts,
            price_type=price_type.lower()
        )
        if chart is not None:
            st.plotly_chart(chart, use_container_width=True)

    except Exception as e:
    # Silently pass or log the error if needed
        pass
