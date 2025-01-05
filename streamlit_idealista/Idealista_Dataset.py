import json
from pathlib import Path
from typing import List, Optional, Union

import folium as folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shapely
import streamlit as st
from folium.plugins import Draw
from PIL import Image
from plotly.subplots import make_subplots
from prophet import Prophet
from pyproj import Transformer
from shapely.geometry import GeometryCollection, shape
from shapely.ops import transform
from streamlit_folium import st_folium
from upath import UPath

from streamlit_idealista.config import (
    INPUT_DATA_PATH,
    INPUT_DTYPES_COUPLED_JSON_PATH,
    INPUT_INE_CENSUSTRACT_GEOJSON,
    INPUT_OPERATION_TYPES_PATH,
    INPUT_SUPERILLES_INTERVENTIONS_GEOJSON,
    INPUT_TYPOLOGY_TYPES_PATH,
    PROJ_ROOT,
)

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

# load data
# See: https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data
@st.cache_data
def load_dtypes(_dtypes_path: UPath) -> dict:
    with _dtypes_path.open("rb") as f:
        return json.load(f)
dtypes_coupled_dict = load_dtypes(INPUT_DTYPES_COUPLED_JSON_PATH)

@st.cache_data
def load_main_data(_main_data_path: UPath) -> pd.DataFrame:
    with _main_data_path.open("rb") as f:
        df = pd.read_csv(f, sep = ';', dtype=dtypes_coupled_dict, encoding="unicode_escape")
    return df
df = load_main_data(INPUT_DATA_PATH)

@st.cache_data
def load_censustract_geojson(_censustract_geojson_path: UPath) -> gpd.GeoDataFrame:
    with _censustract_geojson_path.open("rb") as f:
        gdf_ine = gpd.read_file(f)
    gdf_ine['CENSUSTRACT'] = gdf_ine['CENSUSTRACT'].astype(int).astype(str)
    return gdf_ine
gdf_ine = load_censustract_geojson(INPUT_INE_CENSUSTRACT_GEOJSON)

@st.cache_data
def load_operation_types(_operation_types_path: UPath) -> pd.DataFrame:
    with _operation_types_path.open("rb") as f:
        return pd.read_csv(f, sep=";", dtype=dtypes_coupled_dict, encoding="unicode_escape")
operation_types_df = load_operation_types(INPUT_OPERATION_TYPES_PATH)

@st.cache_data
def load_typology_types(_typology_types_path: UPath) -> pd.DataFrame:
    with _typology_types_path.open("rb") as f:
        return pd.read_csv(f, sep=";", dtype=dtypes_coupled_dict)
typology_types_df = load_typology_types(INPUT_TYPOLOGY_TYPES_PATH)

@st.cache_data
def load_interventions(_interventions_path: UPath) -> gpd.GeoDataFrame:
    with _interventions_path.open("rb") as f:
        return gpd.read_file(f)
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
            price_type=price_type.lower(),
            district = False
        )
        if chart is not None:
            st.plotly_chart(chart, use_container_width=True)

    except Exception as e:
    # Silently pass or log the error if needed
        pass
