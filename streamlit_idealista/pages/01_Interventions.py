from streamlit_idealista.config import   INPUT_DATA_PATH, INPUT_OPERATION_TYPES_PATH, INPUT_TYPOLOGY_TYPES_PATH, INPUT_OPERATION_TYPES_PATH, INPUT_SUPERILLES_INTERVENTIONS_GEOJSON, INPUT_DTYPES_COUPLED_JSON_PATH, INPUT_INE_CENSUSTRACT_GEOJSON, SALE_COLOR, RENT_COLOR, CONTROL_COLOR, INTERVENTION_COLOR, INTERSECT_COLOR, CONTROL_SALE
import functions as fc
from upath import UPath

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
    
    page_title="Select an Intervention and Compare it with the District",
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

# Dashboard Description
#st.description('Explore the effects of a selected urban intervention by comparing its impact on housing prices with the overall trends in the district. Visualize and analyze differences over time to assess intervention outcomes.')

# load data
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

# Streamlit App Logic
st.title("Select an Intervention and Compare it with the District")

geometry_selection = st.multiselect(
    "Select Urban Intervention", 
    options=list(interventions_gdf["TITOL_WO"].unique()),  # Replace 'TITOL_WO' with the column containing geometry names
    help="Select one or more geometries to filter data. Leave empty to use the drawn geometry."
)

left, right = st.columns([1,1])  # You can adjust these numbers to your preference

interventions_gdf = interventions_gdf.to_crs('EPSG:4326')
gdf_ine = gdf_ine.to_crs("EPSG:4326")



with left:
    st.subheader("Map")


    # Create the base map
    m = folium.Map(location=[41.40463, 2.17924], zoom_start=13, tiles="cartodbpositron")

    # Create a GeoJSON layer for all geometries
    geojson_layer = folium.FeatureGroup(name="Show Urban Interventions")

    # selected interventions
    filtered_interventions_gdf = interventions_gdf[interventions_gdf["TITOL_WO"].isin(geometry_selection)].copy()

    # impacted area
    impacted_gdf = fc.get_impacted_gdf(filtered_interventions_gdf, gdf_ine) 

    # district of the selected intervention
    filtered_interventions_gdf['md'] = filtered_interventions_gdf['CENSUSTRACT'].astype(str).str[0:7].astype(int)
    gdf_ine['md'] = gdf_ine['CENSUSTRACT'].astype(str).str[0:6].astype(int)

    # to use district as control group, remove intervened censustracts
    c1 = gdf_ine['md'].isin(filtered_interventions_gdf['md'])
    c2 = ~gdf_ine['CENSUSTRACT'].astype(int).isin(filtered_interventions_gdf['CENSUSTRACT'].astype(str).astype(int))
    district_gdf = gdf_ine[c1 & c2]

    # Add geometries to the layer
    for _, row in interventions_gdf.iterrows():
        folium.GeoJson(
            row["geometry"],
            name=row["TITOL_WO"],
            tooltip = row["TITOL_WO"],
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
            tooltip = row["TITOL_WO"],
            style_function=lambda x: {
                "fillColor": INTERVENTION_COLOR,
                "color": INTERVENTION_COLOR,
                "weight": 2,
                "fillOpacity": 0.6,
            },
        ).add_to(geojson_layer)

    for _, row in impacted_gdf.iterrows():
        folium.GeoJson(
            row["geometry"],
            style_function=lambda x: {
                "fillColor": INTERSECT_COLOR,
                "color": INTERSECT_COLOR,
                "weight": 1,
                "fillOpacity": 0.4,
            },
        ).add_to(geojson_layer)

    for _, row in district_gdf.iterrows():
        folium.GeoJson(
            row["geometry"],
            style_function=lambda x: {
                "fillColor": CONTROL_COLOR,
                "color": CONTROL_COLOR,
                "weight": 1,
                "fillOpacity": 0.3,
            },
        ).add_to(geojson_layer)

    # Add the GeoJSON layer to the map
    geojson_layer.add_to(m)

    # Add layer control to toggle visibility of geometries
    
    # Initialize the Draw plugin
    draw = Draw(
        draw_options={
            'polyline': False,
            'polygon': True,
            'rectangle': False,
            'circle': False,
            'marker': True,
            'color': CONTROL_COLOR

        },
        edit_options={'edit': True},
    )
    draw.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    folium.plugins.Fullscreen(
        position="bottomleft",
        title="Expand me",
        title_cancel="Exit me",
        force_separate_button=True,
    ).add_to(m)
    # Display the map in the Streamlit app
    output = st_folium(m, width=600, height=500)

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

    st.subheader("Time Series")

    # Price type filter
    price_type = 'Both'

    try:
        if geometry_selection: 
            if filtered_interventions_gdf.empty:
                print("No matching interventions found for the selected geometries.")
                my_censustracts = []  # No impacted census tracts
            else:
                chart = fc.plot_timeseries(
                    processed_df,
                    interventions_gdf, 
                    impacted_gdf,
                    gdf_ine,
                    price_type=price_type.lower(),
                    district = True,
                    district_gdf = district_gdf,
                    SALE_COLOR = SALE_COLOR,
                    RENT_COLOR = RENT_COLOR,
                    CONTROL_SALE = CONTROL_SALE, 
                    CONTROL_COLOR = CONTROL_COLOR, 
                    INTERVENTION_COLOR = INTERVENTION_COLOR
                )

                if chart is not None:
                    st.plotly_chart(chart, use_container_width=True)
                if not my_censustracts:
                    print("No census tracts impacted by the selected geometries.")
        else:
            # Use the geometry drawn on the map
            chart = fc.plot_timeseries(
                processed_df,
                interventions_gdf, 
                impacted_gdf,
                gdf_ine,
                price_type=price_type.lower(),
                district = True,
                district_gdf = district_gdf,
                SALE_COLOR = SALE_COLOR,
                RENT_COLOR = RENT_COLOR,
                CONTROL_SALE = CONTROL_SALE, 
                CONTROL_COLOR = CONTROL_COLOR, 
                INTERVENTION_COLOR = INTERVENTION_COLOR
            )

            # Display the chart if available
            if chart is not None:
                st.plotly_chart(chart, use_container_width=True)

    except Exception as e:
        # Silently pass or log the error if needed
        st.error(f"An error occurred: {e}")