
import datetime
import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import folium as folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shapely
import streamlit as st
from folium.plugins import Draw
from plotly.subplots import make_subplots
from prophet import Prophet
from pyproj import Transformer
from shapely.geometry import GeometryCollection, shape
from shapely.ops import transform
from streamlit_folium import st_folium



def transform_geometry(geometry):
    """
    Transform geometry from EPSG:4326 to EPSG:25831.

    Args:
      geometry (dict): Geometry in GeoJSON format.

    Returns:
      shapely.geometry.base.BaseGeometry: Transformed geometry.
    """
    # Convert GeoJSON to Shapely geometry
    geom = shape(geometry)

    # Create a transformer instance to convert from EPSG:4326 to EPSG:25831
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:25830", always_xy=True)

    # Transform the geometry using the transformer instance
    transformed_geom = transform(lambda x, y: transformer.transform(x, y), geom)

    return transformed_geom


def get_impacted_gdf(my_gdf: Union[gpd.GeoDataFrame, None],
                    ine_gdf: gpd.GeoDataFrame
                    ) -> Optional[List[str]]:
    """
    Get the impacted censustracts.

    Args:
      my_gdf (gpd.GeoDataFrame): gdf containing the areas to check in its geometry.
      ine_gdf (gpd.GeoDataFrame): Geopandas with INE information about
      censustracts and their polygons.

    Returns:
      Optional[List[str]]: The impacted censustracts.
    """
    if my_gdf is None:
        return None
    
    mask = ine_gdf['geometry'].intersects(my_gdf["geometry"].union_all())
    impacted_gdf = ine_gdf[mask].copy()

    return impacted_gdf

def get_impacted_censustracts(geometries: Union[shapely.geometry.GeometryCollection, None],
                              ine_gdf: gpd.GeoDataFrame
                               ) -> Optional[List[str]]:
    """
    Get the impacted censustracts.

    Args:
      geometries (shapely.geometry.GeometryCollection): The areas to check.
      ine_gdf (gpd.GeoDataFrame): Geopandas with INE information about
      censustracts and their polygons.

    Returns:
      Optional[List[str]]: The impacted censustracts.
    """
    if geometries is None:
        return None

    mask = ine_gdf['geometry'].intersects(geometries)
    return ine_gdf[mask]['CENSUSTRACT'].unique().tolist()

def get_timeseries_of_census_tracts(df: pd.DataFrame, censustract_list: Optional[List[str]] = None, operation: str = "mean") -> Optional[pd.DataFrame]:
    """
    Get the timeseries of prices (rent, sale) for the given census tracts.
    If more than one census tract, the mean or other specified operation is taken.

    Args:
      df (pd.DataFrame): The dataframe containing the data.
      censustract_list (Optional[List[str]]): The list of census tracts to filter.
      operation (str): Aggregation operation (mean, median).

    Returns:
      Optional[pd.DataFrame]: The timeseries for the given census tracts,
        applying the specified aggregation operation.
        If censustract_list is None, returns None.
    """
    if censustract_list is None:
        return None

    # Check if the operation is valid
    if operation not in ["mean", "median"]:
        raise ValueError("Operation must be 'mean' or 'median'")

    # Filter the dataframe for the given census tracts
    filtered_df = df[df["CENSUSTRACT"].isin(censustract_list)]

    # Define the aggregation methods based on the requested statistics
    # Group by 'PERIOD' and 'ADOPERATION', then apply the aggregation methods
    aggregated_df = (
        filtered_df
        .groupby(["PERIOD", "ADOPERATION"], observed=False)
        .agg({"UNITPRICE_ASKING": operation})
        .reset_index()  # Reset index to flatten the dataframe
        .pivot(index="PERIOD", columns="ADOPERATION", values="UNITPRICE_ASKING")  # Pivot on 'ADOPERATION'
    )

    #print("Resultado dentro de get_timeseries_of_census_tracts:", aggregated_df)

    return aggregated_df

def get_trend_of_timeseries(series: pd.Series) -> pd.Series:
    """
    Get the trend of a time series.

    Args:
      series (pd.Series): The time series.

    Returns:
      pd.Series: The trend of the time series.
    """
    if not isinstance(series, pd.Series):
        raise ValueError("Input must be a pandas Series")

    # Ensure the Series has a name for the output
    if series.name is None:
        series.name = "trend"

    # Prepare the data for Prophet
    series_copy = series.reset_index()
    series_copy.columns = ["ds", "y"]

    # Initialize and fit the Prophet model
    trend = Prophet()
    trend.fit(series_copy)

    # Create a future DataFrame and make predictions
    future = trend.make_future_dataframe(periods=0)
    forecast = trend.predict(future)

    # Extract the trend component and convert it to a Series
    trend_series = forecast[["ds", "trend"]].rename(columns={"ds": "PERIOD", "trend": series.name}).set_index("PERIOD")

    # Flatten the trend values and create a new Series
    trend_series_flat = pd.Series(trend_series[series.name].values, index=trend_series.index, name=series.name)

    return trend_series_flat

def merge_intervals(intervals):
    """Merge overlapping intervals and return merged intervals with associated interventions."""
    if not intervals:
        return []

    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])
    merged = []
    current_start, current_end, interventions = intervals[0][0], intervals[0][1], intervals[0][2]

    for start, end, intervention in intervals[1:]:
        if start <= current_end:  # Overlapping intervals
            current_end = max(current_end, end)
            interventions.update(intervention)
        else:
            merged.append((current_start, current_end, interventions))
            current_start, current_end, interventions = start, end, intervention

    merged.append((current_start, current_end, interventions))
    return merged

def add_geometry_layer(gdf, geojson_layer, style_dict = None):
    for _, row in gdf.iterrows():
        folium.GeoJson(
            row["geometry"],
            name=row["TITOL_WO"],
            tooltip = row["TITOL_WO"],
            # style_function=lambda x: {
            #     "fillColor": "blue",
            #     "color": "blue",
            #     "weight": 2,
            #     "fillOpacity": 0.6,
            # },
            style_function=lambda x: style_dict,
        ).add_to(geojson_layer)
    

def plot_timeseries(df: pd.DataFrame,
                    interventions_gdf: gpd.GeoDataFrame,
                    impacted_gdf: gpd.GeoDataFrame,
                    ine_gdf: gpd.GeoDataFrame,
                    include_trends: bool=True,
                    price_type: str = 'both',
                    district: bool =True,
                    district_gdf: gpd.GeoDataFrame = None,
                    control_polygon: bool = False,
                    control_gdf: gpd.GeoDataFrame = None,
                    SALE_COLOR: str =  '#FBBC05',
                    RENT_COLOR: str =  '#45B905',
                    CONTROL_SALE: str = '#626262',
                    CONTROL_COLOR: str = '#4D779E',
                    INTERVENTION_COLOR: str = '#EE8A82'

                    ) -> go.Figure:
    
    """
    Plot the timeseries of prices (rent, sale) for the given census tracts.
    If more than one census tract, the mean is taken.
    If include_trends is True, the trends are also plotted.

    Args:
      df (pd.DataFrame): The processed dataframe from idealista dataset 02 metricas de mercado.
      interventions_gdf (gpd.GeoDataFrame): The information about interventions.
      censustract_list (Union[List[str], None]): The list of census tracts.
      include_trends (bool): Whether to include the trends.

    Returns:
      go.Figure: The figure.

    """

    interventions_dict = {'Urbanització c. Almogàvers (Badajoz -Roc Boronat).': 'Almogàvers',
                      'Superilla de Poblenou': 'Superilla Poblenou',
                      'Eixos Verds Eixample ': 'Eixample',
                      'Supermanzana EJE VERDE (Consell de cent, Rocafort, Conde Borrell y Girona)': 'Superilla EixVerd',
                      'Eixos Verds LOT 4: Consell de Cent (Aribau - Rambla Catalunya)': 'LOT 4',
                      'Eje verde de la calle de Girona entre la calle de la Diputació y la Gran Via de les Corts Catalanes': 'EixVerd Girona',
                      'Urbanización de tramos de las calles de Puigcerdà, Cristòbal de Moura i Veneçuela ': 'Urbanització PCV',
                      'Supermanzana EJE VERDE Consejo de Ciento con Rocafort, Conde Borrell, Enric Granados y Girona': 'Consell de Cent',
                      'Eixos Verds LOT 2: Borrell (Aragó - Diputació) + Consell de Cent (Calàbria - Urgell) + Cruïlla': 'LOT 2',
                      'Eixos Verds LOT 1: ': 'LOT 1',
                      'Eix verd Sant Antoni': 'Sant Antoni'}

    censustract_list = get_impacted_censustracts(impacted_gdf["geometry"].union_all(), ine_gdf) 
    df_census = get_timeseries_of_census_tracts(df, censustract_list)

    if df_census is None:
        raise ValueError("funtion get_timeseries_of_census_tracts returned None")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=df_census["sale"].index,
                   y=df_census["sale"].values, 
                   name="Average buy",
                   line=dict(color= SALE_COLOR)),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=df_census["rent"].index,
                   y=df_census["rent"].values,
                   name="Average rent",
                   line=dict(color= RENT_COLOR)),
        secondary_y=True,
    )

    if district == True:
        census_district = list(district_gdf['CENSUSTRACT'].astype(int).astype(str))
        df_district_census = get_timeseries_of_census_tracts(df, census_district)

        fig.add_trace(
            go.Scatter(x=df_district_census["sale"].index, 
                       y=df_district_census["sale"].values, 
                       name="District buy", 
                       line=dict(color= CONTROL_SALE)),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=df_district_census["rent"].index, 
                       y=df_district_census["rent"].values, 
                       name="District rent", 
                       line=dict(color= CONTROL_COLOR)),
            secondary_y=True,
        )
    
    if control_polygon == True:
        census_district = list(district_gdf['CENSUSTRACT'].astype(int).astype(str))
        control_gdf_census = get_timeseries_of_census_tracts(control_gdf, control_gdf['CENSUSTRACT'])

        fig.add_trace(
            go.Scatter(x=control_gdf_census["sale"].index, 
                       y=control_gdf_census["sale"].values, 
                       name="Control buy", 
                       line=dict(color=CONTROL_SALE)),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=control_gdf_census["rent"].index, 
                       y=control_gdf_census["rent"].values, 
                       name="Control rent", 
                       line=dict(color=CONTROL_COLOR)),
            secondary_y=True,
        )

    if include_trends:
        trend_sale = get_trend_of_timeseries(df_census["sale"])
        trend_rent = get_trend_of_timeseries(df_census["rent"])

        fig.add_trace(
            go.Scatter(x=trend_sale.index, 
                       y=trend_sale.values, 
                       name="Trend buy",
                       line=dict(color= SALE_COLOR, dash="dash")),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=trend_rent.index, 
                       y=trend_rent.values, 
                       name="Trend rent",
                       line=dict(color= RENT_COLOR, dash="dash")),

            secondary_y=True,
        )

        if district == True:
            trend_sale_district = get_trend_of_timeseries(df_district_census["sale"])
            trend_rent_district = get_trend_of_timeseries(df_district_census["rent"])

            fig.add_trace(
            go.Scatter(x=trend_sale_district.index, 
                       y=trend_sale_district.values, 
                       name="Trend district buy",
                       line=dict(color= CONTROL_SALE, dash="dash")),
            secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=trend_rent_district.index, 
                           y=trend_rent_district.values, 
                           name="Trend district rent",
                           line=dict(color= CONTROL_COLOR, dash="dash")),
                secondary_y=True,
            )
        

        if control_polygon == True:
            trend_sale_control = get_trend_of_timeseries(control_gdf_census["sale"])
            trend_rent_control = get_trend_of_timeseries(control_gdf_census["rent"])

            fig.add_trace(
            go.Scatter(x=trend_sale_control.index, 
                       y=trend_sale_control.values, 
                       name="Trend control buy",
                       line=dict(color= CONTROL_COLOR, dash="dash")),
            secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=trend_rent_control.index, 
                           y=trend_rent_control.values, 
                           name="Trend control rent",
                           line=dict(color= CONTROL_COLOR, dash="dash")),
                secondary_y=True,
            )

    # Ensure CENSUSTRACT values in interventions_gdf are strings with 10 digits
    interventions_gdf["CENSUSTRACT"] = interventions_gdf["CENSUSTRACT"].astype(int).astype(str).str.zfill(10)

    # Ensure the census tract list values are also 10 digits
    censustract_list = [str(ct).zfill(10) for ct in censustract_list]

    # Debugging print
    #print("CENSUSTRACT values in interventions_gdf:", interventions_gdf["CENSUSTRACT"].unique())
    #print("Censustract list provided:", censustract_list)
    #print(interventions_gdf["CENSUSTRACT"].values)

    if any(census_tract in interventions_gdf["CENSUSTRACT"].values for census_tract in censustract_list):
        # Prepare data for merging intervals

        intervals = []
        # Ensure CENSUSTRACT values in interventions_gdf are strings with 10 digits
        interventions_gdf["CENSUSTRACT"] = interventions_gdf["CENSUSTRACT"].astype(str).str.zfill(10)

        # Ensure the census tract list values are also 10 digits
        censustract_list = [str(ct).zfill(10) for ct in censustract_list]

        # Filter censustract_list to include only those that are present in the DataFrame
        valid_censustracts = [ct for ct in censustract_list if ct in interventions_gdf["CENSUSTRACT"].values]

        # Now proceed with only the valid census tracts
        intervals = []
        #print(interventions_gdf.set_index("CENSUSTRACT").loc[valid_censustracts])

        for row, intervention in interventions_gdf.set_index("CENSUSTRACT").loc[valid_censustracts].iterrows():
            # Debugging print
            #print(f"Processing intervention: {intervention}")

            data_inici = pd.to_datetime(intervention["DATA_INICI"].strftime('%Y-%m-%d'))
            data_fi = pd.to_datetime(intervention["DATA_FI_REAL"].strftime('%Y-%m-%d'))
            intervals.append((data_inici, data_fi, {interventions_dict[intervention["TITOL_WO"]]}))

        # Merge overlapping intervals
        merged_intervals = merge_intervals(intervals)

        # Draw rectangles for each merged interval
        for start, end, interventions in merged_intervals:
            # Create annotation text with line breaks
            annotation_text = '<br>'.join(sorted(interventions))

            fig.add_vrect(
                x0=start,
                x1=end,
                annotation_text=annotation_text,
                annotation_position="bottom right",
                fillcolor=INTERVENTION_COLOR,
                opacity=0.25,
                line_width=0
            )
                # Price type filtering

    if price_type == 'sale':
        fig.data = [trace for trace in fig.data if "buy" in trace.name]
    elif price_type == 'rent':
        fig.data = [trace for trace in fig.data if "rent" in trace.name]

    fig.update_layout(
        title_text="Average Rent/Buy prices for all the Census tracts"
    )

    fig.update_xaxes(title_text="Periods")
    fig.update_yaxes(title_text="<b>Rent</b> price", secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text="<b>Buy</b> price", secondary_y=False)
    return fig