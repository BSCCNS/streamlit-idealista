
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


def filter_data_per_district(df: pd.DataFrame, gdf: gpd.GeoDataFrame) -> pd.DataFrame:

    """
    Filter idealista df to the censustracts of the districts contained in the given geopandas
    (e.g. for interventions_gdf)

    Args:
      df (pd.DataFrame): The dataframe containing idealista data.
      gdf (gpd.GeoDataFrame): GeoPandas with geometry 

    Returns:
      pd.DataFrame: filter dataframe 
    """

    df['district'] = df['CENSUSTRACT'].astype(str).str[4:6]
    df['munucipality'] = df['CENSUSTRACT'].astype(str).str[0:4]

    c1 = df.district.isin(gdf.DISTRITO)
    c2 = df.munucipality.isin(gdf['PROVMUN'].astype(int).astype(str))
    df_district = df[c1 & c2]

    return df_district

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

def plot_timeseries(df: pd.DataFrame,
                    interventions_gdf: gpd.GeoDataFrame,
                    censustract_list: Union[List[str], None] = None,
                    include_trends: bool=True,
                    price_type: str = 'both',
                    district: bool =True
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

    #df['district'] = df['CENSUSTRACT'].astype(str).str[4:6]
    #df['munucipality'] = df['CENSUSTRACT'].astype(str).str[0:4]

    df_census = get_timeseries_of_census_tracts(df, censustract_list)

    #print("df_census", df_census)
    if df_census is None:
        raise ValueError("funtion get_timeseries_of_census_tracts returned None")



    fig = make_subplots(specs=[[{"secondary_y": True}]])


    fig.add_trace(
        # TA-change
        go.Scatter(x=df_census["sale"].index, y=df_census["sale"].values, name="Average buy"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=df_census["rent"].index, y=df_census["rent"].values, name="Average rent"),
        secondary_y=True,
    )

    if include_trends:
        trend_sale = get_trend_of_timeseries(df_census["sale"])
        trend_rent = get_trend_of_timeseries(df_census["rent"])

        fig.add_trace(
            go.Scatter(x=trend_sale.index, y=trend_sale.values, name="Trend buy"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=trend_rent.index, y=trend_rent.values, name="Trend rent"),
            secondary_y=True,
        )
        if district == True:
    
            #df_districts = df[df.district.isin(interventions_gdf.DISTRITO)]
            #c1 = df.district.isin(interventions_gdf.DISTRITO)
            #c2 = df.munucipality.isin(interventions_gdf['PROVMUN'].astype(int).astype(str))
            #df_districts = df[c1 & c2]

            df_districts = filter_data_per_district(df, interventions_gdf)

            df_districts_list = get_timeseries_of_census_tracts(df_districts, censustract_list)

            trend_sale_district = get_trend_of_timeseries(df_districts_list["sale"])
            trend_rent_district = get_trend_of_timeseries(df_districts_list["rent"])
            fig.add_trace(
                go.Scatter(x=trend_sale_district.index, y=trend_sale_district.values, name="District buy",line=dict(color='#C1A2CA')),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=trend_rent_district.index, y=trend_rent_district.values, name="District rent", line=dict(color='#C1A2CA')),
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
                fillcolor="green",
                opacity=0.25,
                line_width=0
            )
                # Price type filtering

    #TA-change
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