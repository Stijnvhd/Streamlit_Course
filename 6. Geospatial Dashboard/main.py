import streamlit as st
import pandas as pd
import numpy as np

st.header("Geospatial dashboard")

from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
counties["features"][0]
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/fips-unemp-16.csv",
                   dtype={"fips": str})
df.head()
import plotly.express as px
st.write("An overview of the unenmployment rates in the use in a clear and easy to use map")
fig = px.choropleth_mapbox(df, geojson=counties, locations='fips', color='unemp',
                           color_continuous_scale="Viridis",
                           range_color=(0, 12),
                           mapbox_style="carto-positron",
                           zoom=3, center = {"lat": 37.0902, "lon": -95.7129},
                           opacity=0.5,
                           labels={'unemp':'unemployment rate'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig, use_container_width=True)


df3 = px.data.election()
geojson = px.data.election_geojson()

fig3 = px.choropleth(df3, geojson=geojson, color="Bergeron",
                    locations="district", featureidkey="properties.district",
                    projection="mercator"
                   )
fig3.update_geos(fitbounds="locations", visible=False)
fig3.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig3, use_container_width=True)

st.subheader("What is the difference in income and development?")
df2 = px.data.gapminder()
fig2 = px.scatter_geo(df2, locations="iso_alpha", color="continent",
                     hover_name="country", size="pop",
                     projection="natural earth")

st.plotly_chart(fig2, use_container_width=True)
