import streamlit as st
import plotly.express as px

st.set_page_config(layout="wide")

# -- Create three columns
col1, col2, col3 = st.columns([10, 5, 10])
# -- Put the image in the middle column
# - Commented out here so that the file will run without having the image downloaded

with col1:
    st.image("streamlit.png", width=200)
# -- Put the title in the last column
with col2:
    st.title("Streamlit Demo")

year_col, continent_col, log_x_col = st.columns([5, 5, 5])
with year_col:
    year_choice = st.slider(
        "What year would you like to examine?",
        min_value=1952,
        max_value=2007,
        step=5,
        value=2007,
    )
with continent_col:
    continent_choice = st.selectbox(
        "What continent would you like to look at?",
        ("All", "Asia", "Europe", "Africa", "Americas", "Oceania"),
    )
with log_x_col:
    log_x_choice = st.checkbox("Log X Axis?")

# -- Read in the data
df = px.data.gapminder()
st.subheader("Have a look at the data")
x, y, z = st.columns([5, 5, 5])
with y:
    st.write(df, use_container_width=True)
# -- Apply the year filter given by the user
filtered_df = df[(df.year == year_choice)]
# -- Apply the continent filter
if continent_choice != "All":
    filtered_df = filtered_df[filtered_df.continent == continent_choice]

st.subheader("And here is our graph")
# -- Create the figure in Plotly
fig = px.scatter(
    filtered_df,
    x="gdpPercap",
    y="lifeExp",
    size="pop",
    color="continent",
    hover_name="country",
    log_x=log_x_choice,
    size_max=60,
)
fig.update_layout(title="GDP per Capita vs. Life Expectancy")
# -- Input the Plotly chart to the Streamlit interface
st.plotly_chart(fig, use_container_width=True)
