import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode

st.title("A first example")
st.write("Here we start to explore our first example in Streamlit")
st.header("Welcome here")

@st.cache
def load_data():
    data = pd.read_csv(r'C:\Users\HP\Desktop\Oreilly\Streamlit\A first static dashboard\diabetes.csv')
    return data

data = load_data()

st.subheader('Raw data')
st.dataframe(data)

st.subheader('Data Descripition')
st.write(data.describe())

st.subheader('Relation with outcome')

if st.sidebar.checkbox("Seaborn Pairplot"):
    fig = sns.pairplot(data = data, hue = 'Outcome')
    st.pyplot(fig)
if st.sidebar.checkbox("Histplot"):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    data.hist(alpha=0.5, figsize=(20, 10))
    plt.show()
    st.pyplot()

agree = st.sidebar.button('Click to see nothing')

st.subheader("An interesting table")
def aggrid_interactive_table(df: pd.DataFrame):
    """Creates an st-aggrid interactive table based on a dataframe.
    Args:
        df (pd.DataFrame]): Source dataframe
    Returns:
        dict: The selected row
    """
    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True
    )

    options.configure_side_bar()

    options.configure_selection("single")
    selection = AgGrid(
        df,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        theme="light",
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
    )

    return selection

selection = aggrid_interactive_table(df=data)

if selection:
    st.write("You selected:")
    st.json(selection["selected_rows"])
