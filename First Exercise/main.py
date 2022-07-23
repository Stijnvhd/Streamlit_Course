import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title('The Titanic Dataset')
df = pd.read_csv("second.csv")

st.subheader("The data")
st.write(df)

st.subheader("Some visualizations")

container_1_show = st.expander("Show Descriptive Statistics")
with container_1_show:
    container_1 = st.container()
    container_1.subheader('Looking at the data')
    container_1.write(df.describe())

container_2_show = st.expander("Show basic count plots")
with container_2_show:
    container_2 = st.container()
    col1, col2, col3 = container_2.columns([3, 3, 3])

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.title('Survived')
    sns.countplot(data=df, x='Survived')
    col1.pyplot(plt)

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.title('Gender')
    sns.countplot(x='Sex', data=df)
    col2.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.title('Cabin_Known')
    sns.countplot(x='Cabin_Known', data=df)
    col3.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.title('Pclass')
    sns.countplot(x='Pclass', data=df)
    col1.pyplot(fig)

container_3_show = st.expander("Show comparison plots")
with container_3_show:
    container_2 = st.container()
    col1, col2, col3 = container_2.columns([3, 3, 3])

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.title('Gender')
    sns.countplot(x='Survived', hue='Sex', data=df)
    col1.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.title('Pclass')
    sns.countplot(x='Survived', hue='Pclass', data=df)
    col2.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.title('Cabin_Known')
    sns.countplot(x='Survived', hue='Cabin_Known', data=df)
    col3.pyplot(fig)
