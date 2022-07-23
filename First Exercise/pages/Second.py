import streamlit as st
import pandas as pd
import pickle

filename = 'titanic_model.sav'
model = pickle.load(open(filename, 'rb'))

df = pd.read_csv('second.csv')

titles = ('Not applicable', 'Dr.', 'Rev.',
          'Miss.', 'Master.', 'Don.', 'Mme.',
          'Major.', 'Lady.', 'Sir.', 'Mlle.', 'Col.', 'Capt.', 'Countess.', 'Jonkheer.')

ports_range = ('Queenstown, Ireland', 'Southampton, U.K.')

Pclass = (1, 2, 3)

st.markdown("""
<style>
.big-font {
    font-size:60px !important;

     color :#C7BACC !important;
               font-family: 'Roboto', sans-serif;
}
.colored-font {
    font-size:50px !important;
    color: grey !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Would you make it if you were on the Titanic? Describe your self using left menu and find out. </p>', unsafe_allow_html=True)

st.sidebar.title('Describe yourself')

title = st.sidebar.selectbox('Your title', titles)
Title_Unusual = 0 if title == "Not applicable" else 1

gender = st.sidebar.radio('Sex', ('Male', 'Female'))
Sex = 0 if gender == 'Male' else 1

Age = st.sidebar.slider('Age', 0, 90, 1)

Pclass = st.sidebar.radio('Class', Pclass)
fare_range = df.loc[df['Pclass'] == Pclass, 'Fare']

Cabin_Known = st.sidebar.radio('Cabin', (0, 1))

ports = st.sidebar.radio('Port of departure', ports_range)
Embarked_Q = 1 if ports == 'Queenstown, Ireland' else 0
Embarked_S = 1 if ports == "Southampton, U.K." else 0

Fare = st.sidebar.slider('How much was your ticket (£)?', min(fare_range), max(fare_range))
SibSp = st.sidebar.slider("How many siblings are on the Titanic with you?", 0, 10)
Parch = st.sidebar.slider("Parents or children with you?", 0, 10)
prediction_inp = [Pclass] + [Sex] + [Age] + [SibSp] + [Parch] + [Fare] + \
                 [Title_Unusual] + [Cabin_Known] + [Embarked_Q] + [Embarked_S]

survial = model.predict_proba([prediction_inp])[0,1]
survial = round(survial,2)

if survial*100 >= 50:

    fate = "Survive"
    st.write(survial)
    st.write('You will likely  '+ fate)
else:
    fate = "Die"
    st.write(survial)
    st.write('You will likely  '+ fate)
