import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

st.image("background2.png")

st.write(""" # College Recommendation Site 
         By: Calder Glass & Katie Pyo
         """)

collegeData = pd.read_csv("tentativeCollegeData.csv", index_col=0)

state_book = {
    "AL": "Alabama", 
    "AR": "Arkansas",
    "AZ": "Arizona",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DC": "Washington, DC",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin"
# note Alaska, Delaware, and Wyoming are missing 
}

# concatenating college name + city into general location var
collegeData["General Location"] = collegeData["School Name"] + " " + collegeData["City"]

# cleaning the name of the given school
def clean_schoolName(schoolName):
    return re.sub("[^a-zA-Z0-9 ]", "", schoolName)

# apply this function to the general location variable
collegeData["General Location"].apply(clean_schoolName)
# setting up the vectorizer and getting the tf-idf scores between different colleges
search_vectorizer = TfidfVectorizer(ngram_range=(1,2))
college_tfidif = search_vectorizer.fit_transform(collegeData["General Location"])

# takes a college, cleans the name, and then uses cosine similarity to find similar colleges to it 
# then returns those colleges

def search_college(collegeName):
   collegeName = clean_schoolName(collegeName)
   query_colleges = search_vectorizer.transform([collegeName])
   similaritycollege = cosine_similarity(query_colleges, college_tfidif).flatten()
   collegeIndices = np.argpartition(similaritycollege, -50)[-50:] # 50 most similar colleges
   collegeSubset = collegeData.iloc[collegeIndices][::-1]
   return collegeSubset

collegeData2 = collegeData.copy()

collegeData2 = collegeData2.drop(["General Location"], axis=1)

# first input college name, then expand options

# initializing filter variables ahead of time + list of state values

stateValues = list(state_book.values())
givenStateKey = ""
sat_score = 0
median_earnings = 0
studentFacultyRatio = 0
admission_rate = 0
in_stateTuition = 0
degree_of_Urbanization = ""
onCampus_housingCost = 0
offCampus_housingCost = 0
studentPopulationSize = 0
collegeData3 = pd.DataFrame()

st.write("Restart by leaving the text box blank and pressing enter.")
namePrompt = st.text_input("Type a college's name.")
if namePrompt:
    collegeData2 = search_college(namePrompt)
    with st.form("Filters:"):
        col1, col2, col3 = st.columns(3)
        # filter colleges by the given slider or drop down input, the logical argument for the numerics is less than or equal to
        with col1:
            # button for inputting college state
            state = st.selectbox("Choose the state", stateValues)
            if state:
                givenStateKey = [key for key, val in state_book.items() if val == state]
                givenStateKey = givenStateKey[0]
            # sat score slider
            sat_score = st.slider("Average Admit's SAT Score", min_value = 0, max_value = 1600, step = 10, value = 1020)
            # median earnings slider
            median_earnings = st.slider("Median Earnings, 7 Years after Entering College ($)", min_value = 0, 
                                        max_value = 125000, step = 1000, value = 61000)
            # student to faculty ratio slider
            studentFacultyRatio = st.slider("Student to Faculty Ratio", min_value= 0, max_value=29, value = 14, step = 1)
        with col2:
            # admission rate slider
            admission_rate = st.slider("Admission Rate (%)", min_value = 0, max_value = 100, step = 1, value = 50)
            # instate tuition slider
            in_stateTuition = st.slider("In-State Tuition ($)", min_value= 0, max_value=70000, value = 30000, step = 1000)
            # location drop down menu
            #degree_of_Urbanization = st.selectbox("Select the general location of the college.",
                                           #("City, Population Size >= 250,000", "City, Population Size between 100,000 and 250,000",
                                            #"City, Population < 100,000", "Suburb, Population >= 250,000", 
                                            #"Suburb, Population between 100,000 and 250,000", "Suburb, Population < 100,000",
                                            #"Distant Town, Urban Cluster is that is between 10 and 35 miles from an urban area",
                                            #"Rural Town, Urban Cluster 35 miles away from an urban area",
                                            #"Fringe Rural, <= 5 miles from an urban area and <= 2.5 miles from an Urban Cluster",
                                            #"Distant Rural, Between 5 to 25 miles from an urban area and between 2.5 to 10 miles from an Urban Cluster",
                                            #"Remote Rural, More than 25 miles from an urban area and more than 10 miles from an Urban Cluster"))
        with col3:
            #if st.button("Major"):
            # st.text("Write your major of interest")
            # majorPrompt = st.chat_input("")
            # on-campus housing cost slider
            onCampus_housingCost = st.slider("On-Campus Housing Cost ($)", min_value = 0, max_value = 25000, step = 1000, value = 12000)
            # off-campus housing cost slider
            offCampus_housingCost = st.slider("Off-Campus Housing Cost ($)", min_value = 0, max_value = 26000, step = 1000, value = 13000)
            # student population size slider
            studentPopulationSize = st.slider("Student Population Size", min_value= 0, max_value=60000, value = 30000, step = 1000)
        st.form_submit_button("Click once you are done filtering colleges for select attributes")

collegeData3 = collegeData2[(collegeData2["Student Population Size"] <= studentPopulationSize) 
                            & (collegeData2["Off-Campus Housing Cost"] <= offCampus_housingCost)
                            & (collegeData2["On-Campus Housing Cost"] <= onCampus_housingCost) 
                            #& (collegeData2["Degree of Urbanization"] == degree_of_Urbanization) 
                            & (collegeData2["In-State Tuition"] <= in_stateTuition) 
                            & (collegeData2["Admission Rate"] >= (admission_rate / 100)) 
                            & (collegeData2["Student to Faculty Ratio"] <= studentFacultyRatio) 
                            & (collegeData2["Median Earnings, 7 Years after Entering College"] <= median_earnings) 
                            & (collegeData2["Average SAT Score"] >= sat_score)
                            & (collegeData2["State"].isin([givenStateKey]))]
if collegeData3.empty == False:
    st.data_editor(collegeData3, column_config={"School Name": st.column_config.Column(width="large")}, column_order=("School Name", 
    "City", "State", "Admission Rate", "In-State Tuition", "Student to Faculty Ratio", "Student Population Size", 
    "On-Campus Housing Cost", "Off-Campus Housing Cost", "Degree of Urbanization", "Median Earnings, 7 Years after Entering College",
      "Average SAT Score"), hide_index=True, disabled=True)