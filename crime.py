import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.graph_objs as go
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import plotly.express as px
import joblib

#---------------------------------------------------DATASET--------------------------------------------------------

df_crime_final=pd.read_csv(r"Chicago_Crimes.csv")

#-------------------------------------ANALYSIS GRAPHS AND CHARTS-------------------------------------------
### 1. Temporal Analysis
def crime_rates_over_time():
    crimes_per_year = df_crime_final['Year'].value_counts().sort_index()

    # Creating a Plotly line plot for crimes per year
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=crimes_per_year.index, y=crimes_per_year.values,
                            mode='lines+markers', marker=dict(color='blue'),
                            hovertemplate='<b>Year:</b> %{x}<br><b>Number of Crimes:</b> %{y}<extra></extra>'))
    fig.update_layout(title='Number of Crimes Per Year',
                    xaxis_title='Year',
                    yaxis_title='Number of Crimes')
    st.plotly_chart(fig)


#____________________________________________________________________________________________________________

def crimes_per_month():
    crimes_per_month = df_crime_final.groupby(['Year', 'Month']).size().reset_index(name='Count')

    # Creating a Plotly line plot for crimes per month
    fig = go.Figure()
    for year in crimes_per_month['Year'].unique():
        data_year = crimes_per_month[crimes_per_month['Year'] == year]
        fig.add_trace(go.Scatter(x=data_year['Month'], y=data_year['Count'],
                                mode='lines+markers', name=str(year),
                                hovertemplate='<b>Month:</b> %{x}<br><b>Number of Crimes:</b> %{y}<extra></extra>'))
    fig.update_layout(title='Number of Crimes Per Month',
                    xaxis_title='Month',
                    yaxis_title='Number of Crimes')
    st.plotly_chart(fig)

#_______________________________________________________________________________________________________________

def crimes_per_hour():
    # Grouping data by hour of the day and counting the number of crimes
    crimes_per_hour = df_crime_final['Hour'].value_counts().sort_index()

    # Creating a Plotly bar plot for crimes per hour
    fig = go.Figure()
    fig.add_trace(go.Bar(x=crimes_per_hour.index, y=crimes_per_hour.values,
                        marker=dict(color='green')))
    fig.update_layout(title='Number of Crimes Per Hour',
                    xaxis_title='Hour of the Day',
                    yaxis_title='Number of Crimes')
    st.plotly_chart(fig)
#____________________________________________________________________________________________________________________

def crimes_per_day(): 
    # Group data by day of the week and count the number of crimes
    crimes_per_day = df_crime_final['Day of Week'].value_counts().reset_index()
    crimes_per_day.columns = ['Day of Week', 'Crime Count']

    # Sort days of the week in chronological order
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    crimes_per_day = crimes_per_day.sort_values('Day of Week', key=lambda x: pd.Categorical(x, categories=days_order, ordered=True))

    # Create a line chart for crimes per day of the week
    fig = go.Figure(data=[
        go.Scatter(x=crimes_per_day['Day of Week'], y=crimes_per_day['Crime Count'], mode='lines+markers')
    ])

    fig.update_layout(
        title='Number of Crimes Per Day of the Week',
        xaxis_title='Day of the Week',
        yaxis_title='Number of Crimes'
    )

    st.plotly_chart(fig)
#_______________________________________________________________________________________________________________________________

# Function to determine the season based on the month
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

# Ensure the 'Date' column is datetime
df_crime_final['Date'] = pd.to_datetime(df_crime_final['Date'])

# Create the 'Season' column if it doesn't exist
if 'Season' not in df_crime_final.columns:
    df_crime_final['Month'] = df_crime_final['Date'].dt.month
    df_crime_final['Season'] = df_crime_final['Month'].apply(get_season)

# Define the order of seasons
season_order = ['Winter', 'Spring', 'Summer', 'Fall']

def crimes_per_season():
    # Group data by season and count the number of crimes
    crimes_per_season = df_crime_final['Season'].value_counts().reset_index()
    crimes_per_season.columns = ['Season', 'Crime Count']

    # Ensure seasons are in the correct order
    crimes_per_season['Season'] = pd.Categorical(crimes_per_season['Season'], categories=season_order, ordered=True)
    crimes_per_season = crimes_per_season.sort_values('Season')

    # Create a bar chart for crimes per season
    fig = go.Figure(data=[
        go.Bar(x=crimes_per_season['Season'], y=crimes_per_season['Crime Count'])
    ])

    fig.update_layout(
        title='Number of Crimes Per Season',
        xaxis_title='Season',
        yaxis_title='Number of Crimes'
    )
    st.plotly_chart(fig)
#__________________________________________________________________________________________________________________________

def map(selected_year):
    # Filter the data based on the selected year
    if selected_year != "All Years":
        filtered_df = df_crime_final[df_crime_final['Year'] == selected_year]
    else:
        filtered_df = df_crime_final

    # Create a base map centered around Chicago
    chicago_map = folium.Map(location=[41.8781, -87.6298], zoom_start=11)

    # Extract latitude and longitude coordinates
    crime_locations = filtered_df[['Latitude', 'Longitude']].dropna().values.tolist()

    # Add heatmap layer to the map
    HeatMap(crime_locations, radius=10).add_to(chicago_map)

    # Display the map in Streamlit
    folium_static(chicago_map)



#_______________________________________________________________________________________________________________
def crime_ward():
    # Group data by ward and count the number of crimes
    ward_crime_counts = df_crime_final['Ward'].value_counts().reset_index()
    ward_crime_counts.columns = ['Ward', 'Crime Count']

    # Create a bar chart for crimes per ward
    fig = go.Figure(go.Bar(
        x=ward_crime_counts['Ward'],
        y=ward_crime_counts['Crime Count'],
        marker=dict(color='red'),  # Adjust color if needed
    ))

    fig.update_layout(
        title='Crimes Per Ward',
        xaxis_title='Ward',
        yaxis_title='Number of Crimes',
    )

    st.plotly_chart(fig)
#________________________________________________________________________________________________________________
### Distribution of Crime Types
def crime_type():
    # Count the frequency of each crime type
    crime_type_counts = df_crime_final['Primary Type'].value_counts().reset_index()
    crime_type_counts.columns = ['Crime Type', 'Crime Count']

    # Sort the DataFrame by crime count in descending order
    crime_type_counts = crime_type_counts.sort_values(by='Crime Count', ascending=False)

    # Create a line graph for the distribution of crime types
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=crime_type_counts['Crime Type'], y=crime_type_counts['Crime Count'], mode='lines+markers'))

    fig.update_layout(
        title='Distribution of Crime Types',
        xaxis_title='Crime Type',
        yaxis_title='Crime Count',
        height=800 
    )

    st.plotly_chart(fig)
#_______________________________________________________________________________________________________________________

def crime_severity():
    # List of severe crimes
    severe_crimes = ['HOMICIDE', 'CRIMINAL SEXUAL ASSAULT', 'OFFENSE INVOLVING CHILDREN', 
                    'ROBBERY', 'ASSAULT', 'SEX OFFENSE', 'WEAPONS VIOLATION', 
                    'KIDNAPPING', 'ARSON', 'HUMAN TRAFFICKING', 'CRIM SEXUAL ASSAULT', 
                    'DOMESTIC VIOLENCE']

    # List of less severe crimes
    less_severe_crimes = ['BURGLARY', 'BATTERY', 'CRIMINAL DAMAGE', 'DECEPTIVE PRACTICE', 
                        'THEFT', 'MOTOR VEHICLE THEFT', 'OTHER OFFENSE', 'STALKING', 
                        'CRIMINAL TRESPASS', 'PROSTITUTION', 'NARCOTICS', 
                        'CONCEALED CARRY LICENSE VIOLATION', 'INTERFERENCE WITH PUBLIC OFFICER', 
                        'PUBLIC PEACE VIOLATION', 'OBSCENITY', 'LIQUOR LAW VIOLATION', 
                        'INTIMIDATION', 'GAMBLING', 'OTHER NARCOTIC VIOLATION', 'NON-CRIMINAL', 
                        'PUBLIC INDECENCY', 'RITUALISM', 'NON-CRIMINAL (SUBJECT SPECIFIED)', 
                        'NON - CRIMINAL']

    # Count the frequency of severe and less severe crimes
    severe_count = sum(df_crime_final['Primary Type'].isin(severe_crimes))
    less_severe_count = sum(df_crime_final['Primary Type'].isin(less_severe_crimes))

    # Create hover text for each category
    hover_text_severe = f'Severe Crimes:<br>{"<br>".join(severe_crimes)}'
    hover_text_less_severe = f'Less Severe Crimes:<br>{"<br>".join(less_severe_crimes)}'

    # Create a donut chart for the distribution of crimes by severity
    labels = ['Severe', 'Less Severe']
    values = [severe_count, less_severe_count]
    hover_text = [hover_text_severe, hover_text_less_severe]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, hovertext=hover_text, hoverinfo='text')])
    fig.update_traces(textinfo='label+percent', insidetextorientation='radial')

    fig.update_layout(title='Distribution of Crimes by Severity',height=500)
    fig.update_layout(hoverlabel=dict(font=dict(size=8)))
    st.plotly_chart(fig)
#________________________________________________________________________________________________________________________________

def arrest_rates():
    # Calculate arrest rates by crime type
    arrest_rates = df_crime_final.groupby('Primary Type')['Arrest'].mean().sort_values(ascending=False).reset_index()

    # Create a bar chart for arrest rates
    fig_arrest_rates = go.Figure(data=[
        go.Bar(
            x=arrest_rates['Primary Type'],
            y=arrest_rates['Arrest'] * 100,  # Convert to percentage
            text=arrest_rates['Arrest'] * 100,
            textposition='auto'
        )
    ])

    fig_arrest_rates.update_layout(
        title='Arrest Rates by Crime Type',
        xaxis_title='Crime Type',
        yaxis_title='Arrest Rate (%)',
        xaxis_tickangle=-45,
        height=600
    )

    st.plotly_chart(fig_arrest_rates)
#__________________________________________________________________________________________________________________

def domestic():
    # Calculate the number of domestic vs. non-domestic crimes
    domestic_counts = df_crime_final['Domestic'].value_counts()

    # Create a pie chart for domestic vs. non-domestic crimes
    fig_domestic = go.Figure(data=[
        go.Pie(
            labels=['Non-Domestic', 'Domestic'],
            values=[domestic_counts[False], domestic_counts[True]],
            hole=.3,
            hoverinfo='label+percent+value'
        )
    ])

    fig_domestic.update_layout(
        title='Domestic vs. Non-Domestic Crimes'
    )

    st.plotly_chart(fig_domestic)

def detailed_domestic():
    # Calculate the distribution of crime types for domestic vs. non-domestic incidents
    domestic_crime_types = df_crime_final[df_crime_final['Domestic'] == True]['Primary Type'].value_counts().reset_index()
    domestic_crime_types.columns = ['Primary Type', 'Domestic Count']

    non_domestic_crime_types = df_crime_final[df_crime_final['Domestic'] == False]['Primary Type'].value_counts().reset_index()
    non_domestic_crime_types.columns = ['Primary Type', 'Non-Domestic Count']

    # Merge the dataframes for comparison
    domestic_comparison = domestic_crime_types.merge(non_domestic_crime_types, on='Primary Type', how='outer').fillna(0)

    # Create a bar chart for domestic vs. non-domestic crime types
    fig_domestic_comparison = go.Figure(data=[
        go.Bar(
            x=domestic_comparison['Primary Type'],
            y=domestic_comparison['Domestic Count'],
            name='Domestic'
        ),
        go.Bar(
            x=domestic_comparison['Primary Type'],
            y=domestic_comparison['Non-Domestic Count'],
            name='Non-Domestic'
        )
    ])

    fig_domestic_comparison.update_layout(
        title='Crime Types: Domestic vs. Non-Domestic',
        xaxis_title='Crime Type',
        yaxis_title='Number of Crimes',
        barmode='group',
        xaxis_tickangle=-45,
        height=800
    )

    st.plotly_chart(fig_domestic_comparison)
#___________________________________________________________________________________________________________________________

def location_specific():

    # Calculate the most common locations for crimes
    location_counts = df_crime_final['Location Description'].value_counts().head(30).reset_index()
    location_counts.columns = ['Location Description', 'Crime Count']

    # Create a horizontal bar chart for the most common crime locations
    fig_location_counts = go.Figure(data=[
        go.Bar(
            x=location_counts['Crime Count'],
            y=location_counts['Location Description'],
            orientation='h',
            text=location_counts['Crime Count'],
            textposition='auto'
        )
    ])

    fig_location_counts.update_layout(
        title='Top 30 Most Common Locations for Crimes',
        xaxis_title='Number of Crimes',
        yaxis_title='Location Description',
        height=600
    )

    st.plotly_chart(fig_location_counts)
#_________________________________________________________________________________________________________________________________________

def beat_community():

    # Group the data by beat and community area and count the number of crimes
    crime_heatmap_data = df_crime_final.groupby(['Beat', 'Community Area']).size().reset_index(name='Crime Count')

    # Create the heatmap
    fig_heatmap = go.Figure(data=go.Heatmap(
                    z=crime_heatmap_data['Crime Count'],
                    x=crime_heatmap_data['Beat'],
                    y=crime_heatmap_data['Community Area'],
                    hovertext=['Beat: {}<br>Community Area: {}<br>Crime Count: {}'.format(beat, area, count) 
                                for beat, area, count in zip(crime_heatmap_data['Beat'], 
                                                            crime_heatmap_data['Community Area'], 
                                                            crime_heatmap_data['Crime Count'])],
                    colorscale='Reds',  # Choose the colorscale
                    colorbar=dict(title='Crime Count')  # Add a colorbar
    ))

    # Update layout
    fig_heatmap.update_layout(
        title='Crime Heatmap by Beat and Community Area',
        xaxis_title='Police Beat',
        yaxis_title='Community Area',
        height=800,
        plot_bgcolor='white',  # White background
        font=dict(color='black')  # Font color
    )

    st.plotly_chart(fig_heatmap)
#________________________________________________________________________________________________________________________________

def repeat():

    # Grouping by Block and counting the occurrences
    repeat_crime_locations = df_crime_final['Block'].value_counts().reset_index()
    repeat_crime_locations.columns = ['Block', 'Crime Count']

    # Visualizing the top 10 blocks with the highest crime counts
    fig = px.bar(repeat_crime_locations.head(10), x='Block', y='Crime Count',
                title='Top 10 Repeat Crime Locations',
                labels={'Crime Count':'Number of Crimes', 'Block':'Location (Block)'},
                text='Crime Count')

    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig)
#_______________________________________________________________________________________________________________________
#predictive modelling

# Load the trained model and scaler
model = joblib.load('crime_predictor.pkl')
scaler = joblib.load('scaler.pkl')

#--------------------------------------------------STREAMLIT------------------------------------------------------------
st.set_page_config(page_title="Chicago Crime Analyzer", page_icon="🕵️‍♀️", layout="wide")
st.markdown("<h1 style='font-size: 50px; text-align: Center;'>CHICAGO CRIME ANALYZER</h1>", unsafe_allow_html=True)

# Horizontal Menu Bar with Icons
selected_option = option_menu(
    menu_title=None,
    options=["Home", "Analyzer", "Modeling"],
    icons=["house", "map", "tools"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)


# Handle page navigation
if selected_option == "Home":
    st.session_state['page'] = "Home"
    col1, col2 = st.columns([2, 2])
    with col1:
        st.image("p1.jpg",width=450)
        st.image("p2.jpg",width=450)
        st.image("p3.jpg",width=450)
        st.image("p4.jpg",width=450)
        st.image("p5.jpg",width=450)
    with col2:
        try:
            with open('home.txt', 'r') as file:
                file_contents = file.read()
            st.write(file_contents)
        except Exception as e:
            st.write("An error occurred while reading the file:")
            st.write(e)

if selected_option == "Analyzer":
    with st.sidebar:
        selected_option = option_menu(
        menu_title=None,
        options=["Select the analysis","Temporal Analysis", "Geospatial Analysis", "Crime Type Analysis","Severity Analysis","Arrest and Domestic Incident Analysis","Location-Specific Analysis","Repeat Crime Locations"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical")

    if selected_option=="Temporal Analysis":
        st.subheader("1. Crime Trends Over Time: Examining how the number of crimes has changed over the years.")
        crime_rates_over_time()
        st.subheader("2. Crime Trends Over Time (Number of Crimes Per Month)")
        crimes_per_month()
        st.subheader("3. Peak Crime Hours (Number of Crimes Per Hour)")
        crimes_per_hour()
        st.subheader("4. Crime Trends Over Time (Number of Crimes Per Day of the Week)")
        crimes_per_day()
        st.subheader("5. 5. Crime Trends Over Time (Number of Crimes Per Season)")
        crimes_per_season()
    
    if selected_option=="Geospatial Analysis":
        st.subheader("Crime Hotspots Map")
        years = df_crime_final['Year'].unique().tolist()
        years.sort()
        years.insert(0, "All Years")  
        selected_year = st.selectbox("Select Year", years)
        map(selected_year)
        st.subheader("District/Ward Analysis")
        crime_ward()
    
    if selected_option=="Crime Type Analysis":
        st.subheader("Distribution of Crime Types")
        crime_type()
        
    if selected_option=="Severity Analysis":    
        st.subheader("Severity Analysis")
        crime_severity()

    if selected_option=="Arrest and Domestic Incident Analysis":
        st.subheader("Arrest Rates Analysis")
        arrest_rates()
        st.subheader("Domestic VS Non-Domestic")
        domestic()
        detailed_domestic()

    if selected_option=="Location-Specific Analysis":
        st.subheader("Location Description Analysis")
        location_specific()
        st.subheader("Beat and Community Area Analysis")
        beat_community()
    
    if selected_option=="Repeat Crime Locations":
        st.subheader("Repeat Offenders and Recidivism Analysis")
        repeat()

if selected_option=="Modeling":
    st.header("Chicago Crime Incident Predictor")

    st.subheader("Input Features")
    col1,col2=st.columns(2)
    with col1:
        year = st.number_input("Year", min_value=2000, max_value=2024, value=2023)
        month = st.number_input("Month", min_value=1, max_value=12, value=1)
        day = st.number_input("Day", min_value=1, max_value=31, value=1)
    with col2:
        hour = st.number_input("Hour", min_value=0, max_value=23, value=12)
        latitude = st.number_input("Latitude", min_value=41.0, max_value=42.0, value=41.8781)
        longitude = st.number_input("Longitude", min_value=-88.0, max_value=-87.0, value=-87.6298)

    # Prepare the input data
    input_data = np.array([[year, month, day, hour, latitude, longitude]])
    input_data = scaler.transform(input_data)

    # Predict the crime type
    if st.button("Predict Crime Type",use_container_width=True):
        prediction = model.predict(input_data)
        crime_types = dict(enumerate(df_crime_final['Primary Type'].astype('category').cat.categories))
        st.info(f"Predicted Crime Type: {crime_types[prediction[0]]}")

st.markdown("---")
st.markdown("Created by: Kamayani 👩‍💻")

