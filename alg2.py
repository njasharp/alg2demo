import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Function to create the spider chart
def create_spider_chart(values):
    num_vars = len(values)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    categories = ['Engagement', 'Retention', 'Monetization', 'User Acquisition', 'Game Performance', 'Player Satisfaction']

    # Create the spider chart
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.fill(angles, values, color='green', alpha=0.25)
    ax.plot(angles, values, color='green', linewidth=2)

    # Add labels to each axis
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    # Set the range of the radial axis
    ax.set_ylim(0, 10)

    return fig

# Streamlit Sidebar Sliders
st.sidebar.header("Adjust Mobile Game Metrics (1-10)")
engagement = st.sidebar.slider('Engagement', 1, 10, 5)
retention = st.sidebar.slider('Retention', 1, 10, 5)
monetization = st.sidebar.slider('Monetization', 1, 10, 5)
user_acquisition = st.sidebar.slider('User Acquisition', 1, 10, 5)
game_performance = st.sidebar.slider('Game Performance', 1, 10, 5)
player_satisfaction = st.sidebar.slider('Player Satisfaction', 1, 10, 5)

# Values from sliders
values = [engagement, retention, monetization, user_acquisition, game_performance, player_satisfaction]

# Display the spider chart
st.image("p1.png")
st.title("Spider Chart")
fig = create_spider_chart(values)
st.pyplot(fig)
st.info("build by dw 9-17-24")