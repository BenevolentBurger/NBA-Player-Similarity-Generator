import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from nba_api.stats.endpoints import leaguedashplayerstats
import tkinter as tk
from tkinter import ttk

# Function to load the NBA player data and scale the feature values
def load_data():
    # Get player stats from the NBA API
    player_stats = leaguedashplayerstats.LeagueDashPlayerStats(season='2020-21').get_data_frames()[0]
    features = ['PLAYER_NAME', 'PTS', 'REB', 'AST', 'BLK', 'STL']
    data = player_stats[features]
    data['PLAYER_NAME'] = data['PLAYER_NAME'].str.title()

     # Scale the data using StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.iloc[:, 1:])
    
    return data, scaled_data

# Function to build a Nearest Neighbors model with the scaled data
def build_model(scaled_data):
    model = NearestNeighbors(n_neighbors=4, metric='euclidean')
    model.fit(scaled_data)
    return model

# Function to find the 3 most similar players based on the user's input
def find_similar_players(player_name, data, model):
    # Convert player_name to title case to make it case-insensitive
    player_name = player_name.title()

    # Find the index of the player in the data
    player_index = data[data['PLAYER_NAME'] == player_name].index[0]

    # Get the nearest neighbors of the player
    _, indices = model.kneighbors([scaled_data[player_index]])

    # Extract the 3 most similar players (excluding the first index, which is the input player)
    similar_players = data.iloc[indices[0][1:]]
    
    return similar_players

# Function to handle the 'Find Similar Players' button click
def on_find_similar_players():
    # Get the input player name from the entry field
    player_name = player_name_entry.get()

    # Find the similar players
    similar_players = find_similar_players(player_name, data, model)

    # Create a formatted string with the similar players' names
    result_text = "The 3 most similar players to {} are:\n".format(player_name)
    for idx, player in similar_players.iterrows():
        result_text += "- {}\n".format(player['PLAYER_NAME'])

    # Update the result_label with the formatted string
    result_label.config(text=result_text)

# Function to handle window resizing
def on_resize(event):
    result_label.config(wraplength=event.width - 20)

# Create the main window and configure its properties
root = tk.Tk()
root.title("NBA Player Similarity Finder")

# Create and add widgets to the main window
frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
frame.columnconfigure(0, weight=1)

player_name_label = ttk.Label(frame, text="Enter the name of an NBA player:")
player_name_label.grid(row=0, column=0, sticky=tk.W)

player_name_entry = ttk.Entry(frame, width=25)
player_name_entry.grid(row=1, column=0, sticky=tk.W)

find_button = ttk.Button(frame, text="Find Similar Players", command=on_find_similar_players)
find_button.grid(row=2, column=0, sticky=tk.W, pady=(10, 0))

result_label = ttk.Label(frame)
result_label.grid(row=3, column=0, sticky=(tk.W, tk.E), padx=10)

root.bind("<Configure>", on_resize)

data, scaled_data = load_data()
model = build_model(scaled_data)

root.mainloop()
