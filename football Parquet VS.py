import pandas as pd
import requests
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from datetime import datetime
from fpdf import FPDF
import tempfile

st.set_page_config(layout="wide")

st.sidebar.header("Filters")

# --- Load Data from Google Drive link --- #
@st.cache_data

def load_data():
    file_id = "1IBvy-k0yCDKMynfRTQzXJAoWJpRhFPKk"
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(download_url)
    response.raise_for_status()
    df = pd.read_parquet(BytesIO(response.content))
    df['EVENT_START_TIMESTAMP'] = pd.to_datetime(df['EVENT_START_TIMESTAMP'], errors='coerce')
    return df.dropna(subset=['EVENT_START_TIMESTAMP'])

df = load_data()

# --- Expectancy Options --- #
exp_options = [
    "Favourite Goals", "Underdog Goals", "Total Goals",
    "Favourite Corners", "Underdog Corners", "Total Corners",
    "Favourite Yellow", "Underdog Yellow", "Total Yellow"
]

selected_exp = st.sidebar.multiselect("Select Expectancy Types (up to 6)", exp_options, max_selections=6)

# --- Date Range Filter --- #
min_date = df['EVENT_START_TIMESTAMP'].min().date()
max_date = df['EVENT_START_TIMESTAMP'].max().date()
date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

# --- Favouritism Filter --- #
def label_favouritism(row):
    diff = abs(row['GOAL_EXP_HOME'] - row['GOAL_EXP_AWAY'])
    if diff > 1:
        return "Strong Favourite"
    elif diff > 0.5:
        return "Medium Favourite"
    else:
        return "Slight Favourite"

df['FAVOURITISM_LEVEL'] = df.apply(label_favouritism, axis=1)
fav_filter = st.sidebar.multiselect("Goal Favouritism Level", ["Strong Favourite", "Medium Favourite", "Slight Favourite"], default=["Strong Favourite", "Medium Favourite", "Slight Favourite"])

# --- Scoreline Filter --- #
def label_scoreline(row):
    home_goals = row['GOALS_HOME']
    away_goals = row['GOALS_AWAY']
    fav_is_home = row['GOAL_EXP_HOME'] > row['GOAL_EXP_AWAY']
    if home_goals == away_goals:
        return "Scores Level"
    elif (home_goals > away_goals and fav_is_home) or (away_goals > home_goals and not fav_is_home):
        return "Favourite Winning"
    else:
        return "Underdog Winning"

df['SCORELINE_LABEL'] = df.apply(label_scoreline, axis=1)
scoreline_filter = st.sidebar.multiselect("Goal Scoreline Filter", ["Favourite Winning", "Underdog Winning", "Scores Level"], default=["Favourite Winning", "Underdog Winning", "Scores Level"])

# --- Time Bins --- #
time_bins = [(i, i + 5) for i in range(0, 90, 5)]
time_labels = [f"{start}-{end}" for start, end in time_bins]

def compute_exp_change(df, exp_type):
    exp_cols = {
        "Goals": ("GOAL_EXP_HOME", "GOAL_EXP_AWAY"),
        "Corners": ("CORNERS_EXP_HOME", "CORNERS_EXP_AWAY"),
        "Yellow": ("YELLOW_CARDS_EXP_HOME", "YELLOW_CARDS_EXP_AWAY")
    }

    metric = [key for key in exp_cols if key in exp_type][0]
    col_home, col_away = exp_cols[metric]

    df_sorted = df.sort_values(['SRC_EVENT_ID', 'MINUTES'])
    output = []

    for event_id, group in df_sorted.groupby('SRC_EVENT_ID'):
        base_row = group.loc[group['MINUTES'] == group['MINUTES'].min()].iloc[0]

        base_home = base_row[col_home]
        base_away = base_row[col_away]

        fav_is_home = base_row['GOAL_EXP_HOME'] > base_row['GOAL_EXP_AWAY']

        for _, row in group.iterrows():
            band = next((label for (start, end), label in zip(time_bins, time_labels) if start <= row['MINUTES'] < end), "85-90")

            if "Favourite" in exp_type:
                current = row[col_home] if fav_is_home else row[col_away]
                base = base_home if fav_is_home else base_away
            elif "Underdog" in exp_type:
                current = row[col_home] if not fav_is_home else row[col_away]
                base = base_home if not fav_is_home else base_away
            else:  # Total
                current = row[col_home] + row[col_away]
                base = base_home + base_away

            delta = current - base
            if delta != 0:
                output.append({"Time Band": band, "Change": delta})

    return pd.DataFrame(output)

# --- Filter Data --- #
df_filtered = df[
    (df['EVENT_START_TIMESTAMP'].dt.date >= date_range[0]) &
    (df['EVENT_START_TIMESTAMP'].dt.date <= date_range[1]) &
    (df['FAVOURITISM_LEVEL'].isin(fav_filter)) &
    (df['SCORELINE_LABEL'].isin(scoreline_filter))
]

# --- Generate and Display Graphs --- #
plots = []

if selected_exp:
    n_cols = 2 if len(selected_exp) > 1 else 1
    layout_cols = st.columns(n_cols)

    for i, exp_type in enumerate(selected_exp):
        df_changes = compute_exp_change(df_filtered, exp_type)
        avg_change = df_changes.groupby('Time Band')['Change'].mean().reindex(time_labels, fill_value=0)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(avg_change.index, avg_change.values, marker='o', color='black')
        ax.set_title(f"{exp_type} Expectancy Change\nDate: {date_range[0]} to {date_range[1]} | Fav: {', '.join(fav_filter)} | Scoreline: {', '.join(scoreline_filter)}")
        ax.set_ylabel("Avg Change")
        ax.set_xlabel("Time Band (Minutes)")
        ax.grid(True)

        with layout_cols[i % n_cols]:
            st.pyplot(fig, use_container_width=True)
            plots.append(fig)

    st.markdown("*Favourites are determined using Goal Expectancy at the earliest available minute in each match*")

    if st.button("Download All Charts as PDF"):
        pdf = FPDF()
        for fig in plots:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                fig.savefig(tmpfile.name)
                pdf.add_page()
                pdf.image(tmpfile.name, x=10, y=10, w=190)
        pdf_path = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name
        pdf.output(pdf_path)
        with open(pdf_path, "rb") as f:
            st.download_button("📥 Download PDF", f, file_name="charts.pdf")
else:
    st.warning("Please select at least one expectancy type to display charts.")
