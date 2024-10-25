
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.title("Dataset Visualization App")

# Function to load dataset
def load_dataset(file):
    try:
        df = pd.read_csv(file)
        st.success("Dataset loaded successfully.")
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Function to visualize histogram using Plotly
def visualize_histogram(df, column):
    fig = px.histogram(df, x=column, marginal="box", nbins=30, title=f'Histogram of {column}')
    st.plotly_chart(fig)

# Function to visualize scatter plot using Plotly
def visualize_scatter_plot(df, column1, column2):
    fig = px.scatter(df, x=column1, y=column2, title=f'Scatter Plot between {column1} and {column2}')
    st.plotly_chart(fig)

# Function to visualize line plot using Plotly
def visualize_line_plot(df, column1, column2):
    fig = px.line(df, x=column1, y=column2, title=f'Line Plot between {column1} and {column2}')
    st.plotly_chart(fig)

# Function to visualize box plot using Plotly
def visualize_box_plot(df, column):
    fig = px.box(df, y=column, title=f'Box Plot of {column}')
    st.plotly_chart(fig)

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = load_dataset(uploaded_file)
    
    if df is not None:
        st.write("Available columns:", df.columns.tolist())

        viz_option = st.selectbox("Choose a visualization", ["Histogram", "Scatter Plot", "Line Plot", "Box Plot"])
        
        if viz_option == "Histogram":
            column = st.selectbox("Choose a column for histogram", df.columns)
            if column:
                visualize_histogram(df, column)
        
        elif viz_option == "Scatter Plot":
            column1 = st.selectbox("Choose X-axis for scatter plot", df.columns)
            column2 = st.selectbox("Choose Y-axis for scatter plot", df.columns)
            if column1 and column2:
                visualize_scatter_plot(df, column1, column2)
        
        elif viz_option == "Line Plot":
            column1 = st.selectbox("Choose X-axis for line plot", df.columns)
            column2 = st.selectbox("Choose Y-axis for line plot", df.columns)
            if column1 and column2:
                visualize_line_plot(df, column1, column2)
        
        elif viz_option == "Box Plot":
            column = st.selectbox("Choose a column for box plot", df.columns)
            if column:
                visualize_box_plot(df, column)
