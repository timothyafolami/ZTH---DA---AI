import streamlit as st
import pandas as pd
import tempfile
from utils import *
from ai_helper import *

# Set the page title and configuration
st.set_page_config(
    page_title="Data Analysis App",
    page_icon="🔍",
    layout="wide"
)

# Create a main page title
st.title("Data Analysis App")
st.markdown("Welcome to the Data Analysis App!")

# Add a main content section
st.write("Upload your data using the file uploader on the sidebar, and start exploring your data!")
# Create a sidebar with a file uploader
st.sidebar.title("Upload Your Data")
file_uploader = st.sidebar.file_uploader(label="Upload your CSV file", type=["csv"])
# Create a sidebar with a dropdown menu
options = ["General Analysis", "EDA", "Statistical Analysis", "Data Cleaning", "More Visualizations", "Chat"]
selected_option = st.sidebar.selectbox("Select an option:", options)

if file_uploader :
   #use tempfile because CSVLoader only accepts a file_path
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file_uploader.getvalue())
        tmp_file_path = tmp_file.name
    with st.spinner("Loading data..."):
        # load the data
        df = pd.read_csv(tmp_file_path)
        st.dataframe(df)
        st.balloons()
        st.write("Data loaded successfully!")

        # creating Data Exploration section
        if selected_option == "General Analysis":
                # Perform general analysis here
                st.markdown("## General Analysis")
                st.write("This will perform general analysis on the uploaded data.")
                # Get general analysis results
                shape, columns, missing, duplicated = general_analysis(df)
                # Display the general analysis results in columns
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write("### Data Shape")
                    st.write(f"Rows: {shape[0]}")
                    st.write(f"Columns: {shape[1]}")
                with col2:
                    st.write("### Columns in the Data")
                    st.write(columns)
                with col3:
                    st.write("### Sum of Missing Values")
                    st.write(missing)
                with col4:
                    st.write("### Sum of Duplicated Values")
                    st.write(duplicated)
                # Create a structured text format of the general analysis results   
                text = create_general_text(shape, columns, missing, duplicated)
                # getting ai comment
                comment = get_comment(text)
                st.write("### AI Comment")
                st.write(comment)



        elif selected_option == "EDA":
            # Perform exploratory data analysis here
            st.title("Exploratory Data Analysis")
            st.write("This will perform exploratory data analysis on the uploaded data.")
            # Get EDA results
            unique, data_types, numerical, categorical, summary, correlation, outliers = EDA_analysis(df)
            # Display the EDA results
            st.write("### Unique Values")
            st.write(unique)
            st.write("### Data Types")
            st.write(data_types)
            st.write("### Numerical Columns")
            st.write(numerical)
            st.write("### Categorical Columns")
            st.write(categorical)
            st.write("### Summary Statistics")
            st.write(summary)
            st.write("### Correlation Matrix")
            st.write(correlation)
            st.write("### Outliers")
            st.write(outliers)
            # comverting to text
            text = create_eda_text(unique, data_types, numerical, categorical, summary, correlation, outliers)
            # getting ai comment
            comment = get_comment(text)
            st.write("### AI Comment")
            st.write(comment)
            

        elif selected_option == "Statistical Analysis":
            # Perform statistical analysis here
            st.title("Statistical Analysis")
            st.write("This will perform statistical analysis on the uploaded data.")

        elif selected_option == "Data Cleaning":
            # Perform data cleaning here
            st.title("Data Cleaning")
            st.write("This will perform data cleaning on the uploaded data.")
            # Making some discussion about data cleaning
            question = "Data Cleaning"
            comment = get_topic_answer(question)
            st.write("### AI Comment")
            st.write(comment)
            # now checking if the data needs to be cleaned
            columns, missing, duplicated, numerical, outliers = data_cleaner_ass(df)
            # creating text for this:
            text = create_cleaner_text(columns, missing, duplicated, numerical, outliers)
            # passing to ai to get answer
            recommendation = cleaner_rec_ai(text)
            st.write("### AI Cleaning Recommendation")
            st.write(recommendation)

        elif selected_option == "More Visualizations":
            # Create more visualizations here
            st.title("More Visualizations")
            st.write("This will create more visualizations for the uploaded data.")
