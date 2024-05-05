from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import streamlit as st
import pandas as pd
from data_analysis import *


# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# CSV Data Loader
@st.cache_data
def load_csv_data(file_path):
    if file_path is not None:
        # checking if database exists
        # reading the file
        csv_loader = CSVLoader(file_path)
        # Loading the file
        loaded_data = csv_loader.load()
        # loading the data to FAISS
        db = FAISS.from_documents(loaded_data, embedding_function)
        # saving to disk
        db.save_local("file_path")
        # now loading with pandas
        df = pd.read_csv(file_path)
        return df

# loading the data from db
def load_data_from_db():
    # loading the data
    db = FAISS.load_local("faiss_index", embedding_function)
    return db


# creating a function that tekes in the informations and create a structured text format of it.
def create_general_text(shape, columns, missing, duplicated):
    text = f'The data has {shape[0]} rows and {shape[1]} columns. The columns in the data are {columns}. The data has {missing} missing values and {duplicated} duplicated values.'
    return text

# converting to text format 
def create_eda_text(unique, data_types, numerical, categorical, summary, correlation, outliers):
    text = f"The unique values in the data are {unique}. The data types of the columns are: {data_types}. The numerical columns in the data are {numerical}. The categorical columns in the data are {categorical}. The summary statistics of the data are {summary}. The correlation between the numerical columns is {correlation}. The percentage of outliers in the numerical columns are {outliers}."
    return text

# creating text for the data cleaner
def create_cleaner_text(columns, missing, duplicated, numerical, outliers):
    text = f"These are the columns in the data {columns}.\nThe total missing values in the data is {missing}.\nThere are {duplicated} duplicates in the data.\nThe numerical features in the data include {numerical}.\nThe table below shows the outliers in the data: \n{outliers}"
    return text

# Setting up the things I want in general analysis
# Data Shape, columns in the data ,sum of missing values, sum of duplicated values
def general_analysis(data):
    shape = data_shape(data)
    columns = data_columns(data)
    missing = sum_missing_values(data)
    duplicated = sum_duplicates(data)
    return shape, columns, missing, duplicated

# using a function to get EDA
def EDA_analysis(data):
    unique = unique_values(data)
    data_types = data_info(data)
    numerical = numerical_columns(data)
    categorical = categorical_columns(data)
    summary = summary_stats(data)
    correlation = correlation_check(data, numerical)
    outliers = check_outliers(data, numerical)
    return unique, data_types, numerical, categorical, summary, correlation, outliers

# creating a cleaning assistant
def data_cleaner_ass(data):
    columns = data.columns
    missing = sum_missing_values(data)
    duplicated = sum_duplicates(data)
    numerical = numerical_columns(data)
    outliers = check_outliers(data, numerical)
    return columns, missing, duplicated, numerical, outliers