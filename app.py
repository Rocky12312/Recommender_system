import io
import re
import os
import time
#import nltk
#import pickle
import string
import base64
import joblib
import zipfile
#import logging
import warnings
import numpy as np
import pandas as pd
from flask import jsonify
#from nltk.corpus import stopwords
#from healthcheck import HealthCheck
from flask_bootstrap import Bootstrap
from flask import Flask, make_response, request, render_template, url_for

warnings.filterwarnings("ignore")

app = Flask(__name__)

#logging.basicConfig(filename="flask.log", level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s %(threadName)s:%(message)s")
#logging.info("All useful files loaded")


#health = HealthCheck(app, "/hcheck")


#def app_ab():
    #return True, "i am good"


#health.add_check(app_ab)

joblib_file1 = "Indices.pkl"
joblib_file2 = "Cosine_similarity.pkl"

indices = joblib.load(joblib_file1)
cosine_similarity = joblib.load(joblib_file2)

def zipFiles(file_List, name_list):
    outfile = io.BytesIO()
    with zipfile.ZipFile(outfile, 'w') as zf:
        for name, data in zip(name_list, file_List):
            zf.writestr(name, data.to_csv())
    return outfile.getvalue()

Bootstrap(app)
@app.route('/')
def index():
    movies_titles = pd.read_csv("data/movies.csv")
    movies_titles.sort_values('title',inplace=True)
    movies_list = movies_titles['title'].values.tolist()
    return render_template('index.html',movies = movies_list)


@app.route('/predict', methods=['POST'])
#Now as we have got the similarity matrix our final task is going to be creating a function that gonn'a be taking a movie name as a input and will output a list of similar movies that can be recommended to the users# Function that takes in movie title as input and outputs most similar movies
def predict():
    df = pd.read_csv("data/movies.csv")
    df_neutral = pd.read_csv("data/Neutral.csv")
    #Taking movie name
    argument = request.form['choice']
    #Index of the movie that matches the title
    idx = indices[argument]
    #Getting the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_similarity[idx]))
    #Sorting the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    #Getting the scores of the 20 most similar movies
    sim_scores = sim_scores[1:20]
    #Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    #Return the top 20 most similar movies
    movies = pd.DataFrame()
    #Retrieving movies
    movies = df.iloc[movie_indices]
    #Sorting the recommended movies on basis of average vote
    movies = movies.sort_values(by=['vote_average'], ascending=False)
    movies = movies[["title"]]
    #Resetting the index
    movies.reset_index(drop=True,inplace=True)
    if len(movies)<2:
        movies = pd.DataFrame()
        movies = df_neutral
    #return render_template('view.html',tables=[movies.to_html(classes='recommend')],titles = ['Recommendations'])
    return render_template("view.html", column_names=movies.columns.values, row_data=list(movies.values.tolist()), zip=zip)

if __name__ == '__main__':
	app.run(debug=True)
