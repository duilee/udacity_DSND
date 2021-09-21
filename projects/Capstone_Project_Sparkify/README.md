# Project Overview

Medium post can be found [here](https://duilee.medium.com/sparkify-project-predicting-user-churn-97cf1b1e2fdf)

This project deals with mini subset of full 12GB log data for our fictional music streaming app, sparkify.
The goal of the project is to predict churn for the user.

## Essential skills
- Load large datasets into Spark and manipulate them using Spark SQL and Spark Dataframes
- Use the machine learning APIs within Spark ML to build and tune models

For each user, I have extracted
- Thumbs down count
- Thumbs up count
- error count 
- add to playlist count
- add friend count
- total song

With models of Logistic Regression, Random Forest, Gradient Boosting Forest, Logistic Regression turned out with the best performance with Thumbs down count having the most relevance.

## Techniques used
- PySpark SQL, ML
- Pandas
- NumPy
