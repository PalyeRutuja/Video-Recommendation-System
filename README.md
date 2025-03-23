# Video Recommendation System

This repository contains a video recommendation system that utilizes text-based features from video metadata (such as title, description, tags, and category) to recommend similar videos based on a user query. The system leverages techniques like TF-IDF vectorization and cosine similarity to generate recommendations and is built using Python with Streamlit for the front-end interface.

## Features

- **Video Recommendations:** Recommend similar videos based on a given video title or keyword.
- **Text-based Search:** Users can search for videos using keywords or titles, and the system will return a list of recommended videos.
- **Streamlit Interface:** Interactive and user-friendly interface to input search queries and view results.
- **Data Visualization:** Visualizations of video trends, categories, and tags.
- **Clustering:** K-Means clustering to group videos based on content similarity.
- **Dimensionality Reduction:** PCA to reduce the dimensionality of the data for better clustering visualization.

## Tech Stack

**Python Libraries:**

- **Streamlit:** For the user interface.
- **sklearn:** For machine learning models (TF-IDF, K-Means, cosine similarity).
- **pandas** and **numpy:** For data manipulation.
- **matplotlib** and **seaborn:** For data visualization.
- **pickle:** For saving and loading models and data.
- **wordcloud:** For generating word clouds.
- **datetime:** For handling time data.
