#importing necessary libraries
import pickle
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# **Load Pickled Model & Data**
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("video_data.pkl", "rb") as f:
    df_new = pickle.load(f)

with open("pickel_matrix.pkl", "rb") as f:  
    X = pickle.load(f)  


#recommendation function

def recommend_videos(title, num_recommendations=10):
    df_new_reset = df_new.reset_index(drop=True)
    
    # Transform search query to TF-IDF
    query_vector = vectorizer.transform([title])

    # Compute similarity with precomputed TF-IDF matrix
    similarities = cosine_similarity(query_vector, X)[0]
    
    # Get top matches
    sorted_videos = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
    
    recommended_videos = []
    for video_idx, score in sorted_videos:
        if score > 0:  # Only return relevant results
            recommended_videos.append(df_new_reset.loc[video_idx, ['title', 'thumbnail_link']].tolist())
        
        if len(recommended_videos) >= num_recommendations:
            break
            
    return recommended_videos

# **Streamlit UI**
st.title(" Video Recommendation System")

search_query = st.text_input("Enter a video title or keywords:", "")

if st.button("Get Recommendations"):
    if search_query:
        recommendations = recommend_videos(search_query)

        if not recommendations:
            st.warning(" No related videos found. Try another search.")
        else:
            st.subheader("Recommended Videos:")
        
        if recommendations:
            cols = st.columns(min(len(recommendations), 4))  # Fixed dynamic column layout
    
        for index, (title, thumbnail) in enumerate(recommendations):
            with cols[index % min(len(recommendations), 4)]:  # Adjust for fewer results
                st.image(thumbnail, use_container_width=True)
              
                st.markdown(f"""
<div style="
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: 3; /* Limit to 3 lines */
    overflow: hidden;
    text-overflow: ellipsis;
    word-wrap: break-word;
    width: 250px; /* Fixed width for uniformity */
    font-size: 14px;
    font-weight: bold;
    line-height: 1.4; /* Ensures consistent spacing */
    text-align: center;
    margin-top: 5px;
    padding: 5px;
    white-space: normal; /* Ensures wrapping */
    height: 4.2em; /* Fixed height based on line-height */
    background-color: #222; /* Optional: Background for uniformity */
    display: flex;
    align-items: center;
    justify-content: center;
">
    {title}
</div>
""", unsafe_allow_html=True)
