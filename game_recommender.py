import pandas as pd
import streamlit as st

#importing necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("games_metadata_5k.csv")
#inspecting the dataset
#print(df.head())
#print(df.shape)
#print(df.columns)
#print(df.info())

# Selecting the columns we need
df = df[["name", "description", "genres", "platforms"]]

# Filling missing values
df["genres"] = df["genres"].fillna("")
df["platforms"] = df["platforms"].fillna("")
df["description"] = df["description"].fillna("")

#checking for missing values
#print(df.head())
#print(df.isnull().sum())

# Combine features
df["features"] = (
    df["genres"] + " " +
    df["platforms"] + " " +
    df["description"]
)

# Convert text into numbers
vectorizer = TfidfVectorizer()
feature_matrix = vectorizer.fit_transform(df["features"])

# Calculate similarity
similarity = cosine_similarity(feature_matrix)

# Recommendation function
def recommend_game(game_name):

    index = df[df["name"] == game_name].index[0]

    similarity_scores = list(enumerate(similarity[index]))

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for i in similarity_scores[1:6]:

        game_index = i[0]
        score = i[1]

        recommendations.append(
            (
                df.iloc[game_index]["name"],
                round(score * 100, 2)
            )
        )

    return recommendations

# Test
#recommend_game("The Witcher 3: Wild Hunt")
#recommend_game("Portal 2")
#recommend_game("Grand Theft Auto V")

# Streamlit application
st.title("🎮 Game Recommendation System")

game = st.selectbox(
    "Select a game",
    df["name"]
)

if st.button("Recommend"):
    recommendations = recommend_game(game)
    st.write("### Recommended Games")
    for name, score in recommendations:
        st.write(
            name,
            "->",
            score,
            "% similar"
        )
