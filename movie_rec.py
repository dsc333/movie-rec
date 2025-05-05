'''
Movie recommender that applies collaborative filtering.
Uses MovieLens data of 100,000 reviews by 610 users (mini dataset).
User may search movies in the dataset and rate them.  Several ratings
are needed before the system can make recommendations.

Possible improvements / bug fixes
1) Give the user the option to get recommendations by movie genre or year
    (year is attached to each film name).
2) Currently the system makes a recommendation even if the user has only
    ratings a small number of movies.  Require that a given number of
    ratings be made before a recommendation is possible.
3) The current algorithm recommends movies based on the closest 20 users
   in the dataset.  What if the correlation values are very low?   It might
   be a good idea to include a cutoff of correlation value that will be
   inluded for the purpose of recommendation.  If the threshold is not
   met, the user would be required to rate more movies.
4) The system recommends the 10 movies with the highest average rating
   by user in the dataset, but it doesn't show the predicted rating for
   the user.  Compute the predicted rating and display it (this will require
   using the original (non-centered) ratings.
'''
import streamlit as st
import pandas as pd
import numpy as np
from scipy import spatial
from streamlit_star_rating import st_star_rating

# load any previously saved ratings from {name}_ratings.csv -- ratings
# returned are mean centered.
def load_user_ratings(name):
    try:
        user_ratings = pd.read_csv(f'{name}_ratings.csv')
    except:
        user_ratings = pd.DataFrame(columns = ['Title', 'Rating'])
        user_ratings['Rating'] = user_ratings['Rating'].astype('int32')

        mean_rating = user_ratings['Rating'].mean()
        user_ratings['Rating'] = user_ratings['Rating'].sub(mean_rating, axis='rows')

    return user_ratings


# Return MovieLens data as two dataframes: utility (mean-centered utility matrix)
@st.cache_data
def load_movielens_data():
    utility = pd.read_csv('utility_matrix.csv', index_col = 0)
    movies = pd.read_csv('movies.csv')
    utility = utility.sub(utility.mean(axis='columns'), axis = 'rows')
    utility = utility.fillna(value=0)

    return utility, movies

def recommend_movies(user_ratings, utility, movies):
    MAX_MOVIE_ID = 14076
    MAX_USER_ID = 610
    # Create a series to store user ratings, indexed by movie ID. This has to
    # match the utility matrix in order to compute correlations.
    user_ratings_row = pd.Series([0 for i in range(MAX_MOVIE_ID)], index=utility.columns)
    movies_watched = user_ratings['Title'].values
    for movie in movies_watched:
        movie_id = movies.loc[movies['title'] == movie, ['movieId']].values[0][0]
        movie_rating = user_ratings.loc[user_ratings['Title'] == movie, ['Rating']].values[0][0]
        user_ratings_row[str(movie_id)] = movie_rating

    # Compute the correlation of each of the 610 raters in the dataset with
    # the user's rating and store them in dictionary correlations.
    correlations = {}
    for i in range(MAX_USER_ID):
        x = utility.iloc[i]
        y = user_ratings_row
        # Since data is centered, we can just compute the cosine similarity
        sim = 1 - spatial.distance.cosine(x, y)
        correlations[i] = sim

    sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

    # Get indices of closest 20 users (i.e. highest correlation value)
    relevant_idx = []
    for tup in sorted_corr[:20]:
        relevant_idx.append(tup[0])

    # Create a utility matrix of the closest users
    closests_users = utility.loc[relevant_idx, :]

    # Compute column average not including zeros
    closests_users = closests_users.replace(0, np.nan)
    # Remove all movies that were not rated (nan rating)
    closests_users.dropna(axis = 1, how='all', inplace=True)
    # Compute the mean ratings of remaining movies
    means = closests_users.mean(axis='rows')
    # Create a series of rating averages indexed by movie ID
    means_series = pd.Series(means, index=closests_users.columns)
    # Sort means in descending order
    means_series.sort_values(ascending = False, inplace=True)

    # Get the movie ID's with the 10 highest means
    movie_ids_rec = list(means_series.index)[:10]
    # Lookup the movie title for each of the IDs and display.
    for movie_id in movie_ids_rec[:10]:
        title = movies.loc[movies['movieId'] == int(movie_id), ['title']].values[0][0]
        st.write(title)

def main():
    st.title('Movie recommender')

    data_load_state = st.text('Loading data...')
    utility, movies = load_movielens_data()

    name = st.text_input('Your name: ')
    user_ratings = load_user_ratings(name)

    st.subheader('Movie search')
    movie_name = st.text_input('Search for a movie by title: ').strip()
    if movie_name:
        st.write('Movie titles containing "{}"'.format(movie_name))
        results = movies[movies['title'].str.contains(movie_name, case=False)]['title']
        if len(results) > 0:
            st.write(results)
        else:
            st.write("No match found.")
    st.markdown('---')


    st.subheader("Rate some movies you've seen")
    with st.form("my_form"):
        movie_name = st.text_input(
            'Movie title (search by title first and copy and paste result)').strip()
        movie_rating = st_star_rating("Rating", maxValue=5, defaultValue=3, key="rating")
        submitted = st.form_submit_button("Submit")
        if submitted and movie_name and movie_rating:
            if movie_name in user_ratings['Title'].values:
                user_ratings.loc[user_ratings['Title']==movie_name, 'Rating'] = int(movie_rating)
            else:
                new_entry = [movie_name, int(movie_rating)]
                user_ratings.loc[len(user_ratings)] = new_entry

        user_ratings.to_csv(f'{name}_ratings.csv', index=False)

    if name:
        st.text(f"{name}'s ratings")
    else:
        st.text("Your ratings")
    if user_ratings.shape[0] > 0:
            st.write(user_ratings)

    st.markdown('---')
    st.subheader("Personalize Recommendations")
    if st.button('Recommend movies for me'):
        data_load_state = st.text('Personalizing your recommendations')
        recommended = recommend_movies(user_ratings, utility, movies)

main()