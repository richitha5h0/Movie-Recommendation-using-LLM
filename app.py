from flask import Flask, render_template, request
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

app = Flask(__name__)
print("Starting Flask App ✅")

# Load dataset
df = pd.read_csv('Refined_Movies.csv')

# Clean poster URLs
df['poster'] = df['poster'].fillna('https://dummyimage.com/300x450/000/fff&text=No+Poster')

# Clean year column
df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)

# Clean rating column
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# Clean and standardize languages column
def clean_languages(langs):
    if pd.isna(langs):
        return ''
    return ','.join(lang.strip().lower() for lang in str(langs).split(','))

df['languages'] = df['languages'].astype(str).apply(clean_languages)

# Confirm Telugu movies exist
telugu_movies = df[df['languages'].str.contains('telugu', na=False)]
print(f"Total Telugu movies after cleaning: {len(telugu_movies)}")

# Load model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Finished model loading ✅")

# Precompute embeddings
df['embedding'] = df['plot'].apply(lambda x: model.encode(str(x), convert_to_tensor=True))


@app.route('/')
def intro():
    return render_template('intro.html')


@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/process_login', methods=['POST'])
def process_login():
    username = request.form.get('username')
    password = request.form.get('password')
    if username and password:
        return render_template('success.html')
    else:
        return "Invalid credentials", 401


@app.route('/success', methods=['POST'])
def success():
    return render_template('success.html')


@app.route('/home')
def main_page():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    query = request.form['query']
    query_lower = query.lower()
    print(f"Received query: {query}")

    filtered_df = df.copy()
    filter_applied = False  # New flag to track if any filter was applied

    # Genre filtering
    genres = ['action', 'romance', 'thriller', 'drama', 'comedy', 'horror', 'sci-fi', 'animation', 'adventure']
    for genre in genres:
        if genre in query_lower:
            filtered_df = filtered_df[filtered_df['genres'].str.lower().str.contains(genre, na=False)]
            print(f"Filtered by genre: {genre}. Movies left: {len(filtered_df)}")
            filter_applied = True

    # Language filtering
    languages = ['english', 'hindi', 'telugu', 'tamil', 'malayalam', 'kannada']
    for lang in languages:
        if lang in query_lower:
            filtered_df = filtered_df[filtered_df['languages'].str.contains(lang, na=False)]
            print(f"Filtered by language: {lang}. Movies left: {len(filtered_df)}")
            filter_applied = True

    # Rating filtering
    if 'above' in query_lower and 'rating' in query_lower:
        words = query_lower.split()
        for i in range(len(words)):
            if words[i] == 'above':
                try:
                    min_rating = float(words[i+1])
                    filtered_df = filtered_df[filtered_df['rating'] >= min_rating]
                    print(f"Filtered by rating >= {min_rating}. Movies left: {len(filtered_df)}")
                    filter_applied = True
                except:
                    pass

    # Year filtering
    if 'after' in query_lower:
        words = query_lower.split()
        for i in range(len(words)):
            if words[i] == 'after':
                try:
                    min_year = int(words[i+1])
                    filtered_df = filtered_df[filtered_df['year'] > min_year]
                    print(f"Filtered by year > {min_year}. Movies left: {len(filtered_df)}")
                    filter_applied = True
                except:
                    pass

    # If no relevant movie-related filter applied — stop here
    if not filter_applied:
        print("No movie-related intent detected. No results returned.")
        return render_template('results.html', query=query, recommendations=[])

    if filtered_df.empty:
        return render_template('results.html', query=query, recommendations=[])

    # Otherwise run embedding-based recommendations
    query_embedding = model.encode(query, convert_to_tensor=True)
    embedding_tensor = torch.stack(filtered_df['embedding'].tolist())
    similarities = util.cos_sim(query_embedding, embedding_tensor)
    top_indices = torch.topk(similarities, k=min(10, len(filtered_df))).indices[0].tolist()

    recommendations = []
    for idx in top_indices:
        movie = filtered_df.iloc[idx]
        recommendations.append({
            'title': movie['movie title'],
            'genres': movie['genres'],
            'language': movie['languages'],
            'year': movie['year'],
            'rating': movie['rating'],
            'poster': movie['poster'],
            'plot': movie['plot'],
            'trailer': movie['trailer'] if 'trailer' in movie and pd.notna(movie['trailer']) else None
        })

    return render_template('results.html', query=query, recommendations=recommendations)


if __name__ == '__main__':
    app.run(debug=True)
