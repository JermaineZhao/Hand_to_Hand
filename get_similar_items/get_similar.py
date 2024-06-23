import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import transformers
from transformers import BertTokenizer, BertModel
import torch

# Load the CSV file
file_path = 'amazon.csv'
df = pd.read_csv(file_path)

# Remove currency symbol and commas, and convert to float for actual_price
df['actual_price'] = df['actual_price'].str.replace('₹', '').str.replace(',', '').astype(float)

# Standardize the price
scaler = StandardScaler()
df['price_scaled'] = scaler.fit_transform(df[['actual_price']])

# One-Hot Encoding for category
encoder = OneHotEncoder()
categories_encoded = encoder.fit_transform(df[['category']]).toarray()

# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to encode text using BERT
def encode_text(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Encode product_name and about_product
names_embeddings = np.vstack([encode_text(name) for name in df['product_name']])
descriptions_embeddings = np.vstack([encode_text(description) for description in df['about_product']])

# Combine all features
features = np.hstack([df[['price_scaled']].values, categories_encoded, names_embeddings, descriptions_embeddings])

# Step 2: Dimensionality reduction using PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
embeddings = pca.fit_transform(features)

# Step 3: Calculate cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(embeddings)

def get_recommendations(product_id, top_k=10):
    sim_scores = similarity_matrix[product_id]
    top_indices = np.argsort(sim_scores)[-top_k-1:-1][::-1]
    return top_indices

# Example: Get recommendations for product ID 0
recommended_products = get_recommendations(0)

# Step 4: Display recommended products
recommended_df = df.iloc[recommended_products][['product_id', 'product_name', 'category', 'actual_price']]
import ace_tools as tools; tools.display_dataframe_to_user(name="Recommended Products", dataframe=recommended_df)
recommended_df