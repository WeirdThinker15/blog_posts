from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
import time

print("Model Loading -- Started")
MODEL = SentenceTransformer('all-MiniLM-L6-v2')
print("Model Loading -- Finished")

def search(query, sentences, embeddings, top_k=3):

    # Step 1 : Encode the query
    query_embedding = MODEL.encode([query])

    # Step 2 : Compute Similarity
    scores = cosine_similarity(query_embedding, embeddings)[0]

    # Step 3 : Rank the results
    top_indices = np.argsort(scores)[::-1][:top_k]

    # Step 4 : Return Matches
    results = []
    for index in top_indices:

        results.append((sentences[index], scores[index]))

    return results

def search_space_setup(space_size = 1):

    sentences = [
        "I love machine learning",
        "Artificial intelligence is fascinating",
        "I enjoy studying AI",
        "The weather is nice today",
        "It is raining outside"
    ]

    search_space = sentences * space_size  # Increase dataset size
    return search_space

if __name__ == '__main__':

    for idx in [1,10,100,1000,10000,100000, 1000000]:
        print(f"Search Space Size: {idx * 5}")
        search_space = search_space_setup(idx)

        embeddings = MODEL.encode(search_space)

        query = 'I enjoy learning AI'

        start = time.time()
        results = search(query, search_space, embeddings)
        end = time.time()

        print(f"Search time: {end - start:.4f} seconds")

        for sentence,score in results:
            print(f"{sentence}: {score:.4f}")


