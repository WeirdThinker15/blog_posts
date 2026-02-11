from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

if __name__ == '__main__':
    model = SentenceTransformer("all-MiniLM-L6-v2")

    sentences = [
        "I love machine learning",
        "Artificial intelligence is fascinating",
        "I enjoy studying AI",
        "The weather is nice today",
        "It is raining outside"
    ]

    embeddings = model.encode(sentences)

    print(type(embeddings))
    print(len(embeddings))  # Number of sentences
    print(len(embeddings[0]))
    print(embeddings[0][:10])  # First 10 values

    similarity_matrix = cosine_similarity(embeddings)
    print("--------- Similarity Matrix ---------")
    print(similarity_matrix)

    query = "I like learning artificial intelligence"
    query_embedding = model.encode([query])
    print("---------------- Scores ------------")

    scores = cosine_similarity(query_embedding, embeddings)
    print(scores)

    print("------ Ranking --------------")

    import numpy as np

    top_indices = np.argsort(scores[0])[::-1]

    for idx in top_indices:
        print(sentences[idx], scores[0][idx])


