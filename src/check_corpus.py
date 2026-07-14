import json, pickle

with open('data/corpus/articles_full.json', encoding='utf-8') as f:
    articles = json.load(f)

with open('data/corpus/chunks.json', encoding='utf-8') as f:
    chunks = json.load(f)

# Pickle files are binary ('rb'), so they do not need an encoding specified
with open('data/corpus/faiss_index.pkl', 'rb') as f:
    index = pickle.load(f)

print('Articles:', len(articles))
print('Chunks:  ', len(chunks))
print('Vectors: ', index['index'].ntotal)
print('In sync: ', len(chunks) == index['index'].ntotal)