import requests
import os
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib


def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    if r.status_code != 200:
        print("STATUS:", r.status_code)
        print("BODY:", r.text)
    r.raise_for_status()
    return r.json()["embeddings"]


def batch_embed(texts, batch_size=32):
    """Send texts to Ollama in small batches to avoid overloading the runner."""
    all_embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        print(f"  Embedding batch {start}-{start+len(batch)} of {len(texts)}")
        embeddings = create_embedding(batch)
        all_embeddings.extend(embeddings)
    return all_embeddings


jsons = os.listdir("newjsons")
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"newjsons/{json_file}") as f:
        content = json.load(f)
    print(f"Creating Embeddings for {json_file}")

    raw_chunks = content['chunks']
    print("Total chunks:", len(raw_chunks))

    clean_chunks = []
    for i, c in enumerate(raw_chunks):
        text = c.get('text')
        if not isinstance(text, str) or not text.strip():
            print(f"  SKIPPING bad chunk at index {i}: {repr(text)}")
            continue
        clean_chunks.append(c)
        #if len(clean_chunks) == 4:   # stop collecting once we have 4 valid chunks
          # break

    if not clean_chunks:
        print(f"  No valid chunks in {json_file}, skipping file.")
        continue

    texts = [c['text'] for c in clean_chunks]
    embeddings = batch_embed(texts, batch_size=32)

    for i, chunk in enumerate(clean_chunks):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk)
    # break  # stop processing after the first file for testing  

df = pd.DataFrame.from_records(my_dicts)
#Save this DateFrame
joblib.dump(df, 'embeddings.joblib')




