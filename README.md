# RAG Video Course Search

A retrieval-augmented generation (RAG) pipeline that turns a folder of course videos into a searchable AI teaching assistant. Ask a question in plain language and get pointed to the exact video and timestamp where it's covered — built on Whisper for transcription, local embeddings via Ollama, and an LLM for answer generation.

## How it works

1. **Collect videos** → raw course video files
2. **`video_to_mp3.py`** → extracts audio (ffmpeg)
3. **`mp3_to_json.py`** → transcribes audio into timestamped JSON chunks (Whisper)
4. **`preprocess_json.py`** → merges small chunks into larger, context-rich groups
5. **`merge_chunks.py`** → embeds each chunk and saves as a DataFrame (`embeddings.joblib`)
6. **`process_incoming.py`** → takes a user's question, finds the most relevant chunks via similarity search, and generates a natural-language answer pointing to the right video and timestamp

## Tech stack

- **Python** — pipeline orchestration
- **OpenAI Whisper** — speech-to-text transcription
- **Ollama** (`bge-m3`, `llama3.2`) — local embeddings and LLM inference
- **pandas / scikit-learn / numpy** — data handling and cosine similarity search
- **ffmpeg** — video → audio conversion

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) installed and on your PATH
- [Ollama](https://ollama.com/) running locally, with these models pulled:
  ```
  ollama pull bge-m3
  ollama pull llama3.2
  ```
- Python packages:
  ```
  pip install -r requirements.txt
  ```

## Usage: run it on your own data

**Step 1 — Collect your videos**
Move all your video files into the `videos/` folder.

**Step 2 — Convert to mp3**
```
python video_to_mp3.py
```

**Step 3 — Convert mp3 to JSON (transcription)**
```
python mp3_to_json.py
```
Output saved to `jsons/`.

**Step 4 — Merge chunks**
```
python preprocess_json.py
```
Groups every 5 transcript chunks into one, for better embedding context. Output saved to `newjsons/`.

**Step 5 — Generate embeddings**
```
python merge_chunks.py
```
Embeds every chunk and saves the result as `embeddings.joblib`.

**Step 6 — Ask questions**
```
python process_incoming.py
```
Loads the embeddings, finds the most relevant chunks for your question, and returns an answer with the source video and timestamp.

## Notes

- `videos/`, `audios/`, `jsons/`, `newjsons/`, and `embeddings.joblib` are all generated artifacts, excluded from version control — re-run the pipeline to regenerate them from your own source videos.
- Whisper's `large-v2` model is resource-intensive; a GPU is strongly recommended for `mp3_to_json.py`.

## Possible improvements

- Parameterize hardcoded values (chunk group size, top-k results, model names) via CLI args or a config file.
- Add error handling for missing folders.
- Replace CLI `input()` with a simple web interface.