### Retrieval
import google.genai as genai
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
import faiss


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "dpp_notes.txt")

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

chunks = text.split("\n\n")

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(embeddings)
question = input("Ask: ")

question_enbedding = model.encode([question])

distances, indices = index.search(question_enbedding, k=1)
best_chunk = chunks[indices[0][0]]

### Argumentation & Generation
load_dotenv()
YOUR_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=YOUR_API_KEY)



context = best_chunk
prompt = f"""
You are a helpful assistant.

Use ONLY the provided context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

response = client.models.generate_content(
    model = "gemini-2.5-flash",
    contents=prompt
)
print(response.text)