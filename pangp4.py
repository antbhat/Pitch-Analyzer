import os
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
import google.generativeai as genai
import time
from typing import List

# Set your API keys
GOOGLE_API_KEY = ""  # Replace with your Gemini API key
genai.configure(api_key=GOOGLE_API_KEY)

#  Define paths
PITCH_DECKS_DIR = Path("/Users/akshubhat/Downloads/All_PAN_Data/Training Data/Pitch Deck")
INSTRUCTION_DOCS_DIR = Path("/Users/akshubhat/Downloads/All_PAN_Data/Overview")

# Load instruction documents (used for retrieval context)
instruction_docs: List[str] = []
for file in INSTRUCTION_DOCS_DIR.glob("*.pdf"):
    loader = PyPDFLoader(str(file))
    for doc in loader.load():
        instruction_docs.append(doc.page_content)

# Initialize Gemini model
#  Use a general model name.  The specific model can be set in the generate_content call.
model = genai.GenerativeModel("gemini-1.5-flash")

# Function to generate embeddings with Gemini
def generate_gemini_embeddings(texts: List[str]) -> List[List[float]]:
    """Generates embeddings for a list of texts using Gemini.

    Args:
        texts: A list of text strings to embed.

    Returns:
        A list of embeddings, where each embedding is a list of floats.
    """
    embeddings = []
    for text in texts:
        try:
            response = model.generate_embeddings(content=text)
            embeddings.append(response.embedding)
        except Exception as e:
            print(f"Error generating embedding for text: {text}")
            print(f"Error: {e}")
            embeddings.append([])
    return embeddings

# 🔎 Embed instruction documents
instruction_embeddings = generate_gemini_embeddings(instruction_docs)

# 🔎 Index instruction embeddings (FAISS remains suitable for vector search)
import faiss
import numpy as np

# Convert embeddings to numpy array
dimension = len(instruction_embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(instruction_embeddings).astype(np.float32))


# RAG function using Gemini
def get_gemini_response(query: str, context_texts: List[str], index: faiss.Index, instruction_docs: List[str]) -> str:
    """Retrieves relevant documents and generates a response using Gemini.

    Args:
        query: The user query.
        context_texts: A list of text strings representing the document content.
        index: A FAISS index containing the embeddings of the documents.
        instruction_docs: List of instruction documents

    Returns:
        The generated response from Gemini.
    """
    # 1. Generate embeddings for the query
    query_embedding = generate_gemini_embeddings([query])[0]

    # 2. Find the most similar documents
    distances, indices = index.search(np.array([query_embedding]).astype(np.float32), 3)

    # 3. Extract the context
    context = ""
    for i in indices[0]:
        context += instruction_docs[i] + "\n"

    # prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    # response = model.generate_content(contents=prompt)
    # Construct the message as a dictionary
    messages = [{"role": "user", "parts": [f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"]}]
    response = model.generate_content(
        #model="gemini-pro",  # Specify the model here
        contents=messages,
    )
    return response.text



# Query template
def score_pitch_deck(text: str, metric: str, index: faiss.Index, instruction_docs: List[str]) -> str:
    question = f"Based on the PAN Empowerment Index rubric, how would you rate the following company's {metric}? Respond on a scale from 1 to 10 with a 1-sentence justification. Text:\n{text}"
    return get_gemini_response(question, instruction_docs, index, instruction_docs)


# Process each pitch deck
results = []
for pitch_pdf in PITCH_DECKS_DIR.glob("*.pdf"):
    loader = PyPDFLoader(str(pitch_pdf))
    pitch_texts = loader.load()
    full_text = " ".join([page.page_content for page in pitch_texts])

    # Score each metric
    purpose_score = score_pitch_deck(full_text, "Purpose", index, instruction_docs).split(".")
    time.sleep(5)
    transformation_score = score_pitch_deck(full_text, "Transformation", index, instruction_docs).split(".")
    time.sleep(5)
    performance_score = score_pitch_deck(full_text, "Performance", index, instruction_docs).split(".")
    time.sleep(5)

    results.append(
        {
            "File Name": pitch_pdf.name,
            "Purpose Score": purpose_score[0],
            "Purpose Explanation": purpose_score[1],
            "Transformation Score": transformation_score[0],
            "Transformation Explanation": transformation_score[1],
            "Performance Score": performance_score[0],
            "Performance Explanation": performance_score[1],
        }
    )

    print(results)


# Output results
import pandas as pd

df = pd.DataFrame(results)
df.to_csv("empowerment_scores.csv", index=False)
print("✅ Empowerment scores saved to empowerment_scores.csv")



