import subprocess
import requests
import time
import json
import faiss
import numpy as np
from langdetect import detect
from sentence_transformers import SentenceTransformer
from googletrans import Translator
import PyPDF2

# Map of supported languages
LANG_CODE_MAP = {
    'hi': 'Hindi', 'te': 'Telugu', 'ta': 'Tamil', 'gu': 'Gujarati',
    'kn': 'Kannada', 'mr': 'Marathi', 'pa': 'Punjabi', 'en': 'English',
}

EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
translator = Translator()

# Load and chunk PDF text
def load_pdf_chunks(file_path, chunk_size=300):
    reader = PyPDF2.PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text.strip() + "\n"
    chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
    return chunks

# Embed and index PDF chunks
def embed_chunks(chunks):
    embeddings = EMBED_MODEL.encode(chunks)
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, chunks

# Search top-k relevant chunks
def search_chunks(index, chunks, query, k=1):
    query_embedding = EMBED_MODEL.encode([query])
    D, I = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in I[0]]

# Detect user's language
def detect_language(text):
    try:
        code = detect(text)
        lang_name = LANG_CODE_MAP.get(code)
        if not lang_name:
            print(f"[WARN] Unknown language '{code}' — defaulting to English")
            return 'en', 'English'
        return code, lang_name
    except:
        return 'en', 'English'

# Translate to and from English
def translate_to_english(text):
    try:
        return translator.translate(text, dest='en').text
    except:
        return text

def translate_from_english(text, target_code):
    try:
        return translator.translate(text, dest=target_code).text
    except:
        return text

# Construct prompt for Ollama
def build_prompt(context, user_input):
    return f"""You are a helpful assistant that answers questions using the given context.

Context:
{context}

User Question:
{user_input}

Answer:""" 

# Stream query to Ollama
def query_ollama(model, prompt):
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={'model': model, 'prompt': prompt, 'stream': True},
            stream=True
        )
        output = ""
        for line in response.iter_lines():
            if line:
                try:
                    data = line.decode('utf-8').split('data: ')[-1]
                    chunk = json.loads(data).get('response', '')
                    output += chunk
                except:
                    continue
        return output.strip()
    except requests.exceptions.ConnectionError:
        return "[Error] Cannot connect to Ollama. Is it running?"

# Start Ollama
def start_ollama(model):
    print(f"[Starting Ollama with model: {model}...]")
    return subprocess.Popen(['ollama', 'run', model], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

# Wait for Ollama server
def wait_for_ollama():
    print("[Waiting for Ollama API to become available...]")
    for _ in range(20):
        try:
            if requests.get('http://localhost:11434').status_code == 200:
                print("[Ollama is ready to use]")
                return True
        except:
            pass
        time.sleep(1)
    print("[Ollama did not start in time]")
    return False

# Main chatbot loop
def main():
    model = 'llama3.2'
    pdf_path = "./docs/Dino POS Usermanual.pdf"

    print("[Loading and embedding PDF]")
    chunks = load_pdf_chunks(pdf_path)
    index, all_chunks = embed_chunks(chunks)

    ollama_proc = start_ollama(model)
    if not wait_for_ollama():
        ollama_proc.terminate()
        return

    print("\nAsk your question (in any supported language). Type 'exit' to quit.\n")

    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Goodbye!")
                break

            lang_code, lang_name = detect_language(user_input)
            print(f"[Detected Language: {lang_name}]")

            query_en = translate_to_english(user_input)
            context = "\n\n".join(search_chunks(index, all_chunks, query_en, k=1))
            prompt = build_prompt(context, query_en)

            print(f"Assistant ({lang_name}): ", end='', flush=True)
            answer_en = query_ollama(model, prompt)
            final_answer = translate_from_english(answer_en, lang_code)

            print(final_answer + "\n")
    finally:
        if ollama_proc:
            ollama_proc.terminate()
            print("[Ollama subprocess terminated]")

if __name__ == "__main__":
    main()
