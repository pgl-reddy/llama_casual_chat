import subprocess
import requests
import time
from langdetect import detect
import signal

LANG_CODE_MAP = {
    'hi': 'Hindi',
    'te': 'Telugu',
    'ta': 'Tamil',
    'gu': 'Gujarati',
    'kn': 'Kannada',
    'mr': 'Marathi',
    'pa': 'Punjabi',
    'en': 'English',
}

def detect_language(text):
    try:
        code = detect(text)
        return LANG_CODE_MAP.get(code, 'the user\'s language')
    except:
        return "the user's language"

def build_prompt(language_name, user_text):
    return f"""You are a helpful assistant that always replies in the same language as the user.
Respond only in {language_name}. Be clear and concise.

User: {user_text}
Assistant:"""

def query_ollama(model, prompt):
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model,
                'prompt': prompt,
                'stream': False
            }
        )
        if response.status_code == 200:
            return response.json()['response'].strip()
        else:
            return f"[Ollama Error] {response.status_code}: {response.text}"
    except requests.exceptions.ConnectionError:
        return "[Error] Cannot connect to Ollama. Is it running?"

def start_ollama(model):
    print(f"[Starting Ollama with model: {model}...]")
    # Start Ollama in the background
    return subprocess.Popen(
        ['ollama', 'run', model],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
    )

def wait_for_ollama():
    print("[Waiting for Ollama API to become available...]")
    for _ in range(20):
        try:
            r = requests.get('http://localhost:11434')
            if r.status_code == 200:
                print("[Ollama is ready to use]")
                return True
        except:
            pass
        time.sleep(1)
    print("[Ollama did not start in time]")
    return False

def main():
    model = 'llama3.2'  
    ollama_proc = start_ollama(model)
    if not wait_for_ollama():
        ollama_proc.terminate()
        return
    print("\nYou can now chat with the assistant. Type 'exit' to quit.\n")
    try:
        while True:
            user_input = input("User: ").strip()
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Goodbye!")
                break

            lang_name = detect_language(user_input)
            print(f"[Detected Language: {lang_name}]")
            prompt = build_prompt(lang_name, user_input)
            reply = query_ollama(model, prompt)

            print(f"Assistant ({lang_name}): {reply}\n")
    finally:
        if ollama_proc:
            ollama_proc.terminate()
            print("[Shutting down Ollama subprocess]")

if __name__ == "__main__":
    main()
