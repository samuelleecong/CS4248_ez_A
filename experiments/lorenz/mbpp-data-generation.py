import pandas as pd
import requests
import json
import time
import re

# -------- CONFIG --------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"   # make sure this is installed: ollama pull mistral
OUTPUT_FILE = "mbpp_queries_synthetic.json"

mbpp = pd.read_json("sanitized-mbpp.json")


# -------- JSON EXTRACTOR --------
def extract_json(text):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except:
        pass
    return None


# -------- GENERATION FUNCTION --------
def generate_queries(prompt):
    prompt_text = f"""
You MUST return ONLY valid JSON.

Format:
{{
  "queries": [
    "query1",
    "query2",
    "query3",
    "query4",
    "query5"
  ]
}}

Rules:
- No explanation
- No numbering
- No extra text
- Each query must be short (3-6 words)

Task:
{prompt}
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt_text,
                "stream": False
            },
            timeout=60
        )

        data = response.json()

        # 🔥 Handle missing response key
        if "response" not in data:
            print("\n❌ Unexpected Ollama response:")
            print(data)
            return None

        text = data["response"]

        parsed = extract_json(text)

        if parsed is None:
            print("\n❌ JSON PARSE FAILED:")
            print(text[:300])
            return None

        return parsed.get("queries", [])

    except Exception as e:
        print("\n🔥 REQUEST FAILED:")
        print(e)
        return None


# -------- MAIN LOOP --------

# Ensure file exists
open(OUTPUT_FILE, "w").close()

with open(OUTPUT_FILE, "a") as f:

    for i, prompt in enumerate(mbpp["prompt"]):
        print(f"Processing {i}", flush=True)

        queries = generate_queries(prompt)

        if not queries:
            continue

        for q in queries:
            record = {
                "source_id": i,
                "original_prompt": prompt,
                "query": q
            }

            f.write(json.dumps(record) + "\n")

        f.flush()

        # small delay to avoid overload
        time.sleep(0.3)