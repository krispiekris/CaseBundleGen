import subprocess
import json
from pathlib import Path

def run_ollama(model: str, prompt: str):
    process = subprocess.Popen(
        ["ollama", "run", model],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    out, err = process.communicate(prompt)
    if err:
        print("Ollama error:", err)
    return out