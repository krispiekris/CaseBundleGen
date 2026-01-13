from pathlib import Path
import subprocess

def run_ollama(prompt):
    process = subprocess.Popen(
        ["ollama", "run", "llama3"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    out, err = process.communicate(prompt)
    return out, err

# Load prompt
prompt = Path("prompts/metadata_prompt.txt").read_text()

# Run Ollama
output, error = run_ollama(prompt)

print("=== RAW OUTPUT ===")
print(output)
print("=== ERROR ===")
print(error)

# Save raw output
case_dir = Path("cases/case_001")
case_dir.mkdir(parents=True, exist_ok=True)

(case_dir / "metadata_raw.txt").write_text(output)