import json
import re
from pathlib import Path
from ollama_utils import run_ollama

def extract_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")

    json_str = match.group(0)

    if not json_str.strip().endswith("}"):
        json_str += "}"

    return json.loads(json_str)

BASE_DIR = Path("cases")

def load_prompt(name):
    return Path(f"prompts/{name}").read_text()

def generate_metadata(case_id, model="ma3"):
    prompt = load_prompt("metadata_prompt.txt").replace("CASE_ID", case_id)

    for attempt in range(2):  # two attempts max
        output = run_ollama(model, prompt)

        try:
            metadata = extract_json(output)
            return metadata

        except Exception as e:
            print(f"❌ Attempt {attempt+1} failed to parse JSON.")
            print("Raw output was:")
            print(output)
            print("Retrying...\n")

    raise RuntimeError("Failed to generate valid JSON after 2 attempts.")

def save_file(case_dir, filename, content):
    (case_dir / filename).write_text(content)

def generate_case(case_id="case_001", model="llama3"):
    case_dir = Path("cases") / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    sample_files = [ "report_1.xml", "report_2.xml", "report_3.xml",
                    "bank_statement_1.csv", "bank_statement_2.csv", "brochure_1.txt",
                    "brochure_2.txt", "email_2.txt" ]
    
    samples = ""
    for f in sample_files:
        samples += f"\n\n### SAMPLE FILE: {f}\n"
        samples += Path(f"samples/{f}").read_text()

    prompt = samples + "\n\n" + Path("prompts/generate_full_case.txt").read_text()

    # Run Ollama once
    output = run_ollama(model, prompt)

    # Extract JSON
    data = extract_json(output)

    # Save metadata
    (case_dir / "metadata.json").write_text(
        json.dumps(data["metadata"], indent=2)
    )

    # Save all generated files
    for f in data["files"]:
        (case_dir / f["filename"]).write_text(f["content"])

    print(f"Case {case_id} generated successfully.")

if __name__ == "__main__":
    generate_case("case_001")