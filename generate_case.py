import json
import re
from pathlib import Path
from ollama_utils import run_ollama

def extract_json(text):
    """Extract JSON object from model output."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")

    json_str = match.group(0)
    json_str = json_str.strip()

    if not json_str.endswith("}"):
        json_str += "}"

    # Escape literal control characters that can appear inside generated strings.
    json_str = json_str.replace("\t", "\\t").replace("\r", "\\r")

    try:
        return json.loads(json_str, strict=False)
    except json.JSONDecodeError as exc:
        # Fix stray backslashes that are not part of valid JSON escapes.
        repaired = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str)
        try:
            return json.loads(repaired, strict=False)
        except json.JSONDecodeError:
            snippet_start = max(0, exc.pos - 80)
            snippet_end = min(len(json_str), exc.pos + 80)
            snippet = json_str[snippet_start:snippet_end]
            raise ValueError(
                f"Failed to parse JSON output at position {exc.pos}: {exc.msg}\n"
                f"Output snippet: {snippet!r}\n"
                f"Full model output:\n{text}"
            ) from exc

def save_file(case_dir, filename, content):
    """Save content to a file in the case directory."""
    (case_dir / filename).write_text(content)

def step1_generate_metadata_and_filelist(model="gemma3:12b"):
    """Step 1: Generate metadata and file structure."""
    print("  [Step 1] Generating metadata and file structure...")
    
    # Use sample files as style reference
    sample_files = [
        "report_1.xml",
        "bank_statement_1.csv",
        "brochure_1.txt",
        "email_2.txt"
    ]
    
    samples = ""
    for f in sample_files:
        samples += f"\n\n### SAMPLE FILE: {f}\n"
        samples += Path(f"samples/{f}").read_text()

    prompt = (
        "### STYLE SAMPLES (DO NOT COPY CONTENT)\n"
        + samples
        + "\n\n### GENERATION INSTRUCTIONS\n"
        + Path("prompts/step1_metadata_filelist.txt").read_text()
    )

    output = run_ollama(model, prompt)
    data = extract_json(output)
    
    return data["metadata"], data["file_structure"]

def step2_generate_file_content(metadata, file_structure, model="gemma3:12b"):
    """Step 2: Generate file content based on metadata."""
    print("  [Step 2] Generating file content...")
    
    # Use sample files as style reference
    sample_files = [
        "report_1.xml",
        "bank_statement_1.csv",
        "brochure_1.txt",
        "email_2.txt"
    ]
    
    samples = ""
    for f in sample_files:
        samples += f"\n\n### SAMPLE FILE: {f}\n"
        samples += Path(f"samples/{f}").read_text()

    prompt_template = Path("prompts/step2_file_content.txt").read_text()
    
    # Format the template with metadata and file structure
    prompt = (
        "### STYLE SAMPLES (DO NOT COPY CONTENT)\n"
        + samples
        + "\n\n### GENERATION INSTRUCTIONS\n"
        + prompt_template.format(
            metadata=json.dumps(metadata, indent=2),
            file_structure=json.dumps(file_structure, indent=2)
        )
    )

    output = run_ollama(model, prompt)
    data = extract_json(output)
    
    return data["files"]

def step3_generate_qa(files, model="gemma3:12b"):
    """Step 3: Generate Q&A pairs based on file content."""
    print("  [Step 3] Generating Q&A pairs...")
    
    # Prepare files content for the prompt
    files_content = ""
    for f in files:
        files_content += f"\n\n### FILE: {f['filename']}\n"
        files_content += f["content"]

    prompt_template = Path("prompts/step3_qa_generation.txt").read_text()
    
    # Format the template with files content only
    prompt = (
        prompt_template.format(
            files_content=files_content
        )
    )

    output = run_ollama(model, prompt)
    data = extract_json(output)
    
    return data["qa"]

def generate_case(case_id="case_001", model="gemma3:12b"):
    """Generate a complete case using a three-step pipeline."""
    case_dir = Path("cases") / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Generating {case_id} ===")
    
    # Step 1: Generate metadata and file structure
    metadata, file_structure = step1_generate_metadata_and_filelist(model)
    
    # Step 2: Generate file content
    files = step2_generate_file_content(metadata, file_structure, model)
    
    # Step 3: Generate Q&A
    qa = step3_generate_qa(files, model)
    
    # Save all results
    (case_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    
    (case_dir / "qa.json").write_text(
        json.dumps(qa, indent=2)
    )
    
    for f in files:
        (case_dir / f["filename"]).write_text(f["content"])

    print(f"  ✓ Case {case_id} generated successfully.")

def generate_batch(n=10, model="gemma3:12b"):
    """Generate multiple cases in a batch."""
    for i in range(1, n + 1):
        case_id = f"case_{i:03d}"
        generate_case(case_id, model=model)

if __name__ == "__main__":
    generate_batch(10)