import base64
import json
import mimetypes
import re
from pathlib import Path
from ollama_utils import run_ollama


def _is_image_file(path: Path):
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}


def _format_sample_file_for_prompt(path: Path):
    if _is_image_file(path):
        content_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return (
            f"[IMAGE BASE64 {path.name}]\n"
            f"Content-Type: {mime_type}\n"
            f"{content_b64}"
        )
    return path.read_text()


def _is_escaped(json_str, pos):
    backslashes = 0
    i = pos - 1
    while i >= 0 and json_str[i] == "\\":
        backslashes += 1
        i -= 1
    return backslashes % 2 == 1


def _escape_control_character(char):
    return {
        "\n": "\\n",
        "\r": "\\r",
        "\t": "\\t",
    }.get(char, char)


def _repair_json_string(json_str):
    for _ in range(20):
        try:
            return json.loads(json_str, strict=False)
        except json.JSONDecodeError as exc:
            pos = exc.pos
            if pos >= len(json_str):
                break

            current = json_str[pos]
            if current in "\n\r\t":
                json_str = json_str[:pos] + _escape_control_character(current) + json_str[pos+1:]
                continue

            if current == "\\":
                if pos + 1 < len(json_str) and json_str[pos + 1] not in '"\\/bfnrtu':
                    json_str = json_str[:pos] + "\\\\" + json_str[pos+1:]
                    continue

            if current == '"' and not _is_escaped(json_str, pos):
                json_str = json_str[:pos] + "\\" + json_str[pos:]
                continue

            break

    return json.loads(json_str, strict=False)


def _extract_first_json_object(text):
    start = text.find('{')
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == '\\':
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None


def extract_json(text):
    """Extract JSON object from model output."""
    json_str = _extract_first_json_object(text)
    if json_str is None:
        raise ValueError("No JSON object found in model output")

    # Escape raw control characters that may be present inside generated strings.
    json_str = json_str.replace("\t", "\\t").replace("\r", "\\r")

    try:
        return json.loads(json_str, strict=False)
    except json.JSONDecodeError as exc:
        try:
            repaired = _repair_json_string(json_str)
            return repaired
        except json.JSONDecodeError as repaired_exc:
            snippet_start = max(0, exc.pos - 80)
            snippet_end = min(len(json_str), exc.pos + 80)
            snippet = json_str[snippet_start:snippet_end]
            raise ValueError(
                f"Failed to parse JSON output at position {exc.pos}: {exc.msg}\n"
                f"Output snippet: {snippet!r}\n"
                f"Full model output:\n{text}"
            ) from repaired_exc

def save_file(case_dir, filename, content):
    """Save content to a file in the case directory."""
    (case_dir / filename).write_text(content)


def _run_with_retry(prompt: str, model: str, max_attempts: int = 3):
    """Run Ollama with retries, raising on all attempts exhausted."""
    last_error = None
    for attempt in range(1, max_attempts + 1):
        output = run_ollama(model, prompt)
        try:
            return extract_json(output)
        except (ValueError, KeyError) as exc:
            last_error = exc
            print(f"    [retry {attempt}/{max_attempts}] JSON extraction failed: {exc}")
    raise RuntimeError(
        f"Failed to get valid JSON after {max_attempts} attempts."
    ) from last_error


def step1_generate_metadata_and_filelist(model="gemma3:12b"):
    """Step 1: Generate metadata and file structure."""
    print("  [Step 1] Generating metadata and file structure...")
    
    # Use all sample files in the samples directory as style reference
    sample_dir = Path("samples")
    sample_files = sorted(
        [p for p in sample_dir.iterdir() if p.is_file()],
        key=lambda p: p.name
    )
    
    samples = ""
    for sample_path in sample_files:
        samples += f"\n\n### SAMPLE FILE: {sample_path.name}\n"
        samples += _format_sample_file_for_prompt(sample_path)

    prompt = (
        "### STYLE SAMPLES (DO NOT COPY CONTENT)\n"
        + samples
        + "\n\n### IMAGE SAMPLE INSTRUCTIONS\n"
        + "Image samples are provided as Base64-encoded image data. "
        + "Use them only for visual style and layout guidance; do not reproduce exact screenshots or logos.\n\n"
        + "### GENERATION INSTRUCTIONS\n"
        + Path("prompts/step1_metadata_filelist.txt").read_text()
    )

    data = _run_with_retry(prompt, model)
    return data["metadata"], data["file_structure"]

def step2_generate_file_content(metadata, file_structure, model="gemma3:12b"):
    """Step 2: Generate file content based on metadata."""
    print("  [Step 2] Generating file content...")
    
    prompt_template = Path("prompts/step2_file_content.txt").read_text()
    
    # Format the template with metadata and file structure (no style samples)
    prompt = (
        "### GENERATION INSTRUCTIONS\n"
        + prompt_template.format(
            metadata=json.dumps(metadata, indent=2),
            file_structure=json.dumps(file_structure, indent=2)
        )
    )

    data = _run_with_retry(prompt, model)
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

    data = _run_with_retry(prompt, model)
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

def generate_batch(n=100, model="gemma3:12b"):
    """Generate multiple cases in a batch."""
    for i in range(1, n + 1):
        case_id = f"case_{i:03d}"
        generate_case(case_id, model=model)

if __name__ == "__main__":
    generate_batch(100)