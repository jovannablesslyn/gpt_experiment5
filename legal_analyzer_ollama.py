"""
24CS2506 - Building Applications using GPT-4 (local Ollama variant)
Experiment 5 - Automated Legal Document Analyzer using Ollama (local LLM)

Setup:
    1. Install Ollama from https://ollama.com and make sure it is running
    2. ollama pull llama3.2            (or any model: mistral, qwen2.5, gemma2 ...)
    3. pip install ollama pypdf
Run:
    python legal_analyzer_ollama.py
Optional: choose another model without editing code
    set OLLAMA_MODEL=mistral           (Mac/Linux: export OLLAMA_MODEL=mistral)
"""

import os
import re
import unicodedata
from datetime import datetime

import ollama

MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
NUM_CTX = 8192                  # context window requested from the local model
MAX_CHUNK_CHARS = 6000          # ~1500 tokens; fits an 8K context window
OUTPUT_DIR = "outputs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "legal_analysis.txt")


def check_ollama():
    """Verify the Ollama server is running and the chosen model is downloaded."""
    try:
        listing = ollama.list()
    except Exception:
        raise RuntimeError("Cannot reach Ollama. Install it from https://ollama.com "
                           "and make sure it is running (try:  ollama serve).")
    models = listing["models"] if isinstance(listing, dict) else listing.models
    names = []
    for m in models:
        n = m["model"] if isinstance(m, dict) else m.model
        names.append(n)
    if not any(n == MODEL or n.startswith(MODEL + ":") for n in names):
        raise RuntimeError(f"Model '{MODEL}' not found. Download it with:  ollama pull {MODEL}\n"
                           f"Installed models: {', '.join(names) or 'none'}")


# --------------------------------------------------------------------------
# PART A - Legal Document Input
# --------------------------------------------------------------------------
def read_from_keyboard():
    print("Paste the legal text. Type END on a new line when finished:")
    lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def read_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(path):
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("PDF support needs pypdf:  pip install pypdf")
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def load_document():
    """Q1: Read a legal document via keyboard, .txt or .pdf."""
    print("\n  1. Type / paste text")
    print("  2. Load .txt file")
    print("  3. Load .pdf file")
    choice = input("Select input method (1-3): ").strip()

    if choice == "1":
        return read_from_keyboard(), "keyboard input"
    if choice in ("2", "3"):
        path = input("Enter file path: ").strip().strip('"')
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")
        ext = os.path.splitext(path)[1].lower()
        if choice == "2" and ext != ".txt":
            raise ValueError("Please select a .txt file.")
        if choice == "3" and ext != ".pdf":
            raise ValueError("Please select a .pdf file.")
        return (read_txt(path) if choice == "2" else read_pdf(path)), path
    raise ValueError("Invalid choice.")


def display_document_info(text, source):
    """Q2: Display the document information neatly."""
    words = len(text.split())
    lines = len([l for l in text.splitlines() if l.strip()])
    first_line = next((l.strip() for l in text.splitlines() if l.strip()), "N/A")
    print("\n" + "=" * 60)
    print("DOCUMENT INFORMATION")
    print("=" * 60)
    print(f"Source       : {source}")
    print(f"Title (guess): {first_line[:80]}")
    print(f"Characters   : {len(text)}")
    print(f"Words        : {words}")
    print(f"Lines        : {lines}")
    print("-" * 60)
    print("Preview:")
    print(text[:400].strip() + ("..." if len(text) > 400 else ""))
    print("=" * 60)


# --------------------------------------------------------------------------
# PART B - Preprocessing
# --------------------------------------------------------------------------
SECTION_PATTERN = re.compile(
    r"^\s*(?:(?:article|section|clause)\s+\w+[.:)]?|\d+(?:\.\d+)*[.)]\s+|[A-Z][A-Z \-&]{4,}$)",
    re.IGNORECASE | re.MULTILINE,
)


def clean_text(text):
    """Q3: remove unwanted characters, extra spaces, normalise text."""
    text = unicodedata.normalize("NFKC", text)             # normalise unicode
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"[^\x09\x0A\x20-\x7E]", " ", text)      # strip control / odd chars
    text = re.sub(r"[ \t]+", " ", text)                    # collapse spaces/tabs
    text = re.sub(r" *\n *", "\n", text)                   # trim around newlines
    text = re.sub(r"\n{3,}", "\n\n", text)                 # collapse blank lines
    return text.strip()


def identify_sections(text):
    """Split text into sections using headings like 'Section 3', '4.', 'TERMINATION'."""
    starts = [m.start() for m in SECTION_PATTERN.finditer(text)]
    if not starts:
        return [text]
    if starts[0] != 0:
        starts.insert(0, 0)
    starts.append(len(text))
    sections = [text[starts[i]:starts[i + 1]].strip() for i in range(len(starts) - 1)]
    return [s for s in sections if s]


def chunk_document(text, max_chars=MAX_CHUNK_CHARS):
    """Group sections into chunks <= max_chars. Over-long sections are split by sentence."""
    chunks, current = [], ""
    for section in identify_sections(text):
        pieces = [section]
        if len(section) > max_chars:
            pieces = re.split(r"(?<=[.;])\s+", section)
        for piece in pieces:
            if len(current) + len(piece) + 2 > max_chars and current:
                chunks.append(current.strip())
                current = ""
            current += piece + "\n\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks


def preprocess(text):
    cleaned = clean_text(text)
    return cleaned, chunk_document(cleaned)


# --------------------------------------------------------------------------
# PART C - Prompt Generation
# --------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a careful legal document analysis assistant. "
    "You extract and summarise information strictly from the text provided. "
    "You do not give legal advice."
)

ANALYSIS_FIELDS = """\
Document Type:
Parties:
Purpose:
Summary:
Obligations:
Payment Terms:
Duration:
Termination Conditions:
Important Dates:
Key Clauses:"""


def build_analysis_prompt(document_text, part_info=""):
    """Q4: Prompt for analysing a (chunk of a) legal document."""
    return f"""Analyze the legal document {part_info}below and produce a structured report.

Use EXACTLY these headings, in this order:

{ANALYSIS_FIELDS}

Instructions:
- Summary: 3-5 concise sentences.
- Obligations: list the obligations of each party separately, as bullet points.
- Key Clauses: list important clauses (e.g. confidentiality, liability, indemnity,
  governing law, dispute resolution) with a one-line description each.
- Do not invent information. If information is not available, state
  "Not mentioned in the document."
- Do not add any extra commentary outside the headings.

LEGAL DOCUMENT:
\"\"\"
{document_text}
\"\"\"
"""


def build_merge_prompt(partial_reports):
    """Prompt to combine per-chunk analyses of a long document into one report."""
    joined = "\n\n--- PART ---\n\n".join(partial_reports)
    return f"""Below are partial analyses of consecutive parts of ONE legal document.
Merge them into a single consolidated report using EXACTLY these headings:

{ANALYSIS_FIELDS}

Instructions:
- Remove duplicates and resolve overlaps; keep the Summary to 3-5 sentences.
- If a field is "Not mentioned in the document." in some parts but present in
  others, keep the information that is present.
- Do not invent information. If information is not available, state
  "Not mentioned in the document."

PARTIAL ANALYSES:
\"\"\"
{joined}
\"\"\"
"""


# --------------------------------------------------------------------------
# PART D - LLM Analysis / Summarisation
# --------------------------------------------------------------------------
def ask_llm(prompt):
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={
            "temperature": 0,        # deterministic, reduces hallucination
            "num_ctx": NUM_CTX,
            "num_predict": 1200,
        },
    )
    return response["message"]["content"].strip()


def analyze_document(chunks):
    """Q5: analyse every chunk, then merge if there is more than one."""
    if len(chunks) == 1:
        return ask_llm(build_analysis_prompt(chunks[0]))

    partials = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  Analysing part {i}/{len(chunks)} ...")
        partials.append(ask_llm(build_analysis_prompt(chunk, f"(part {i} of {len(chunks)}) ")))
    print("  Merging partial analyses ...")
    return ask_llm(build_merge_prompt(partials))


def generate_summary(analysis):
    """Extract the headline fields into the 'LEGAL DOCUMENT SUMMARY' layout."""
    wanted = ["Document Type", "Parties", "Purpose", "Duration", "Summary"]
    headings = [h.rstrip(":") for h in ANALYSIS_FIELDS.splitlines()]
    pattern = r"^\s*\**(%s)\**\s*:\**\s*" % "|".join(re.escape(h) for h in headings)
    parts, current = {}, None
    for line in analysis.splitlines():
        m = re.match(pattern, line, re.IGNORECASE)
        if m:
            current = next(h for h in headings if h.lower() == m.group(1).lower())
            parts[current] = [line[m.end():].strip()]
        elif current:
            parts[current].append(line.rstrip())

    out = ["LEGAL DOCUMENT SUMMARY", "-" * 32]
    for field in wanted:
        body = "\n".join(parts.get(field, ["Not mentioned in the document."])).strip()
        out.append(f"{field}:\n{body or 'Not mentioned in the document.'}\n")
    return "\n".join(out).rstrip()


# --------------------------------------------------------------------------
# PART E - Save Analysis
# --------------------------------------------------------------------------
def save_report(source, summary, analysis):
    """Q6: write report to outputs/legal_analysis.txt."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("AUTOMATED LEGAL DOCUMENT ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated : {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Source    : {source}\n")
        f.write(f"Model     : {MODEL} (Ollama, local)\n")
        f.write("=" * 60 + "\n\n")
        f.write(summary + "\n\n")
        f.write("=" * 60 + "\nDETAILED ANALYSIS\n" + "=" * 60 + "\n\n")
        f.write(analysis + "\n\n")
        f.write("-" * 60 + "\nNote: AI-generated; not legal advice. "
                "Verify against the original document.\n")
    return OUTPUT_FILE


# --------------------------------------------------------------------------
# PART F - Complete Application
# --------------------------------------------------------------------------
def menu():
    print("\n" + "=" * 60)
    print(" AUTOMATED LEGAL DOCUMENT ANALYZER (Ollama)")
    print("=" * 60)
    print(" 1. Upload / Enter Legal Document")
    print(" 2. Analyze Document")
    print(" 3. Generate Summary")
    print(" 4. Save Report")
    print(" 5. Exit")


def main():
    raw_text = source = None
    chunks, analysis, summary = [], None, None

    while True:
        menu()
        choice = input("Enter choice (1-5): ").strip()

        try:
            if choice == "1":
                raw_text, source = load_document()
                if not raw_text.strip():
                    raw_text = None
                    print("The document is empty.")
                    continue
                display_document_info(raw_text, source)
                analysis = summary = None      # reset previous results

            elif choice == "2":
                if not raw_text:
                    print("Please load a document first (option 1).")
                    continue
                cleaned, chunks = preprocess(raw_text)
                print(f"Preprocessed: {len(cleaned)} chars, {len(chunks)} chunk(s).")
                check_ollama()
                print(f"Sending to Ollama ({MODEL}) ...")
                analysis = analyze_document(chunks)
                summary = None
                print("\n" + analysis)

            elif choice == "3":
                if not analysis:
                    print("Please analyse the document first (option 2).")
                    continue
                summary = generate_summary(analysis)
                print("\n" + summary)

            elif choice == "4":
                if not analysis:
                    print("Nothing to save. Analyse the document first.")
                    continue
                summary = summary or generate_summary(analysis)
                print(f"Report saved to: {save_report(source, summary, analysis)}")

            elif choice == "5":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Enter a number from 1 to 5.")

        except Exception as e:            # keep the menu alive on errors
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
