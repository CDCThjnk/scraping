# requirements:
#   pip install openai pydantic python-dotenv tqdm
#
# env:
#   export OPENAI_API_KEY=sk-...
#
# run:
#   python extract_astronauts_openai.py

import os
import json
from pathlib import Path
from typing import List, Optional
from dataclasses import asdict

from dotenv import load_dotenv
from tqdm import tqdm
from pydantic import BaseModel, Field, validator
from openai import OpenAI

# ---------- Config ----------
ROOT = Path("wikipedia_pages")  # directory containing subfolders, each with biography.txt
OUTPUT_JSONL = Path("astronauts_structured.jsonl")
MODEL = "gpt-4o-2024-08-06"  # any Responses API model that supports Structured Outputs

SYSTEM_PROMPT = """You are an information extraction system.
You will be given a single astronaut biography as raw text. 
Return ONLY the fields defined by the JSON schema. 
If a field is missing in the text, return null (or [] where applicable). 
Do not infer facts not present in the text. Use ISO dates (YYYY-MM-DD) if present; otherwise leave null.
"""

USER_INSTRUCTIONS = """Extract the following from the input:

- Degree(s): a list of degree strings like "Engineer-Physicist (Moscow Institute of Electronic Technology, 1989)"
- Education: array of objects: { institution, year (int or null), qualification }
- Occupation(s): list of roles/occupations (e.g., "Lieutenant Colonel, Russian Air Force", "Cosmonaut", etc.)
- time_in_space: free text as stated (e.g., "124 days 23 hours 52 minutes")
- interests: list of hobbies/interests exactly as mentioned (e.g., "tourism", "skiing", "water skiing", "balloon flights", "photo and videotaping")
- nationality: country name in English as stated (e.g., "Russian" -> return "Russia" if explicitly listed as Nationality: Russian; otherwise keep the literal text)
- age: integer age ONLY if explicitly present as a number in the text (e.g., "(age 59)"). If not present, return null.

Strictly follow the schema; do not add extra fields.
"""

# ---------- Schema ----------
class EducationItem(BaseModel):
    institution: Optional[str] = None
    year: Optional[int] = None
    qualification: Optional[str] = None

class AstroOutput(BaseModel):
    name: Optional[str] = Field(
        None, description="Astronaut's full name if present in the text header."
    )
    degrees: List[str] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)
    occupations: List[str] = Field(default_factory=list)
    time_in_space: Optional[str] = None
    interests: List[str] = Field(default_factory=list)
    nationality: Optional[str] = None
    age: Optional[int] = None

    @validator("degrees", "occupations", "interests", pre=True)
    def strip_list_items(cls, v):
        if not v:
            return []
        return [s.strip() for s in v if isinstance(s, str) and s.strip()]

# ---------- OpenAI client ----------
load_dotenv()
client = OpenAI(api_key="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")

def extract_with_openai(text: str) -> AstroOutput:
    """
    Use OpenAI Structured Outputs (Responses API) with a Pydantic model to guarantee valid JSON.
    """
    resp = client.responses.parse(
        model=MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_INSTRUCTIONS},
            {"role": "user", "content": text},
        ],
        text_format=AstroOutput,  # <- the schema: SDK converts to JSON Schema and parses the result
    )
    return resp.output_parsed  # pydantic model instance

def read_biography_txt(folder: Path) -> Optional[str]:
    bio = folder / "biography.txt"
    if bio.exists():
        return bio.read_text(encoding="utf-8", errors="ignore")
    return None

def guess_name_from_folder(folder: Path) -> str:
    # e.g., astronauts/Sergey_Revin -> "Sergey Revin"
    return folder.name.replace("_", " ").strip()

def main():
    OUTPUT_JSONL.unlink(missing_ok=True)
    folders = [p for p in ROOT.iterdir() if p.is_dir()]
    with OUTPUT_JSONL.open("a", encoding="utf-8") as out:
        for f in tqdm(sorted(folders), desc="Extracting"):
            txt = read_biography_txt(f)
            if not txt:
                continue
            try:
                data: AstroOutput = extract_with_openai(txt)
                # Ensure name if not in text:
                if not data.name:
                    data.name = guess_name_from_folder(f)
                out.write(json.dumps(data.dict(), ensure_ascii=False) + "\n")
            except Exception as e:
                # Write a minimal error record so you can triage later
                err_record = {
                    "name": guess_name_from_folder(f),
                    "error": str(e),
                }
                out.write(json.dumps(err_record, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {OUTPUT_JSONL.resolve()}")

if __name__ == "__main__":
    main()
