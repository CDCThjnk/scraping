import re
import json
from datetime import datetime, date
from typing import List, Dict, Optional

MONTHS = "|".join([
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
])

def _extract_first(pattern: str, text: str, flags=re.I|re.M) -> Optional[str]:
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None

def _extract_all(pattern: str, text: str, flags=re.I|re.M) -> List[str]:
    return [g.strip() for g in re.findall(pattern, text, flags)]

def _clean_commas_list(s: str) -> List[str]:
    # split by commas and "and" while keeping phrases like "water skiing"
    parts = re.split(r",|\band\b", s)
    return [p.strip(" .;:") for p in parts if p and p.strip(" .;:")]

def _parse_birthdate(text: str) -> Optional[date]:
    """
    Accepts any of:
      - 'Born: ( 1966-01-12 ) January 12, 1966 ...'
      - '(1966-01-12)'
      - 'January 12, 1966'
      - '12 January 1966'
    Returns a date or None.
    """
    # ISO in parens
    iso = _extract_first(r"Born:\s*\(\s*(\d{4}-\d{2}-\d{2})\s*\)", text)
    if iso:
        try:
            return datetime.strptime(iso, "%Y-%m-%d").date()
        except ValueError:
            pass
    # Month D, YYYY
    mdyyyy = _extract_first(rf"({MONTHS}\s+\d{{1,2}},\s+\d{{4}})", text)
    if mdyyyy:
        try:
            return datetime.strptime(mdyyyy, "%B %d, %Y").date()
        except ValueError:
            pass
    # D Month YYYY
    dmyyyy = _extract_first(rf"(\d{{1,2}}\s+(?:{MONTHS})\s+\d{{4}})", text)
    if dmyyyy:
        try:
            return datetime.strptime(dmyyyy, "%d %B %Y").date()
        except ValueError:
            pass
    return None

def _parse_age(text: str, birth: Optional[date]) -> Optional[int]:
    # Prefer explicit (age 59)
    age_explicit = _extract_first(r"\(age\s*([0-9]{1,3})\)", text)
    if age_explicit:
        try:
            return int(age_explicit)
        except ValueError:
            pass
    # Fallback: compute from birthdate if present
    if birth:
        today = date.today()
        years = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
        return years
    return None

def _parse_degrees_and_education(text: str):
    """
    Build:
      degrees: List[str]
      education: List[{"institution":..., "year":..., "qualification":...}]
    """
    degrees: List[str] = []
    education: List[Dict[str, Optional[str]]] = []

    # Scan education/narrative bullets for lines with 'graduated', 'post-graduate', 'qualified as'
    edu_lines = _extract_all(
        r"^\s*-\s*(?:Revin|He|She)??.*?(graduated|post-graduate|qualified as).*?$",
        text
    )

    # Also pull lines under an 'Education' section bullets
    edu_section_lines = _extract_all(
        r"^\s*-\s*Revin.*?qualified.*?$|^\s*-\s*.*?post-graduate.*?$",
        text
    )
    candidate_lines = list(set(edu_lines + edu_section_lines))

    # Parse each candidate line
    for line in candidate_lines:
        inst = _extract_first(r"from the ([A-Z][A-Za-z0-9 .\-()']+)", line)
        if not inst:
            inst = _extract_first(r"at the ([A-Z][A-Za-z0-9 .\-()']+)", line)
        year = _extract_first(r"(?:in|,)\s*(\d{4})(?!\d)", line)
        qual = _extract_first(r"qualified as (?:an? )?([A-Za-z \-]+?)(?:\.|,|$)", line)

        entry = {
            "institution": inst,
            "year": int(year) if year else None,
            "qualification": qual
        }
        if any(entry.values()):
            education.append(entry)

        if qual:
            if inst and year:
                degrees.append(f"{qual} ({inst}, {year})")
            elif inst:
                degrees.append(f"{qual} ({inst})")
            elif year:
                degrees.append(f"{qual} ({year})")
            else:
                degrees.append(qual)

    # Also catch explicit degree phrases like "Candidate of Pedagogic Sciences (2013)"
    explicit_degrees = _extract_all(r"(Candidate of [A-Za-z ]+ \(\d{4}\))", text)
    degrees.extend(explicit_degrees)

    # De-dup & clean
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    degrees = [_norm(d) for d in degrees]
    # remove duplicates preserving order
    seen = set()
    dedup_degrees = []
    for d in degrees:
        if d not in seen:
            seen.add(d)
            dedup_degrees.append(d)

    # Clean education entries (drop empties)
    cleaned_edu = []
    for e in education:
        if any(v is not None for v in e.values()):
            cleaned_edu.append(e)

    return dedup_degrees, cleaned_edu

def parse_astronaut_profile(text: str) -> Dict:
    # Occupations
    occ_line = _extract_first(r"^-+\s*Occupation\(s\):\s*([^\n]+)$", text)
    occupations = []
    if occ_line:
        occupations = [o.strip() for o in occ_line.split(",") if o.strip()]

    # Nationality
    nationality = _extract_first(r"^-+\s*Nationality:\s*([A-Za-z \-]+)$", text)

    # Time in space
    time_in_space = _extract_first(r"^-+\s*Time in space:\s*([^\n]+)$", text)

    # Interests (Personal: 'He enjoys ...')
    enjoys = _extract_first(r"\benjoys\s+([^.\n]+)", text)
    interests = _clean_commas_list(enjoys) if enjoys else []

    # Education & Degrees
    degrees, education = _parse_degrees_and_education(text)

    # Age
    birth = _parse_birthdate(text)
    age = _parse_age(text, birth)

    return {
        "degrees": degrees,                    # List[str]
        "education": education,                # List[{institution, year, qualification}]
        "occupations": occupations,            # List[str]
        "time_in_space": time_in_space,        # str or None
        "interests": interests,                # List[str]
        "nationality": nationality,            # str or None
        "age": age                              # int or None
    }

if __name__ == "__main__":
    # Example usage: read from a file and print JSON
    import sys
    if sys.stdin.isatty() and len(sys.argv) < 2:
        print("Usage: python extract_astronaut.py <input.txt>\n  or: cat input.txt | python extract_astronaut.py")
        sys.exit(1)

    text = sys.stdin.read() if not sys.stdin.isatty() else open(sys.argv[1], "r", encoding="utf-8").read()
    result = parse_astronaut_profile(text)
    print(json.dumps(result, ensure_ascii=False, indent=2))
