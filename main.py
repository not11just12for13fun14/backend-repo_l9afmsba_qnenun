import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr

app = FastAPI(title="FlamesAI Document Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Models ----------
class AnalyzeRequest(BaseModel):
    content: str = Field(..., description="Raw document text or form content")


class AnalyzeResponse(BaseModel):
    filled_fields: Dict[str, Any]
    corrected_fields: Dict[str, Any]
    missing_fields: List[str]
    suggested_improvements: List[str]
    final_clean_version: str
    structured_json: Dict[str, Any]


# ---------- Utilities ----------
KEY_ALIASES: Dict[str, str] = {
    "full name": "name",
    "applicant name": "name",
    "first name": "first_name",
    "last name": "last_name",
    "e-mail": "email",
    "mail": "email",
    "email address": "email",
    "mobile": "phone",
    "telephone": "phone",
    "phone number": "phone",
    "dob": "date_of_birth",
    "birth date": "date_of_birth",
    "date of birth": "date_of_birth",
    "country of residence": "country",
    "nationality": "country",
    "doc id": "document_id",
    "document id": "document_id",
    "id number": "document_id",
    "signature date": "signature_date",
    "signed on": "signature_date",
}

STANDARD_REQUIRED_FIELDS = [
    "name",
    "email",
    "phone",
    "address",
    "date_of_birth",
    "country",
    "document_id",
    "signature_date",
]

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
PHONE_RE = re.compile(r"[+]?\d[\d\s().-]{7,}")
DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4})\b")


def normalize_key(k: str) -> str:
    base = k.strip().lower().replace("_", " ").replace("-", " ")
    base = re.sub(r"\s+", " ", base)
    if base in KEY_ALIASES:
        return KEY_ALIASES[base]
    return base.replace(" ", "_")


def extract_pairs(text: str) -> Dict[str, str]:
    pairs: Dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            if key.strip():
                pairs[normalize_key(key)] = val.strip()
        else:
            # Heuristics: detect standalone emails/phones/dates
            if EMAIL_RE.search(line):
                pairs.setdefault("email", EMAIL_RE.search(line).group(0))
            if PHONE_RE.search(line):
                pairs.setdefault("phone", PHONE_RE.search(line).group(0))
            if DATE_RE.search(line):
                pairs.setdefault("signature_date", DATE_RE.search(line).group(0))
    return pairs


def clean_value(key: str, value: str) -> Optional[str]:
    v = value.strip()
    if not v:
        return None
    if key == "email":
        return v if EMAIL_RE.match(v) else None
    if key == "phone":
        digits = re.sub(r"[^0-9+]", "", v)
        return digits if len(re.sub(r"\D", "", digits)) >= 8 else None
    if key in {"date_of_birth", "signature_date"}:
        # Normalize date formats
        m = DATE_RE.search(v)
        if not m:
            return None
        raw = m.group(0)
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
            try:
                dt = datetime.strptime(raw, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        return None
    return v


def analyze_text(text: str) -> AnalyzeResponse:
    raw_pairs = extract_pairs(text)
    filled: Dict[str, Any] = {}
    corrected: Dict[str, Any] = {}
    improvements: List[str] = []

    # Clean and validate
    for k, v in raw_pairs.items():
        cleaned = clean_value(k, v)
        if cleaned is None:
            improvements.append(f"Invalid or ambiguous value for '{k}'.")
        else:
            filled[k] = cleaned
            if cleaned != v:
                corrected[k] = cleaned

    # Missing fields from standard set
    missing = [f for f in STANDARD_REQUIRED_FIELDS if f not in filled]

    # Compose a clean, professional version
    lines = []
    order = [
        "name",
        "first_name",
        "last_name",
        "email",
        "phone",
        "address",
        "country",
        "date_of_birth",
        "document_id",
        "signature_date",
    ]
    for key in order:
        if key in filled:
            label = key.replace("_", " ").title()
            lines.append(f"{label}: {filled[key]}")
    final_text = (
        "Validated document. Fields are organized, normalized, and formatted.\n" + "\n".join(lines)
        if lines
        else "No structured fields detected. Provide clearer key:value lines for best results."
    )

    suggestions = list(dict.fromkeys(improvements))  # de-duplicate

    structured = {
        "fields": filled,
        "corrected": corrected,
        "missing": missing,
        "confidence": "heuristic",
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    return AnalyzeResponse(
        filled_fields=filled,
        corrected_fields=corrected,
        missing_fields=missing,
        suggested_improvements=suggestions,
        final_clean_version=final_text,
        structured_json=structured,
    )


# ---------- Routes ----------
@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": [],
    }

    try:
        from database import db  # type: ignore

        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, "name") else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:  # noqa: BLE001
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:  # noqa: BLE001
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """Lightweight heuristic analyzer that extracts key:value pairs, validates common fields, and returns a clean, acceptance-oriented result."""
    return analyze_text(req.content)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
