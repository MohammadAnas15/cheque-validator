import re
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageEnhance, ImageFilter
import easyocr

app = FastAPI(title="Pakistani Bank Cheque Validator")

_reader = easyocr.Reader(["en"], gpu=False, verbose=False)

PAKISTANI_BANKS: dict[str, str] = {
    "0010": "National Bank of Pakistan (NBP)",
    "0020": "State Bank of Pakistan (SBP)",
    "0315": "Bank Alfalah",
    "0340": "Habib Bank Limited (HBL)",
    "0341": "Habib Bank Limited (HBL)",
    "0350": "Soneri Bank",
    "0360": "United Bank Limited (UBL)",
    "0380": "Standard Chartered Pakistan",
    "0450": "Allied Bank Limited (ABL)",
    "0500": "Bank of Khyber",
    "0550": "Faysal Bank",
    "0570": "JS Bank",
    "0620": "Citibank Pakistan",
    "0630": "MCB Bank Limited",
    "0650": "Habib Metropolitan Bank",
    "0700": "Dubai Islamic Bank Pakistan",
    "0706": "Meezan Bank",
    "0750": "Bank Al Habib",
    "0760": "Al Baraka Bank Pakistan",
    "0830": "Silk Bank",
    "0870": "FINCA Microfinance Bank",
    "0890": "Askari Bank",
    "0920": "National Bank of Pakistan (NBP)",
    "0960": "Bank of Punjab (BoP)",
    "0990": "SME Bank",
}

BANK_KEYWORDS: dict[str, list[str]] = {
    "National Bank of Pakistan (NBP)": ["national bank", "nbp"],
    "State Bank of Pakistan (SBP)": ["state bank", "sbp"],
    "Bank Alfalah": ["alfalah"],
    "Habib Bank Limited (HBL)": ["habib bank", "hbl"],
    "Soneri Bank": ["soneri"],
    "United Bank Limited (UBL)": ["united bank", "ubl"],
    "Standard Chartered Pakistan": ["standard chartered", "stanchart", "scb"],
    "Allied Bank Limited (ABL)": ["allied bank", "abl"],
    "Bank of Khyber": ["bank of khyber", "bok"],
    "Faysal Bank": ["faysal"],
    "JS Bank": ["js bank", "jsbl"],
    "Citibank Pakistan": ["citibank", "citi"],
    "MCB Bank Limited": ["mcb"],
    "Habib Metropolitan Bank": ["habib metro", "habibmetro", "hmb", "metropolitan"],
    "Dubai Islamic Bank Pakistan": ["dubai islamic", "dib"],
    "Meezan Bank": ["meezan"],
    "Bank Al Habib": ["bank al habib", "bahl"],
    "Al Baraka Bank Pakistan": ["al baraka", "baraka", "albaraka"],
    "Silk Bank": ["silk bank", "silkbank"],
    "FINCA Microfinance Bank": ["finca"],
    "Askari Bank": ["askari"],
    "Bank of Punjab (BoP)": ["bank of punjab", "bop"],
    "SME Bank": ["sme bank"],
}

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

# Noise words that should never be mistaken for MICR content
_NOISE = {"camscanner", "cs", "adobe", "scan", "scanned", "please do not write"}


def _enhance(img: Image.Image) -> np.ndarray:
    gray = img.convert("L")
    enhanced = ImageEnhance.Contrast(gray).enhance(2.5)
    sharpened = enhanced.filter(ImageFilter.SHARPEN)
    return np.array(sharpened)


def _ocr_with_boxes(img: Image.Image):
    """Return list of (bbox, text, confidence) for the image."""
    return _reader.readtext(_enhance(img), detail=1, paragraph=False)


def _total_chars(results) -> int:
    return sum(len(r[1]) for r in results)


def _best_rotation(image: Image.Image) -> tuple[Image.Image, list]:
    """
    Try 0 / 90 / 180 / 270 degree rotations; pick the one EasyOCR
    returns the most characters for. Returns (rotated_image, ocr_results).
    """
    best_img, best_results, best_count = image, [], 0
    for angle in (0, 90, 180, 270):
        rotated = image.rotate(angle, expand=True)
        results = _ocr_with_boxes(rotated)
        count = _total_chars(results)
        if count > best_count:
            best_count = count
            best_img = rotated
            best_results = results
    return best_img, best_results


def _clean_text(results) -> str:
    parts = [r[1] for r in results if not any(n in r[1].lower() for n in _NOISE)]
    return " ".join(parts)


def _detect_bank_from_text(text: str) -> str | None:
    lower = text.lower()
    for bank_name, keywords in BANK_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return bank_name
    return None


def _micr_region(image: Image.Image, results) -> tuple[Image.Image, list]:
    """
    Find the horizontal band that contains the most digit clusters —
    that is the MICR line. Falls back to the bottom 22 % of the image.
    """
    w, h = image.size

    # Group OCR boxes by vertical band (10 % slices)
    band_digits: dict[int, int] = {}
    for bbox, text, _ in results:
        top_y = int(min(p[1] for p in bbox))
        band = int(top_y / h * 10)
        digit_count = sum(1 for c in text if c.isdigit())
        band_digits[band] = band_digits.get(band, 0) + digit_count

    if band_digits:
        best_band = max(band_digits, key=band_digits.get)
        # Widen the crop by one band on each side
        y0 = max(0, (best_band - 1) * h // 10)
        y1 = min(h, (best_band + 2) * h // 10)
    else:
        y0, y1 = int(h * 0.78), h

    micr_crop = image.crop((0, y0, w, y1))
    micr_results = [r for r in results if min(p[1] for p in r[0]) >= y0]
    return micr_crop, micr_results


def _parse_micr(results) -> tuple[str | None, str | None]:
    """
    Pakistani MICR transit block = 10 digits: bank code (4) + branch code (6).
    Try first-4 and last-4 of any 10-digit group against PAKISTANI_BANKS.
    """
    texts = [r[1] for r in results]
    raw = " ".join(texts)

    # Strip noise
    digit_only_groups = re.findall(r"\d+", re.sub(r"[^\d\s]", " ", raw))

    bank_code: str | None = None

    # Priority 1: 10-digit group (full transit)
    for g in digit_only_groups:
        if len(g) == 10:
            if g[:4] in PAKISTANI_BANKS:
                bank_code = g[:4]
                break
            if g[6:] in PAKISTANI_BANKS:
                bank_code = g[6:]
                break

    # Priority 2: adjacent groups forming 10 digits
    if bank_code is None:
        for i in range(len(digit_only_groups) - 1):
            combined = digit_only_groups[i] + digit_only_groups[i + 1]
            if len(combined) == 10:
                if combined[:4] in PAKISTANI_BANKS:
                    bank_code = combined[:4]
                    break
                if combined[6:] in PAKISTANI_BANKS:
                    bank_code = combined[6:]
                    break

    # Priority 3: standalone 4-digit code
    if bank_code is None:
        for g in digit_only_groups:
            if len(g) == 4 and g in PAKISTANI_BANKS:
                bank_code = g
                break

    return raw.strip() or None, bank_code


def _banks_match(code_bank: str, logo_bank: str) -> bool:
    a_keys = BANK_KEYWORDS.get(code_bank, [code_bank.lower()])
    b_keys = BANK_KEYWORDS.get(logo_bank, [logo_bank.lower()])
    return any(k in logo_bank.lower() for k in a_keys) or any(
        k in code_bank.lower() for k in b_keys
    )


@app.post("/api/analyze")
async def analyze_cheque(file: UploadFile = File(...)) -> dict:
    media_type = file.content_type or "image/jpeg"
    if media_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{media_type}'. Upload JPEG, PNG, GIF, or WebP.",
        )

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10 MB.")

    image = Image.open(BytesIO(contents)).convert("RGB")

    # --- Auto-rotate to the orientation with most readable text ---
    rotated, all_results = _best_rotation(image)

    full_text = _clean_text(all_results)

    # --- Find MICR region ---
    _, micr_results = _micr_region(rotated, all_results)
    raw_micr, bank_code = _parse_micr(micr_results)

    # --- Detect bank from full image text ---
    logo_bank = _detect_bank_from_text(full_text)

    # --- Is this a cheque? Needs at least two digit groups of 4+ digits ---
    all_digits = re.findall(r"\d{4,}", re.sub(r"[^\d\s]", " ", full_text))
    is_cheque = len(all_digits) >= 2 and (bank_code is not None or logo_bank is not None)

    bank_name_from_code = PAKISTANI_BANKS.get(bank_code) if bank_code else None

    is_match: bool | None = None
    if bank_name_from_code and logo_bank:
        is_match = _banks_match(bank_name_from_code, logo_bank)

    confidence = (
        "high" if (bank_code and logo_bank)
        else "medium" if (bank_code or logo_bank)
        else "low"
    )

    notes = f"MICR OCR: {raw_micr[:150]}" if raw_micr else "No MICR digits detected."

    return {
        "is_cheque": is_cheque,
        "micr_line": raw_micr,
        "bank_code": bank_code,
        "logo_bank": logo_bank,
        "is_match": is_match,
        "confidence": confidence,
        "notes": notes,
        "bank_name_from_code": bank_name_from_code,
    }


_static = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static)), name="static")


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(str(_static / "index.html"))
