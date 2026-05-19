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

# 3-digit NIFT bank codes sourced from Banks.txt
PAKISTANI_BANKS: dict[str, str] = {
    "001": "STATE BANK OF PAK.",
    "010": "ZARAI TARAQIATI BANK",
    "013": "DUBAI ISLAMIC BANK",
    "014": "ALLIED BANK LTD",
    "017": "ASKARI BANK LTD",
    "018": "JS BANK LTD",
    "021": "BANK ISLAMI PAK.",
    "023": "BANK AL-HABIB LTD",
    "028": "SAMBA BANK LTD",
    "031": "ALBARAKA BANK LTD",
    "037": "BANK OF CHINA",
    "038": "STD. CHARTERED BANK",
    "041": "TELENOR MF BANK",
    "042": "DEUTSCHE BANK AG PAKISTAN",
    "043": "SINDH BANK LTD",
    "046": "CITI BANK NA",
    "047": "FIRST WOMEN BANK LTD",
    "053": "BANK AL-FALAH LTD",
    "054": "HABIB BANK LTD",
    "058": "INDUSTRIAL DEVELOPMENT BANK",
    "060": "FAYSAL BANK LTD",
    "061": "BANK OF KHYBER",
    "062": "MCB BANK LTD",
    "063": "MCB ISLAMIC BANK LTD",
    "064": "HABIB METRO BANK LTD",
    "066": "SILK BANK LTD",
    "068": "SME BANK LTD",
    "070": "NATIONAL BANK OF PAK",
    "081": "BANK MAKRAMAH LTD",
    "083": "BANK OF PUNJAB",
    "085": "SONERI BANK LTD",
    "086": "UNITED BANK LTD",
    "088": "I&C BANK OF CHINA",
    "089": "MEEZAN BANK LTD",
    "090": "CDNS-NAT. SAVING ORG",
    "099": "MUSHRIK BANK",
    "101": "ADVANS MF BANK LTD",
    "102": "APNA MF BANK",
    "103": "FINCA MF BANK",
    "104": "HBL MF BANK LTD",
    "105": "KHUSHHALI MF BANK LTD",
    "106": "NRSP MF BANK",
    "107": "MOBILINK MF BANK",
    "108": "U MF BANK LTD",
}

BANK_KEYWORDS: dict[str, list[str]] = {
    "STATE BANK OF PAK.":         ["state bank", "sbp"],
    "ZARAI TARAQIATI BANK":       ["zarai taraqiati", "ztbl"],
    "DUBAI ISLAMIC BANK":         ["dubai islamic", "dib"],
    "ALLIED BANK LTD":            ["allied bank", "abl"],
    "ASKARI BANK LTD":            ["askari"],
    "JS BANK LTD":                ["js bank", "jsbl"],
    "BANK ISLAMI PAK.":           ["bank islami"],
    "BANK AL-HABIB LTD":          ["bank al-habib", "bank al habib", "bahl"],
    "SAMBA BANK LTD":             ["samba"],
    "ALBARAKA BANK LTD":          ["albaraka", "al baraka"],
    "BANK OF CHINA":              ["bank of china"],
    "STD. CHARTERED BANK":        ["standard chartered", "stanchart", "std. chartered"],
    "TELENOR MF BANK":            ["telenor"],
    "DEUTSCHE BANK AG PAKISTAN":  ["deutsche bank"],
    "SINDH BANK LTD":             ["sindh bank"],
    "CITI BANK NA":               ["citibank", "citi bank"],
    "FIRST WOMEN BANK LTD":       ["first women bank"],
    "BANK AL-FALAH LTD":          ["alfalah", "al falah", "bank alfalah"],
    "HABIB BANK LTD":             ["habib bank", "hbl"],
    "INDUSTRIAL DEVELOPMENT BANK":["industrial development bank", "idbp"],
    "FAYSAL BANK LTD":            ["faysal"],
    "BANK OF KHYBER":             ["bank of khyber", "bok"],
    "MCB BANK LTD":               ["mcb bank", "muslim commercial", "mcb"],
    "MCB ISLAMIC BANK LTD":       ["mcb islamic"],
    "HABIB METRO BANK LTD":       ["habib metro", "habibmetro", "hmb"],
    "SILK BANK LTD":              ["silk bank", "silkbank"],
    "SME BANK LTD":               ["sme bank"],
    "NATIONAL BANK OF PAK":       ["national bank", "nbp"],
    "BANK MAKRAMAH LTD":          ["bank makramah"],
    "BANK OF PUNJAB":             ["bank of punjab", "bop"],
    "SONERI BANK LTD":            ["soneri"],
    "UNITED BANK LTD":            ["united bank", "ubl"],
    "I&C BANK OF CHINA":          ["i&c bank", "icbc", "industrial commercial bank"],
    "MEEZAN BANK LTD":            ["meezan"],
    "CDNS-NAT. SAVING ORG":       ["cdns", "national saving"],
    "MUSHRIK BANK":               ["mushrik"],
    "ADVANS MF BANK LTD":         ["advans"],
    "APNA MF BANK":               ["apna"],
    "FINCA MF BANK":              ["finca"],
    "HBL MF BANK LTD":            ["hbl microfinance", "hbl mf"],
    "KHUSHHALI MF BANK LTD":      ["khushhali"],
    "NRSP MF BANK":               ["nrsp"],
    "MOBILINK MF BANK":           ["mobilink"],
    "U MF BANK LTD":              ["u microfinance", "u mf"],
}

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
_NOISE = {"camscanner", "cs", "adobe", "scan", "scanned", "please do not write"}


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _enhance(img: Image.Image) -> np.ndarray:
    gray = img.convert("L")
    enhanced = ImageEnhance.Contrast(gray).enhance(2.5)
    sharpened = enhanced.filter(ImageFilter.SHARPEN)
    return np.array(sharpened)


def _ocr_with_boxes(img: Image.Image):
    return _reader.readtext(_enhance(img), detail=1, paragraph=False)


def _total_chars(results) -> int:
    return sum(len(r[1]) for r in results)


def _best_rotation(image: Image.Image) -> tuple[Image.Image, list]:
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


# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------

def _bbox_center(bbox) -> tuple[float, float]:
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _region(results, x0: float, y0: float, x1: float, y1: float) -> list:
    """Return OCR results whose center falls within the rectangle."""
    out = []
    for bbox, text, conf in results:
        cx, cy = _bbox_center(bbox)
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            out.append((bbox, text, conf))
    return out


def _near_label(results, keywords: list[str], *, right=True, below=False,
                max_dx: float = 600, max_dy: float = 45) -> str | None:
    """Return text found to the right of / below the first matching keyword box."""
    for i, (bbox, text, _) in enumerate(results):
        if not any(kw in text.lower() for kw in keywords):
            continue
        label_rx = max(p[0] for p in bbox)
        label_by = max(p[1] for p in bbox)
        _, label_cy = _bbox_center(bbox)

        candidates: list[tuple[float, str]] = []
        for j, (ob, ot, _) in enumerate(results):
            if i == j:
                continue
            if any(kw in ot.lower() for kw in keywords):
                continue
            if any(n in ot.lower() for n in _NOISE):
                continue
            ocx, ocy = _bbox_center(ob)
            olx = min(p[0] for p in ob)
            oty = min(p[1] for p in ob)

            if right and olx >= label_rx - 10 and abs(ocy - label_cy) <= max_dy and 0 < ocx - label_rx <= max_dx:
                candidates.append((ocx, ot))
            if below and oty >= label_by - 5 and abs(ocx - label_rx) <= max_dx and 0 < ocy - label_cy <= max_dy * 3:
                candidates.append((ocy * 10000 + ocx, ot))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            return " ".join(c[1] for c in candidates[:6]).strip()
    return None


# ---------------------------------------------------------------------------
# Bank detection from printed text
# ---------------------------------------------------------------------------

def _detect_bank_from_text(text: str) -> str | None:
    lower = text.lower()
    for bank_name, keywords in BANK_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return bank_name
    return None


# ---------------------------------------------------------------------------
# MICR parser
# ---------------------------------------------------------------------------
# Pakistani MICR (E13B) format:
#   ⑆ CHEQUE_NO ⑆  BANK(3) BRANCH  ⑉  ACCOUNT_NO ⑆  AMOUNT ⑈
# EasyOCR typically reads ⑆ as " or | and ⑉ as :

def _parse_micr(results, image_height: int) -> dict:
    micr_r = [r for r in results if min(p[1] for p in r[0]) > image_height * 0.72]
    raw = " ".join(r[1] for r in micr_r).strip()

    out: dict = {
        "micr_raw": raw or None,
        "micr_cheque_number": None,
        "micr_issuer_bank": None,
        "micr_issuer_branch": None,
        "micr_issuer_account_number": None,
        "micr_amount": None,
    }

    if not raw:
        return out

    # Normalise transit symbol (⑆) variants that OCR produces
    norm = re.sub(r'[""''`´|⑆⑆]', '"', raw)

    # Patterns from most to least specific
    # Format: "CHEQUE_NO" BANK(3) BRANCH : "ACCOUNT" AMOUNT
    _pats = [
        r'"(\d{4,10})"\s*(\d{3})\s*(\d[\d\s]{1,8}?)\s*[:⑇]\s*"([^"]{6,})"\s*(\d*)',
        r'"(\d{4,10})"\s*(\d{3})\s*(\d+)[:⑇]"([^"]{6,})"\s*(\d*)',
        r'"(\d{4,10})".{0,15}(\d{3})(\d{3,8}).{0,6}"([\d\s?]{6,})"\s*(\d*)',
    ]
    for pat in _pats:
        m = re.search(pat, norm)
        if m:
            out["micr_cheque_number"]        = m.group(1).strip()
            out["micr_issuer_bank"]          = m.group(2).strip()
            out["micr_issuer_branch"]        = re.sub(r"\s+", "", m.group(3))
            out["micr_issuer_account_number"] = re.sub(r"[\s?]", "", m.group(4)) or None
            out["micr_amount"]               = m.group(5).strip() or None
            return out

    # Fallback: heuristic digit-group scan
    groups = re.findall(r"\d+", re.sub(r"[^\d\s]", " ", raw))
    for g in groups:
        if len(g) == 3 and g in PAKISTANI_BANKS and out["micr_issuer_bank"] is None:
            out["micr_issuer_bank"] = g
        elif len(g) >= 6 and out["micr_cheque_number"] is None:
            out["micr_cheque_number"] = g

    return out


# ---------------------------------------------------------------------------
# OCR field extraction  (computer-printed text)
# ---------------------------------------------------------------------------

def _extract_ocr_fields(results, w: int, h: int) -> dict:
    top_right = _region(results, w * 0.45, 0,        w,        h * 0.30)
    mid_area  = _region(results, 0,        h * 0.45, w * 0.80, h * 0.82)

    tr_text = " ".join(r[1] for r in top_right)
    mid_text = " ".join(r[1] for r in mid_area)

    # Cheque number — near "Cheque No" label or standalone 6-8 digit run
    cheque_no = _near_label(top_right, ["cheque no", "cheque", "chq"])
    if not cheque_no:
        m = re.search(r"\b(\d{6,8})\b", tr_text)
        cheque_no = m.group(1) if m else None

    # Date — near "Date" / "Dt" label (printed date if any)
    ocr_date = _near_label(top_right, ["date", "dt."])
    if not ocr_date:
        m = re.search(r"\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b", tr_text)
        ocr_date = m.group(1) if m else None

    # IBAN / account number in the middle area
    iban_m = re.search(r"PK\d{2}\s*[A-Z0-9]{4}\s*[\d\s]{12,}", mid_text, re.IGNORECASE)
    ocr_acc_no = re.sub(r"\s+", "", iban_m.group(0)) if iban_m else None

    # Account title — line containing Mr/Mrs/MR or near "account title" label
    ocr_acc_title = _near_label(mid_area, ["mr.", "mrs.", "ms.", "m/s", "account title", "a/c title"])
    if not ocr_acc_title:
        for _, text, _ in mid_area:
            if re.match(r"(mr\.?|mrs\.?|ms\.?|m/s)\s+", text, re.IGNORECASE):
                ocr_acc_title = text
                break

    return {
        "ocr_cheque_number":        cheque_no,
        "ocr_cheque_date":          ocr_date,
        "ocr_issuer_account_number": ocr_acc_no,
        "ocr_issuer_account_title": ocr_acc_title,
    }


# ---------------------------------------------------------------------------
# ICR field extraction  (handwritten text)
# ---------------------------------------------------------------------------

def _extract_icr_fields(results, w: int, h: int) -> dict:
    pay_region    = _region(results, 0,        h * 0.20, w,        h * 0.52)
    words_region  = _region(results, 0,        h * 0.33, w * 0.78, h * 0.62)
    amount_region = _region(results, w * 0.55, h * 0.28, w,        h * 0.65)
    date_region   = _region(results, w * 0.45, 0,        w,        h * 0.28)

    # Payee name — handwritten after "Pay" label
    payee = _near_label(pay_region, ["pay ", "pay:", "payee"], max_dx=900)

    # Amount in words — handwritten after "Rupees" / "Rs."
    amount_words = _near_label(words_region, ["rupees", "rs.", "amount in words"], max_dx=900)
    if not amount_words:
        amount_words = _near_label(words_region, ["rupees", "rs."], below=True, max_dy=60)

    # Amount in figures — PKR box on the right
    amt_text = " ".join(r[1] for r in amount_region)
    m = re.search(r"(?:PKR|Rs\.?)\s*([\d,]+(?:\.\d{1,2})?(?:/\s*-?)?)", amt_text, re.IGNORECASE)
    icr_amount = m.group(1).strip() if m else None
    if not icr_amount:
        m = re.search(r"([\d,]{4,}(?:\.\d{1,2})?(?:/\s*-?)?)", amt_text)
        icr_amount = m.group(1).strip() if m else None

    # Date — handwritten digits in date boxes
    dt_text = " ".join(r[1] for r in date_region)
    icr_date = _near_label(date_region, ["date", "dt."])
    if not icr_date:
        m = re.search(r"\b(\d{1,2}[/\-\s]\d{1,2}[/\-\s]\d{2,4})\b", dt_text)
        icr_date = m.group(1).strip() if m else None
    if not icr_date:
        # Cheque date boxes produce individual digits — collect and format
        singles = re.findall(r"(?<!\d)\d(?!\d)", dt_text)
        if len(singles) >= 8:
            d = "".join(singles[:8])
            icr_date = f"{d[:2]}/{d[2:4]}/{d[4:]}"

    return {
        "icr_payee_title":  payee,
        "icr_amount_words": amount_words,
        "icr_amount":       icr_amount,
        "icr_cheque_date":  icr_date,
    }


# ---------------------------------------------------------------------------
# Bank match validation
# ---------------------------------------------------------------------------

def _banks_match(code_bank: str, logo_bank: str) -> bool:
    a_keys = BANK_KEYWORDS.get(code_bank, [code_bank.lower()])
    b_keys = BANK_KEYWORDS.get(logo_bank, [logo_bank.lower()])
    return any(k in logo_bank.lower() for k in a_keys) or any(
        k in code_bank.lower() for k in b_keys
    )


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

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

    rotated, all_results = _best_rotation(image)
    w, h = rotated.size

    full_text = _clean_text(all_results)

    micr      = _parse_micr(all_results, h)
    ocr_f     = _extract_ocr_fields(all_results, w, h)
    icr_f     = _extract_icr_fields(all_results, w, h)

    bank_code          = micr["micr_issuer_bank"]
    bank_name_from_code = PAKISTANI_BANKS.get(bank_code) if bank_code else None
    logo_bank          = _detect_bank_from_text(full_text)

    all_digits = re.findall(r"\d{4,}", re.sub(r"[^\d\s]", " ", full_text))
    is_cheque  = len(all_digits) >= 2 and (bank_code is not None or logo_bank is not None)

    is_match: bool | None = None
    if bank_name_from_code and logo_bank:
        is_match = _banks_match(bank_name_from_code, logo_bank)

    confidence = (
        "high"   if (bank_code and logo_bank) else
        "medium" if (bank_code or  logo_bank) else
        "low"
    )

    return {
        # Validation summary
        "is_cheque":   is_cheque,
        "confidence":  confidence,
        "is_match":    is_match,
        "logo_bank":   logo_bank,
        # MICR fields
        "micr_cheque_number":         micr["micr_cheque_number"],
        "micr_issuer_bank":           bank_code,
        "micr_issuer_bank_name":      bank_name_from_code,
        "micr_issuer_branch":         micr["micr_issuer_branch"],
        "micr_issuer_account_number": micr["micr_issuer_account_number"],
        "micr_amount":                micr["micr_amount"],
        "micr_raw":                   micr["micr_raw"],
        # OCR fields (computer-printed)
        "ocr_cheque_number":          ocr_f["ocr_cheque_number"],
        "ocr_cheque_date":            ocr_f["ocr_cheque_date"],
        "ocr_issuer_account_number":  ocr_f["ocr_issuer_account_number"],
        "ocr_issuer_account_title":   ocr_f["ocr_issuer_account_title"],
        # ICR fields (handwritten)
        "icr_payee_title":   icr_f["icr_payee_title"],
        "icr_amount_words":  icr_f["icr_amount_words"],
        "icr_amount":        icr_f["icr_amount"],
        "icr_cheque_date":   icr_f["icr_cheque_date"],
    }


_static = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static)), name="static")


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(str(_static / "index.html"))
