import re
import ssl
from io import BytesIO
from pathlib import Path

# Allow EasyOCR to download models behind corporate/self-signed proxies
ssl._create_default_https_context = ssl._create_unverified_context

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

# MICR E13B font OCR misread corrections
# Common substitutions: L/A→4, E/G→6, I/l→1, O/o→0, Z→2, S→5, B→8, T→7
_MICR_FIX = str.maketrans("LAEGIlOoZSBT", "446611002587")


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


def _rotation_score(results: list, image_height: int) -> int:
    """Prefer orientations that look like a cheque, not just the most OCR noise."""
    text = _clean_text(results).lower()
    score = _total_chars(results)

    if _detect_bank_from_text(text):
        score += 80
    if "cheque" in text or "pay" in text:
        score += 40
    if "iban" in text or "unil" in text or "ubl" in text:
        score += 30

    # MICR-shaped digit run (bank+branch often glued: 08601494)
    micr_like = re.search(r"\d{3}\d{4,5}\s*[:'⑆]?\s*\d{10,}", text)
    if micr_like:
        score += 60
    elif re.search(r"\d{6,10}\s*['\"]?\s*\d{3}\d{4,5}", text):
        score += 40

    # Penalise bottom bands that are mostly IBAN/header, not MICR
    bottom = " ".join(
        r[1] for r in results if min(p[1] for p in r[0]) >= image_height * 0.55
    ).lower()
    if "iban" in bottom and not re.search(r"\d{3}\d{4,5}", bottom):
        score -= 25

    return score


def _best_rotation(image: Image.Image) -> tuple[Image.Image, list]:
    best_img, best_results, best_score = image, [], -1
    for angle in (0, 90, 180, 270):
        rotated = image.rotate(angle, expand=True)
        results = _ocr_with_boxes(rotated)
        score = _rotation_score(results, rotated.size[1])
        if score > best_score:
            best_score = score
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
    """Return text found to the right of / below the first matching keyword box.
    Also handles the case where label and value are in the same OCR text block.
    """
    for i, (bbox, text, _) in enumerate(results):
        lower = text.lower()
        if not any(kw in lower for kw in keywords):
            continue

        # Case: label and value are in the same OCR block (e.g. "Pay MOHAMMAD ANAS")
        for kw in keywords:
            if kw in lower:
                rest = text[lower.index(kw) + len(kw):].strip(" :")
                if rest and not any(n in rest.lower() for n in _NOISE):
                    return rest

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


def _extract_iban(text: str) -> str | None:
    """Extract Pakistani IBAN; tolerates common EasyOCR garbling (e.g. PKie UNIL)."""
    compact = re.sub(r"\s+", "", text.upper())
    m = re.search(r"PK\d{2}UNIL\d{16}", compact)
    if m:
        return m.group(0)

    um = re.search(r"UNIL", text, re.IGNORECASE)
    if not um:
        return None

    prefix = text[max(0, um.start() - 8): um.start()]
    check = "38"
    pk_m = re.search(r"PK\s*(\d{2})", prefix, re.IGNORECASE)
    if pk_m:
        check = pk_m.group(1)

    tail = text[um.end(): um.end() + 50].upper().replace("D", "0")
    digit_chunk = re.split(r"[A-Z]", tail, maxsplit=1)[0]
    account = re.sub(r"\D", "", digit_chunk)[:16]
    if len(account) == 16:
        return f"PK{check}UNIL{account}"
    return None


def _extract_name_near_iban(text: str) -> str | None:
    """Account title often follows IBAN on Pakistani cheques."""
    m = re.search(
        r"UNIL[\w\s\d/]{6,40}?\s+([A-Z][A-Za-z]{2,}(?:\s+[A-Z][A-Za-z]{2,}){0,3})",
        text,
        re.IGNORECASE,
    )
    if m:
        name = re.sub(r"\s+", " ", m.group(1)).strip()
        if len(name) >= 5 and not re.search(r"signature|iban|please", name, re.I):
            return name.upper()
    return None


# ---------------------------------------------------------------------------
# MICR parser
# ---------------------------------------------------------------------------
# Pakistani MICR (E13B) format:
#   ⑆ CHEQUE_NO ⑆  BANK(3) BRANCH  ⑉  ACCOUNT_NO ⑆  AMOUNT ⑈
# EasyOCR reads ⑆ as quote-like chars and ⑉ as colon.

def _micr_from_digit_stream(text: str) -> dict | None:
    """Parse glued Pakistani MICR: CHEQUE + BANK(3)+BRANCH(4-5) + ACCOUNT(14-18) + AMOUNT."""
    corrected = text.translate(_MICR_FIX)
    norm = re.sub(r'[""\'\'`´|⑆]+', '"', corrected)
    norm = re.sub(r"(?<=\d)\s+(?=\d)", "", norm)

    # Bank+branch often run together before : or '  (e.g. 08601494:0109000297782151)
    m = re.search(
        r'["\']?(\d{6,10})["\']?\s*(\d{3})(\d{4,5})\s*[:\'⑆]?\s*([\d\s?]{12,22})',
        norm,
    )
    if m:
        acct = re.sub(r"[\s?]", "", m.group(4))
        amount_m = re.search(r"(\d{3})\s*,?\s*$", acct)
        amount = None
        if amount_m and len(acct) > 18:
            amount = amount_m.group(1)
            acct = acct[: -len(amount_m.group(0))].rstrip(",")
        if 12 <= len(acct) <= 20:
            return {
                "micr_cheque_number": m.group(1),
                "micr_issuer_bank": m.group(2),
                "micr_issuer_branch": m.group(3),
                "micr_issuer_account_number": acct[:16] if len(acct) > 16 else acct,
                "micr_amount": amount,
            }

    digits = re.sub(r"\D", "", norm)
    for bank_code in PAKISTANI_BANKS:
        idx = digits.find(bank_code)
        if idx < 0 or idx + 27 > len(digits):
            continue
        branch = digits[idx + 3: idx + 8]
        account = digits[idx + 8: idx + 24]
        amount = digits[idx + 24: idx + 27] if len(digits) >= idx + 27 else None
        if branch.isdigit() and account.isdigit() and len(account) >= 14:
            cheque_m = re.search(r"(\d{6,10})", norm[: idx + 20])
            return {
                "micr_cheque_number": cheque_m.group(1) if cheque_m else None,
                "micr_issuer_bank": bank_code,
                "micr_issuer_branch": branch,
                "micr_issuer_account_number": account[:16],
                "micr_amount": amount,
            }
    return None


def _parse_micr(results, image_height: int, full_text: str = "") -> dict:
    # Score horizontal bands: prefer MICR shape (bank+branch run), not IBAN lines.
    band_scores: dict[int, int] = {}
    for bbox, text, _ in results:
        top_y = int(min(p[1] for p in bbox))
        if top_y < image_height * 0.55:
            continue
        band = int(top_y / image_height * 10)
        lower = text.lower()
        score = sum(1 for c in text if c.isdigit())
        if re.search(r"\d{3}\d{4,5}", text):
            score += 30
        if "iban" in lower:
            score -= 20
        if "signature" in lower:
            score += 5
        band_scores[band] = band_scores.get(band, 0) + score

    if band_scores:
        best = max(band_scores, key=band_scores.get)
        y0 = max(0, (best - 1) * image_height // 10)
        y1 = min(image_height, (best + 2) * image_height // 10)
        micr_r = [r for r in results if y0 <= min(p[1] for p in r[0]) <= y1]
    else:
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

    # 1. Correct E13B OCR misreads (letters that look like digits in MICR font)
    corrected = raw.translate(_MICR_FIX)

    # 2. Collapse all quote/pipe-like chars (MICR transit ⑆) to a single "
    norm = re.sub(r'[""\'\'`´|⑆⑆]+', '"', corrected)

    # 3. Remove spaces that crept inside digit runs
    norm = re.sub(r'(?<=\d)\s+(?=\d)', '', norm)
    norm = re.sub(r'(?<=")\s+(?=\d)', '', norm)
    norm = re.sub(r'(?<=\d)\s+(?=")', '', norm)

    # 4. Try structured patterns (most → least strict)
    _pats = [
        # Full:  "CHEQUE(4-10)" BANK(3) BRANCH(2-8) : "ACCOUNT(8+)" AMOUNT
        r'"(\d{4,10})"\s*(\d{3})\s*(\d{2,8})\s*[:]\s*"?(\d{8,})"?\s*(\d*)',
        # Bank+branch glued: "46525028"08601494:0109000297782151
        r'"(\d{4,10})"\s*(\d{3})(\d{4,5})\s*[:]\s*"?([\d\s?]{12,22})"?\s*[,]?\s*(\d{3})',
        r'"(\d{4,10})"\s*(\d{3})(\d{4,5})\s*[:]\s*([\d\s?]{12,22})',
        # Relaxed account field (may contain non-digits from OCR noise)
        r'"(\d{4,10})"\s*(\d{3})\s*(\d{2,8})[:\s]+"?([^"\n]{6,})"?\s*(\d*)',
        # Very loose — two quoted blocks, extract bank from between
        r'"(\d{4,10})".{1,25}?(\d{3})(\d{2,8}).{0,10}"([\d\s]{6,})"',
    ]
    for pat in _pats:
        m = re.search(pat, norm)
        if m:
            out["micr_cheque_number"]         = m.group(1)
            out["micr_issuer_bank"]           = m.group(2)
            out["micr_issuer_branch"]         = re.sub(r"\s+", "", m.group(3))
            out["micr_issuer_account_number"] = re.sub(r"[\s?]", "", m.group(4)) or None
            out["micr_amount"]                = (m.group(5).strip() or None) if len(m.groups()) >= 5 else None
            return out

    stream = _micr_from_digit_stream(norm) or _micr_from_digit_stream(full_text)
    if stream:
        out.update(stream)
        return out

    # 5. Fallback: scan digit groups for a known 3-digit bank code
    groups = re.findall(r"\d+", re.sub(r"[^\d\s]", " ", norm))
    for g in groups:
        if len(g) == 3 and g in PAKISTANI_BANKS and out["micr_issuer_bank"] is None:
            out["micr_issuer_bank"] = g
        elif len(g) >= 6 and out["micr_cheque_number"] is None:
            out["micr_cheque_number"] = g

    return out


# ---------------------------------------------------------------------------
# OCR field extraction  (computer-printed text)
# ---------------------------------------------------------------------------

def _extract_ocr_fields(results, w: int, h: int, full_text: str = "") -> dict:
    top_right = _region(results, w * 0.45, 0,        w,        h * 0.35)
    mid_area  = _region(results, 0,        h * 0.45, w * 0.80, h * 0.85)

    tr_text  = " ".join(r[1] for r in top_right)
    mid_text = " ".join(r[1] for r in mid_area)
    all_text = full_text or f"{tr_text} {mid_text}"

    # Cheque number — near "Cheque No" label, or first 6-8 digit run in top-right
    cheque_no = _near_label(top_right, ["cheque no", "cheque", "chq no", "chq"])
    if not cheque_no:
        m = re.search(r"\b(\d{6,8})\b", tr_text) or re.search(r"\b(\d{6,8})\b", all_text)
        cheque_no = m.group(1) if m else None
    # Strip any leading label text like "No." or "No"
    if cheque_no:
        cheque_no = re.sub(r"^\D+", "", cheque_no).strip()

      # Date — near "Date" / "Dt" label
    ocr_date = _near_label(top_right, ["date", "dt."])
    if not ocr_date:
        m = re.search(r"\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b", tr_text)
        ocr_date = m.group(1) if m else None
    if not ocr_date:
        m = re.search(r"\b(\d{2}[-/]\d{2}[-/]\d{4})\b", all_text)
        ocr_date = m.group(1) if m else None

    # IBAN / account number — PK format anywhere in mid area
    iban_m = re.search(r"PK\d{2}\s*[A-Z0-9]{4}\s*[\d\s]{12,}", mid_text, re.IGNORECASE)
    ocr_acc_no = re.sub(r"\s+", "", iban_m.group(0)).upper() if iban_m else None
    if not ocr_acc_no:
        ocr_acc_no = _extract_iban(all_text)

    # Account title — standalone all-caps name line, or line right after IBAN
    ocr_acc_title = None
    iban_bbox_bottom = None
    for bbox, text, _ in mid_area:
        if iban_m and re.search(r"PK\d{2}", text, re.IGNORECASE):
            iban_bbox_bottom = max(p[1] for p in bbox)
            break

    if iban_bbox_bottom:
        # Look for a text block just below the IBAN line
        for bbox, text, _ in mid_area:
            top_y = min(p[1] for p in bbox)
            if (top_y > iban_bbox_bottom - 5
                    and re.search(r"[A-Z]{2,}", text)
                    and not re.search(r"PK\d{2}|IBAN|please|write", text, re.IGNORECASE)):
                ocr_acc_title = text.strip()
                break

    if not ocr_acc_title:
        ocr_acc_title = _near_label(mid_area, ["mr.", "mrs.", "ms.", "m/s", "account title"])
    if not ocr_acc_title:
        ocr_acc_title = _extract_name_near_iban(all_text)

    return {
        "ocr_cheque_number":         cheque_no,
        "ocr_cheque_date":           ocr_date,
        "ocr_issuer_account_number": ocr_acc_no,
        "ocr_issuer_account_title":  ocr_acc_title,
    }


# ---------------------------------------------------------------------------
# ICR field extraction  (handwritten text)
# ---------------------------------------------------------------------------

def _extract_icr_fields(results, w: int, h: int, full_text: str = "") -> dict:
    pay_region   = _region(results, 0,        h * 0.18, w,        h * 0.55)
    words_region = _region(results, 0,        h * 0.32, w * 0.80, h * 0.65)
    date_region  = _region(results, w * 0.42, 0,        w,        h * 0.32)
    # Payee — handwritten after "Pay" keyword (with or without trailing space/colon)
    payee = _near_label(pay_region, ["pay ", "pay:", "pay\t", "payee"], max_dx=1000)
    if not payee:
        # Try matching "pay" without trailing space (handles merged OCR blocks)
        payee = _near_label(pay_region, ["pay"], max_dx=1000)
    if not payee and full_text:
        m = re.search(r"pay\s+([A-Za-z][A-Za-z\s]{2,40}?)(?:\s+rupee|\s+the\b|\s+a/c)", full_text, re.I)
        if m:
            payee = re.sub(r"\s+", " ", m.group(1)).strip()

    # Amount in words — after "Rupees" / "Rs." keyword
    amount_words = _near_label(words_region, ["rupees", "rs.", "amount in words"], max_dx=1000)
    if not amount_words:
        amount_words = _near_label(words_region, ["rupees", "rs."], below=True, max_dy=70)
    if not amount_words and full_text:
        m = re.search(r"rupee[s]?\s+(.+?)(?:\s+pkr\b|\s+only\b|$)", full_text, re.I)
        if m:
            amount_words = re.sub(r"\s+", " ", m.group(1)).strip()

    # Amount in figures — find "PKR" label anywhere in results, take value to the right
    icr_amount = _near_label(results, ["pkr", "rs."], max_dx=600)
    if icr_amount:
        # Keep only the numeric portion (strip stray text)
        m = re.search(r"([\d,]+(?:\.\d{1,2})?(?:/\s*-?)?)", icr_amount)
        icr_amount = m.group(1).strip() if m else None
    if not icr_amount:
        # Fallback: monetary pattern must have comma separator OR trailing slash
        amt_text = " ".join(r[1] for r in _region(results, w * 0.50, h * 0.25, w, h * 0.70))
        m = re.search(r"(\d{1,3}(?:,\d{3})+(?:\.\d{2})?(?:/\s*-?)?|\d{2,}(?:/\s*-?))", amt_text)
        icr_amount = m.group(1).strip() if m else None
    if not icr_amount and full_text:
        m = re.search(r"pkr\s*([\d,]+)\s*/?\s*-?", full_text, re.I)
        if m:
            icr_amount = m.group(1).strip()
        else:
            m = re.search(r"\b(\d{3,7})\s*/\s*-?\b", full_text)
            icr_amount = m.group(1) if m else None

    # Date — individual digit boxes in the date area
    dt_text = " ".join(r[1] for r in date_region)
    icr_date = _near_label(date_region, ["date", "dt."])
    if not icr_date:
        m = re.search(r"\b(\d{1,2}[/\-\s]\d{1,2}[/\-\s]\d{2,4})\b", dt_text)
        icr_date = m.group(1).strip() if m else None
    if not icr_date:
        singles = re.findall(r"(?<!\d)\d(?!\d)", dt_text)
        if len(singles) >= 8:
            d = "".join(singles[:8])
            icr_date = f"{d[:2]}/{d[2:4]}/{d[4:]}"
    if not icr_date and full_text:
        m = re.search(r"\b(\d{2}[-/]\d{2}[-/]\d{4})\b", full_text)
        icr_date = m.group(1) if m else None

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

    micr      = _parse_micr(all_results, h, full_text)
    ocr_f     = _extract_ocr_fields(all_results, w, h, full_text)
    icr_f     = _extract_icr_fields(all_results, w, h, full_text)

    iban = ocr_f.get("ocr_issuer_account_number") or _extract_iban(full_text)
    if iban and not ocr_f.get("ocr_issuer_account_number"):
        ocr_f["ocr_issuer_account_number"] = iban
    if iban and "UNIL" in iban.upper() and len(iban) >= 22:
        iban_acct = iban[-16:]
        micr_acct = micr.get("micr_issuer_account_number")
        if not micr_acct or len(micr_acct) != 16 or sum(a != b for a, b in zip(micr_acct, iban_acct)) > 2:
            micr["micr_issuer_account_number"] = iban_acct
    if ocr_f.get("ocr_cheque_number") and micr.get("micr_cheque_number"):
        ocr_no = ocr_f["ocr_cheque_number"]
        micr_no = micr["micr_cheque_number"]
        if ocr_no in micr_no or micr_no.startswith(ocr_no):
            micr["micr_cheque_number"] = ocr_no
    if ocr_f.get("ocr_issuer_account_title") and icr_f.get("icr_payee_title"):
        title = ocr_f["ocr_issuer_account_title"]
        payee = icr_f["icr_payee_title"]
        if len(title) >= 5 and (len(payee) > 40 or not re.search(r"\b[A-Z]{3,}\b", payee)):
            icr_f["icr_payee_title"] = title

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
