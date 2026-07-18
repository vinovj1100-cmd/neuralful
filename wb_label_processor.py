"""WB (Wildberries) Label Processor — Advanced Detection Engine
Handles vertical text, QR data extraction, multi-criteria matching, and
rotation-aware OCR for Wildberries shipping labels.

v4.1 Enhancement: Detects WB vertical tracking numbers, parses structured
QR payloads, and provides confidence-scored multi-criteria matching.
"""

import re
import io
import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import pypdf
from pdf2image import convert_from_bytes
from pyzbar.pyzbar import decode
import pytesseract


@dataclass
class WBLabelMatch:
    """Result of matching a label page against a target entry."""
    page_idx: int
    target_idx: int
    target_raw: str
    match_type: str  # 'exact', 'tracking', 'phone_code', 'qr_data', 'fuzzy'
    confidence: float  # 0.0 - 1.0
    matched_fields: Dict[str, str] = field(default_factory=dict)
    debug_info: Dict = field(default_factory=dict)


@dataclass
class WBLabelData:
    """Extracted data from a single WB label page."""
    page_idx: int
    tracking_number: Optional[str] = None
    wb_order_id: Optional[str] = None
    phone_number: Optional[str] = None
    delivery_code: Optional[str] = None
    address: Optional[str] = None
    qr_payloads: List[str] = field(default_factory=list)
    barcodes: List[str] = field(default_factory=list)
    raw_text: str = ""
    vertical_text: str = ""
    rotation_detected: int = 0  # degrees

    def to_dict(self):
        return {
            "page_idx": self.page_idx,
            "tracking_number": self.tracking_number,
            "wb_order_id": self.wb_order_id,
            "phone_number": self.phone_number,
            "delivery_code": self.delivery_code,
            "address": self.address,
            "qr_payloads": self.qr_payloads,
            "barcodes": self.barcodes,
            "rotation_detected": self.rotation_detected,
        }


class WBLabelProcessor:
    """Advanced processor for Wildberries shipping labels.

    Features:
      • Rotation-aware OCR (0°, 90°, 180°, 270°) for vertical text
      • QR code structured data parsing
      • Multi-criteria matching (tracking #, phone+code, order ID, QR payload)
      • Visual debug overlay generation
      • Confidence scoring per match
    """

    # WB tracking number patterns
    WB_TRACKING_RE = re.compile(r'WB[A-Z0-9]{10,20}')
    WB_ORDER_RE = re.compile(r'\b\d{9,15}\b')  # WB order IDs are typically 9-15 digits
    PHONE_RE = re.compile(r'\b\d{7}\b')  # 7-digit phone suffix
    CODE_RE = re.compile(r'\b\d{4}\b')   # 4-digit delivery code

    # Vertical text detection: WB labels often have "WB" and numbers printed vertically
    VERTICAL_WB_RE = re.compile(r'W\s*B|WB', re.IGNORECASE)

    def __init__(self, dpi: int = 300, ocr_psm: int = 6):
        self.dpi = dpi
        self.ocr_psm = ocr_psm
        self._detection_cache: Dict[int, WBLabelData] = {}

    def process_pdf(self, pdf_bytes: bytes) -> List[WBLabelData]:
        """Process all pages of a PDF and extract WB label data."""
        images = convert_from_bytes(pdf_bytes, dpi=self.dpi)
        results = []
        for idx, img in enumerate(images):
            data = self._process_page(img, idx)
            self._detection_cache[idx] = data
            results.append(data)
        return results

    def _process_page(self, img: Image.Image, page_idx: int) -> WBLabelData:
        """Process a single page/image and extract all WB-relevant data."""
        data = WBLabelData(page_idx=page_idx)
        w, h = img.size

        # 1. Decode barcodes and QR codes
        barcodes = decode(img)
        for bc in barcodes:
            decoded = bc.data.decode('utf-8', errors='ignore')
            data.barcodes.append(decoded)
            # Check if it's a WB tracking number
            if self.WB_TRACKING_RE.search(decoded):
                data.tracking_number = self.WB_TRACKING_RE.search(decoded).group()
            # QR codes often contain structured data
            if bc.type == 'QRCODE':
                data.qr_payloads.append(decoded)

        # 2. OCR at multiple rotations to catch vertical text
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        rotations = [0, 90, 180, 270]
        all_texts = []

        for angle in rotations:
            if angle == 0:
                rotated = img_cv
            else:
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(img_cv, M, (w, h), borderValue=(255, 255, 255))

            # Preprocess for better OCR
            gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

            # OCR
            config = f'--psm {self.ocr_psm} -l eng+rus'
            text = pytesseract.image_to_string(binary, config=config)
            all_texts.append((angle, text))

            # Detect vertical text regions specifically
            if angle in (90, 270):
                data.vertical_text += " " + text

        # Find best rotation (most WB indicators)
        best_angle = self._find_best_rotation(all_texts)
        data.rotation_detected = best_angle
        data.raw_text = all_texts[rotations.index(best_angle)][1] if best_angle in rotations else all_texts[0][1]

        # Combine all text for comprehensive extraction
        combined_text = " ".join([t for _, t in all_texts])

        # 3. Extract structured fields from combined text
        data.tracking_number = data.tracking_number or self._extract_tracking(combined_text)
        data.wb_order_id = self._extract_order_id(combined_text)
        data.phone_number = self._extract_phone(combined_text)
        data.delivery_code = self._extract_code(combined_text)
        data.address = self._extract_address(combined_text)

        # 4. Parse QR payloads for additional structured data
        for payload in data.qr_payloads:
            qr_data = self._parse_qr_payload(payload)
            if qr_data.get('tracking') and not data.tracking_number:
                data.tracking_number = qr_data['tracking']
            if qr_data.get('phone') and not data.phone_number:
                data.phone_number = qr_data['phone']
            if qr_data.get('code') and not data.delivery_code:
                data.delivery_code = qr_data['code']

        return data

    def _find_best_rotation(self, all_texts: List[Tuple[int, str]]) -> int:
        """Determine which rotation yields the most WB label indicators."""
        scores = []
        for angle, text in all_texts:
            score = 0
            if self.WB_TRACKING_RE.search(text):
                score += 3
            if self.PHONE_RE.search(text):
                score += 2
            if self.CODE_RE.search(text):
                score += 2
            if self.VERTICAL_WB_RE.search(text):
                score += 1
            if 'wildberries' in text.lower() or 'wb' in text.lower():
                score += 1
            scores.append((angle, score))
        return max(scores, key=lambda x: x[1])[0]

    def _extract_tracking(self, text: str) -> Optional[str]:
        match = self.WB_TRACKING_RE.search(text)
        return match.group() if match else None

    def _extract_order_id(self, text: str) -> Optional[str]:
        # Look for standalone long digit sequences (WB order IDs)
        candidates = self.WB_ORDER_RE.findall(text)
        # Filter: WB order IDs are usually 9-15 digits, prefer longer ones
        if candidates:
            return max(candidates, key=len)
        return None

    def _extract_phone(self, text: str) -> Optional[str]:
        matches = self.PHONE_RE.findall(text)
        return matches[0] if matches else None

    def _extract_code(self, text: str) -> Optional[str]:
        matches = self.CODE_RE.findall(text)
        # Usually the delivery code appears near the phone or in a specific context
        # Return the last 4-digit match as it's often the delivery code
        return matches[-1] if matches else None

    def _extract_address(self, text: str) -> Optional[str]:
        # Look for Russian address patterns
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if any(kw in line.lower() for kw in ['улица', 'ул.', 'г.', 'город', 'проспект', 'переулок']):
                return line
        return None

    def _parse_qr_payload(self, payload: str) -> Dict[str, str]:
        """Parse structured data from QR code payloads.
        WB QR codes often contain JSON or pipe-delimited data."""
        result = {}
        # Try JSON
        try:
            import json
            obj = json.loads(payload)
            result['tracking'] = obj.get('tracking', obj.get('track', ''))
            result['phone'] = str(obj.get('phone', ''))[-7:] if obj.get('phone') else ''
            result['code'] = str(obj.get('code', obj.get('pin', '')))
            result['order_id'] = str(obj.get('order_id', obj.get('orderId', '')))
        except (json.JSONDecodeError, ValueError):
            pass

        # Try pipe-delimited: tracking|phone|code
        if '|' in payload:
            parts = payload.split('|')
            if len(parts) >= 3:
                result.setdefault('tracking', parts[0])
                result.setdefault('phone', parts[1][-7:] if len(parts[1]) >= 7 else parts[1])
                result.setdefault('code', parts[2])

        # Fallback: regex extraction from raw payload
        if not result.get('tracking'):
            m = self.WB_TRACKING_RE.search(payload)
            if m:
                result['tracking'] = m.group()

        return result

    def match_targets(self, targets: List[Dict], label_data: List[WBLabelData]) -> List[WBLabelMatch]:
        """Match target entries against extracted label data.

        targets: list of dicts with keys like 'tracking', 'phone', 'code', 'order_id', 'raw'
        label_data: list of WBLabelData from process_pdf

        Returns: list of WBLabelMatch objects
        """
        matches = []
        used_pages = set()

        for t_idx, target in enumerate(targets):
            best_match = None
            best_confidence = 0.0

            target_tracking = target.get('tracking', '')
            target_phone = target.get('phone', '')
            target_code = target.get('code', '')
            target_order = target.get('order_id', '')
            target_raw = target.get('raw', '')

            for ld in label_data:
                if ld.page_idx in used_pages:
                    continue

                confidence = 0.0
                matched_fields = {}
                match_type = 'none'

                # Criteria 1: Exact tracking number match (highest confidence)
                if target_tracking and ld.tracking_number:
                    if target_tracking.upper() == ld.tracking_number.upper():
                        confidence = 1.0
                        match_type = 'exact'
                        matched_fields['tracking'] = ld.tracking_number

                # Criteria 2: Phone + Code pair match
                if confidence < 0.9 and target_phone and target_code:
                    phone_match = (target_phone == ld.phone_number) or (target_phone in ld.phone_number if ld.phone_number else False)
                    code_match = (target_code == ld.delivery_code) or (target_code in ld.delivery_code if ld.delivery_code else False)
                    if phone_match and code_match:
                        confidence = 0.95
                        match_type = 'phone_code'
                        matched_fields['phone'] = ld.phone_number
                        matched_fields['code'] = ld.delivery_code
                    elif phone_match:
                        confidence = max(confidence, 0.6)
                        match_type = 'phone_only'
                        matched_fields['phone'] = ld.phone_number

                # Criteria 3: Order ID match
                if confidence < 0.9 and target_order and ld.wb_order_id:
                    if target_order == ld.wb_order_id:
                        confidence = 0.9
                        match_type = 'order_id'
                        matched_fields['order_id'] = ld.wb_order_id

                # Criteria 4: QR payload contains target data
                if confidence < 0.8:
                    for payload in ld.qr_payloads:
                        if target_tracking and target_tracking in payload:
                            confidence = 0.85
                            match_type = 'qr_data'
                            matched_fields['qr_tracking'] = target_tracking
                            break
                        if target_phone and target_phone in payload:
                            confidence = max(confidence, 0.7)
                            match_type = 'qr_phone'
                            matched_fields['qr_phone'] = target_phone

                # Criteria 5: Fuzzy raw text match
                if confidence < 0.5 and target_raw:
                    target_digits = set(re.findall(r'\d+', target_raw))
                    label_digits = set(re.findall(r'\d+', ld.raw_text + ld.vertical_text))
                    if target_digits and label_digits:
                        overlap = len(target_digits & label_digits) / len(target_digits)
                        if overlap > 0.5:
                            confidence = overlap * 0.5
                            match_type = 'fuzzy'

                if confidence > best_confidence and confidence >= 0.5:
                    best_confidence = confidence
                    best_match = WBLabelMatch(
                        page_idx=ld.page_idx,
                        target_idx=t_idx,
                        target_raw=target_raw,
                        match_type=match_type,
                        confidence=confidence,
                        matched_fields=matched_fields,
                        debug_info=ld.to_dict()
                    )

            if best_match:
                used_pages.add(best_match.page_idx)
                matches.append(best_match)

        return matches

    def generate_debug_overlay(self, img: Image.Image, data: WBLabelData) -> Image.Image:
        """Generate a debug overlay image showing detected regions and data."""
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]
        overlay = img_cv.copy()

        # Draw border
        cv2.rectangle(overlay, (5, 5), (w-5, h-5), (0, 255, 136), 3)

        # Info panel
        panel_h = 140
        cv2.rectangle(overlay, (10, 10), (w-10, 10+panel_h), (5, 10, 25), -1)
        cv2.rectangle(overlay, (10, 10), (w-10, 10+panel_h), (100, 255, 218), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 35
        lines = [
            f"Page: {data.page_idx + 1} | Rotation: {data.rotation_detected}°",
            f"Tracking: {data.tracking_number or 'NOT FOUND'}",
            f"Phone: {data.phone_number or 'N/A'} | Code: {data.delivery_code or 'N/A'}",
            f"Order ID: {data.wb_order_id or 'N/A'} | QRs: {len(data.qr_payloads)} | Barcodes: {len(data.barcodes)}",
        ]
        for line in lines:
            cv2.putText(overlay, line, (20, y_offset), font, 0.55, (100, 255, 218), 2)
            y_offset += 25

        # Highlight barcode/QR regions if we had their coordinates
        # (pyzbar gives rect, but we'd need to store it - simplified here)

        return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))


def parse_target_list(text: str) -> List[Dict]:
    """Parse a pasted target list into structured entries.

    Supports multiple formats:
      • Tracking numbers: WBAERUGBACE0900JRM
      • Phone + Code: 5261288 1844
      • Order IDs: 123456789
      • Mixed: any combination separated by whitespace
    """
    entries = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        entry = {"raw": line}
        digits = re.findall(r'\d+', line)

        # Look for WB tracking number
        wb_match = re.search(r'WB[A-Z0-9]{10,20}', line, re.IGNORECASE)
        if wb_match:
            entry['tracking'] = wb_match.group().upper()

        # Look for phone (7 digits) and code (4 digits)
        phone = next((d for d in digits if len(d) == 7), None)
        code = next((d for d in digits if len(d) == 4), None)
        if phone:
            entry['phone'] = phone
        if code:
            entry['code'] = code

        # Long digit sequence might be order ID
        long_digits = [d for d in digits if 9 <= len(d) <= 15]
        if long_digits and 'tracking' not in entry:
            entry['order_id'] = long_digits[0]

        entries.append(entry)

    return entries
