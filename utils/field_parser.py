"""
Field Parser Module
Rule-based field extraction from OCR text with multilingual support

Handles:
- English, Hindi, Gujarati text
- Multiple invoice formats
- Various date/number formats
"""

import re
from typing import Dict, List, Optional, Tuple


class FieldParser:
    """
    Extract structured fields from OCR text using patterns and heuristics.
    
    Extraction Strategy:
    1. Pattern matching with regex
    2. Keyword proximity search
    3. Layout-based heuristics
    """
    
    # ============ HORSE POWER PATTERNS ============
    HP_PATTERNS = [
        # English patterns
        r'(\d{2,3})\s*(?:HP|hp|H\.P\.|Hp)',
        r'(\d{2,3})\s*(?:BHP|bhp|B\.H\.P\.)',
        r'(?:Horse\s*Power|HP|Power)[:\s]*(\d{2,3})',
        r'(?:Engine|Motor)[:\s]*(\d{2,3})\s*(?:HP|hp)',
        # Hindi patterns
        r'(\d{2,3})\s*(?:अश्वशक्ति|एचपी|हॉर्स\s*पावर)',
        r'(?:अश्वशक्ति|पावर)[:\s]*(\d{2,3})',
        # Gujarati patterns  
        r'(\d{2,3})\s*(?:અશ્વશક્તિ|એચપી)',
        # Generic: number near HP keyword
        r'HP[:\s\-]*(\d{2,3})',
        r'(\d{2,3})[:\s\-]*HP',
    ]
    
    # ============ ASSET COST PATTERNS ============
    COST_PATTERNS = [
        # Labeled totals (most reliable)
        r'(?:Total|Grand\s*Total|Net\s*Amount|Final|Payable)[:\s]*[₹Rs\.INR\s]*([\d,\.]+)',
        r'(?:कुल|टोटल|राशि|मूल्य)[:\s]*[₹Rs\.INR\s]*([\d,\.]+)',
        r'(?:કુલ|કિંમત)[:\s]*[₹Rs\.INR\s]*([\d,\.]+)',
        # Currency symbol patterns
        r'[₹]\s*([\d,]+)(?:\s*/-)?',
        r'Rs\.?\s*([\d,]+)(?:\s*/-)?',
        r'INR\s*([\d,]+)',
        # Price/Amount labels
        r'(?:Price|Amount|Cost|Value)[:\s]*[₹Rs\.INR\s]*([\d,]+)',
        r'(?:Ex[\-\s]*Showroom|On[\-\s]*Road)[:\s]*[₹Rs\.INR\s]*([\d,]+)',
    ]
    
    # ============ MODEL NAME PATTERNS ============
    MODEL_PATTERNS = [
        # Mahindra variants
        r'(Mahindra\s+\d{3,4}\s*(?:DI|XP|XT|Plus|Power)?(?:\s*Plus)?)',
        r'(MAHINDRA\s+\d{3,4}\s*(?:DI|XP|XT|Plus|Power)?)',
        r'(Arjun\s+(?:Novo\s+)?\d{3,4})',
        # John Deere
        r'(John\s*Deere\s+\d{4}[A-Z]?)',
        r'(JD[\s\-]*\d{4}[A-Z]?)',
        # TAFE / Massey Ferguson
        r'(TAFE\s+\d{4})',
        r'(Massey\s*Ferguson\s+\d{3,4})',
        r'(MF[\s\-]*\d{3,4})',
        # Swaraj
        r'(Swaraj\s+\d{3,4}(?:\s*FE|\s*XT)?)',
        # Sonalika
        r'(Sonalika\s+(?:DI\s*)?\d+(?:\s*[A-Z]+)?)',
        r'(Sonalika\s+[A-Z]+\s*\d+)',
        # New Holland
        r'(New\s*Holland\s+\d{4})',
        r'(NH[\s\-]*\d{4})',
        # Kubota
        r'(Kubota\s+\w+[\s\-]?\d+)',
        # Eicher
        r'(Eicher\s+\d{3,4})',
        # Escorts/Farmtrac/Powertrac
        r'(Farmtrac\s+\d+)',
        r'(Powertrac\s+\d+)',
        r'(Escorts\s+\d+)',
        # Indo Farm
        r'(Indo\s*Farm\s+\d+)',
        # Generic patterns
        r'(?:Model|Tractor)[:\s]+([A-Za-z]+[\s\-]*\d{3,4}[A-Za-z]*)',
    ]
    
    # ============ DEALER PATTERNS ============
    DEALER_PATTERNS = [
        # Explicit labels
        r'(?:Dealer|Seller|From|Sold\s*by|Authorized)[:\s]+([^\n\r]+)',
        r'(?:विक्रेता|डीलर|एजेंसी)[:\s]+([^\n\r]+)',
        r'(?:વિક્રેતા|ડીલર)[:\s]+([^\n\r]+)',
    ]
    
    # Business suffixes for dealer detection
    BUSINESS_SUFFIXES = [
        'pvt', 'ltd', 'private', 'limited', 'llp',
        'tractors', 'motors', 'auto', 'automobiles', 'agencies',
        'enterprises', 'trading', 'corporation', 'associates',
        'ट्रैक्टर्स', 'मोटर्स', 'एजेंसी',  # Hindi
        'ટ્રેક્ટર્સ', 'મોટર્સ',  # Gujarati
    ]
    
    def __init__(self):
        # Compile patterns for efficiency
        self._hp_compiled = [re.compile(p, re.IGNORECASE) for p in self.HP_PATTERNS]
        self._cost_compiled = [re.compile(p, re.IGNORECASE) for p in self.COST_PATTERNS]
        self._model_compiled = [re.compile(p, re.IGNORECASE) for p in self.MODEL_PATTERNS]
    
    def extract_all(self, ocr_result: Dict) -> Dict:
        """
        Extract all fields from OCR result.
        
        Args:
            ocr_result: Dict with 'texts', 'boxes', 'full_text' from OCR
            
        Returns:
            Dict with (value, confidence) tuples for each field
        """
        full_text = ocr_result.get('full_text', '')
        texts = ocr_result.get('texts', [])
        boxes = ocr_result.get('boxes', [])
        
        return {
            'dealer_name': self.extract_dealer(full_text, texts, boxes),
            'model_name': self.extract_model(full_text, texts),
            'horse_power': self.extract_hp(full_text),
            'asset_cost': self.extract_cost(full_text)
        }
    
    def extract_hp(self, text: str) -> Tuple[Optional[int], float]:
        """
        Extract horse power value.
        
        Returns: (value, confidence) or (None, 0.0)
        """
        # Try each pattern
        for pattern in self._hp_compiled:
            matches = pattern.findall(text)
            for match in matches:
                try:
                    hp = int(match)
                    if 15 <= hp <= 150:  # Valid HP range for tractors
                        return (hp, 0.9)
                except (ValueError, TypeError):
                    continue
        
        # Fallback: find numbers near HP keywords
        hp_context = re.search(
            r'(\d{2,3})\s*.{0,15}(?:HP|hp|अश्वशक्ति|અશ્વશક્તિ)|(?:HP|hp|अश्वशक्ति|અશ્વશક્તિ).{0,15}(\d{2,3})',
            text, re.IGNORECASE
        )
        if hp_context:
            try:
                hp = int(hp_context.group(1) or hp_context.group(2))
                if 15 <= hp <= 150:
                    return (hp, 0.7)
            except (ValueError, TypeError):
                pass
        
        return (None, 0.0)
    
    def extract_cost(self, text: str) -> Tuple[Optional[int], float]:
        """
        Extract asset cost/price.
        
        Returns: (value, confidence) or (None, 0.0)
        """
        # Try labeled patterns first (most reliable)
        for pattern in self._cost_compiled[:4]:  # First 4 are labeled patterns
            matches = pattern.findall(text)
            for match in matches:
                cost = self._parse_indian_number(match)
                if cost and 50000 <= cost <= 5000000:
                    return (cost, 0.9)
        
        # Try currency patterns
        for pattern in self._cost_compiled[4:]:
            matches = pattern.findall(text)
            for match in matches:
                cost = self._parse_indian_number(match)
                if cost and 50000 <= cost <= 5000000:
                    return (cost, 0.8)
        
        # Fallback: find large numbers in Indian format
        large_nums = re.findall(r'(\d{1,2}[,\.]?\d{2}[,\.]?\d{3})', text)
        valid_costs = []
        
        for num in large_nums:
            cost = self._parse_indian_number(num)
            if cost and 50000 <= cost <= 5000000:
                valid_costs.append(cost)
        
        if valid_costs:
            # Take the largest as it's likely the total
            return (max(valid_costs), 0.6)
        
        return (None, 0.0)
    
    def _parse_indian_number(self, num_str: str) -> Optional[int]:
        """Parse Indian number format (e.g., 5,25,000)."""
        if not num_str:
            return None
        try:
            # Remove commas, dots (thousands separator), spaces
            cleaned = re.sub(r'[,\.\s]', '', str(num_str))
            return int(cleaned)
        except ValueError:
            return None
    
    def extract_model(self, text: str, texts: List[str] = None) -> Tuple[Optional[str], float]:
        """
        Extract tractor model name.
        
        Returns: (value, confidence) or (None, 0.0)
        """
        # Try brand-specific patterns
        for pattern in self._model_compiled:
            matches = pattern.findall(text)
            if matches:
                model = re.sub(r'\s+', ' ', matches[0]).strip()
                # Normalize common variations
                model = self._normalize_model(model)
                return (model, 0.9)
        
        # Try to find model in individual text lines
        if texts:
            for line in texts:
                for pattern in self._model_compiled:
                    matches = pattern.findall(line)
                    if matches:
                        model = re.sub(r'\s+', ' ', matches[0]).strip()
                        return (self._normalize_model(model), 0.85)
        
        return (None, 0.0)
    
    def _normalize_model(self, model: str) -> str:
        """Normalize model name for consistency."""
        if not model:
            return model
        
        # Capitalize brand names properly
        model = re.sub(r'\bmahindra\b', 'Mahindra', model, flags=re.IGNORECASE)
        model = re.sub(r'\bjohn\s*deere\b', 'John Deere', model, flags=re.IGNORECASE)
        model = re.sub(r'\btafe\b', 'TAFE', model, flags=re.IGNORECASE)
        model = re.sub(r'\bswaraj\b', 'Swaraj', model, flags=re.IGNORECASE)
        model = re.sub(r'\bsonalika\b', 'Sonalika', model, flags=re.IGNORECASE)
        model = re.sub(r'\bmassey\s*ferguson\b', 'Massey Ferguson', model, flags=re.IGNORECASE)
        model = re.sub(r'\bnew\s*holland\b', 'New Holland', model, flags=re.IGNORECASE)
        
        return model.strip()
    
    def extract_dealer(
        self,
        text: str,
        texts: List[str],
        boxes: List[List[int]] = None
    ) -> Tuple[Optional[str], float]:
        """
        Extract dealer/seller name.
        
        Strategy:
        1. Look for labeled dealer name
        2. Find business names in header region
        3. Pattern match company names
        
        Returns: (value, confidence) or (None, 0.0)
        """
        # Strategy 1: Explicit labels
        for pattern in self.DEALER_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                dealer = match.group(1).strip()
                dealer = re.sub(r'[:\-\|]+$', '', dealer).strip()
                if len(dealer) > 3 and not dealer.isdigit():
                    return (dealer, 0.95)
        
        # Strategy 2: Check header lines (first 5-7 lines) for business names
        for line in texts[:7]:
            line_lower = line.lower()
            
            # Check for business suffixes
            if any(suffix in line_lower for suffix in self.BUSINESS_SUFFIXES):
                dealer = self._clean_dealer_name(line)
                if dealer and len(dealer) > 5:
                    return (dealer, 0.85)
        
        # Strategy 3: Pattern match company names anywhere
        company_patterns = [
            r'([A-Z][A-Za-z\s&]+(?:Tractors?|Motors?|Auto|Agencies|Enterprises)[A-Za-z\s&]*(?:Pvt\.?|Private)?\.?\s*(?:Ltd\.?|Limited)?)',
            r'([A-Z][A-Za-z\s]+(?:Pvt\.?|Private)\s*(?:Ltd\.?|Limited))',
        ]
        
        for pattern in company_patterns:
            match = re.search(pattern, text)
            if match:
                dealer = self._clean_dealer_name(match.group(1))
                if dealer and len(dealer) > 5:
                    return (dealer, 0.75)
        
        # Strategy 4: First line often contains dealer name (letterhead)
        if texts and len(texts[0]) > 5:
            first_line = texts[0].strip()
            if not first_line.isdigit() and not re.match(r'^(Date|Invoice|Quotation)', first_line, re.I):
                return (first_line, 0.6)
        
        return (None, 0.0)
    
    def _clean_dealer_name(self, name: str) -> str:
        """Clean and normalize dealer name."""
        if not name:
            return name
        
        # Remove leading/trailing punctuation
        name = re.sub(r'^[:\-\|\s]+|[:\-\|\s]+$', '', name)
        
        # Remove phone numbers, emails
        name = re.sub(r'\b\d{10,}\b', '', name)
        name = re.sub(r'\S+@\S+', '', name)
        
        # Normalize whitespace
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name
    
    def get_text_regions(
        self,
        texts: List[str],
        boxes: List[List[int]],
        image_height: int
    ) -> Dict[str, List[str]]:
        """
        Divide text by document region.
        
        Returns dict with 'header', 'body', 'footer' text lists.
        """
        header_texts = []
        body_texts = []
        footer_texts = []
        
        header_cutoff = image_height * 0.2
        footer_cutoff = image_height * 0.75
        
        for text, box in zip(texts, boxes):
            y_center = (box[1] + box[3]) / 2
            
            if y_center < header_cutoff:
                header_texts.append(text)
            elif y_center > footer_cutoff:
                footer_texts.append(text)
            else:
                body_texts.append(text)
        
        return {
            'header': header_texts,
            'body': body_texts,
            'footer': footer_texts
        }
