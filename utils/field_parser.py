"""
Field Parser Module
Advanced rule-based field extraction with visual + textual understanding

Handles:
- English, Hindi, Gujarati text
- Multiple invoice formats and layouts
- Key-value pair detection with spatial awareness
- Table structure interpretation
- Contextual positioning analysis

Optimized for IDFC Bank's tractor loan quotation processing.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict


class FieldParser:
    """
    Extract structured fields using hybrid visual-textual analysis.
    
    Advanced Extraction Strategy:
    1. Spatial key-value pair detection
    2. Table structure recognition
    3. Region-based analysis (header/body/footer)
    4. Pattern matching with multilingual support
    5. Contextual keyword proximity
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
    
    # Key-value keywords for spatial detection
    KEY_PATTERNS = {
        'horse_power': [
            'hp', 'h.p.', 'horse power', 'horsepower', 'bhp', 'power',
            'अश्वशक्ति', 'एचपी', 'हॉर्स पावर', 'પાવર', 'અશ્વશક્તિ'
        ],
        'asset_cost': [
            'total', 'grand total', 'amount', 'price', 'cost', 'value',
            'net amount', 'payable', 'ex-showroom', 'on-road', 'final',
            'कुल', 'राशि', 'मूल्य', 'कीमत', 'કુલ', 'કિંમત', 'રકમ'
        ],
        'model_name': [
            'model', 'tractor', 'vehicle', 'product', 'item', 'description',
            'मॉडल', 'ट्रैक्टर', 'વાહન', 'મોડલ', 'ટ્રેક્ટર'
        ],
        'dealer_name': [
            'dealer', 'seller', 'from', 'sold by', 'authorized', 'agent',
            'डीलर', 'विक्रेता', 'एजेंट', 'ડીલર', 'વિક્રેતા'
        ]
    }
    
    def __init__(self):
        # Compile patterns for efficiency
        self._hp_compiled = [re.compile(p, re.IGNORECASE) for p in self.HP_PATTERNS]
        self._cost_compiled = [re.compile(p, re.IGNORECASE) for p in self.COST_PATTERNS]
        self._model_compiled = [re.compile(p, re.IGNORECASE) for p in self.MODEL_PATTERNS]
    
    def extract_all(self, ocr_result: Dict) -> Dict:
        """
        Extract all fields using hybrid visual-textual analysis.
        
        Enhanced extraction with:
        1. Spatial key-value pair detection
        2. Table structure analysis
        3. Region-based contextual extraction
        4. Pattern matching fallback
        
        Args:
            ocr_result: Dict with 'texts', 'boxes', 'full_text' from OCR
            
        Returns:
            Dict with (value, confidence) tuples for each field
        """
        full_text = ocr_result.get('full_text', '')
        texts = ocr_result.get('texts', [])
        boxes = ocr_result.get('boxes', [])
        
        # Build spatial index for key-value detection
        text_elements = list(zip(texts, boxes)) if texts and boxes else []
        
        # Detect key-value pairs spatially
        kv_pairs = self._detect_key_value_pairs(text_elements)
        
        # Detect table structures
        table_data = self._detect_table_structure(text_elements)
        
        # Extract using multiple strategies
        results = {}
        
        # === DEALER NAME ===
        results['dealer_name'] = self._extract_with_fallback(
            'dealer_name',
            [
                lambda: self._extract_from_kv(kv_pairs, 'dealer_name'),
                lambda: self.extract_dealer(full_text, texts, boxes)
            ]
        )
        
        # === MODEL NAME ===
        results['model_name'] = self._extract_with_fallback(
            'model_name',
            [
                lambda: self._extract_model_from_table(table_data),
                lambda: self._extract_from_kv(kv_pairs, 'model_name'),
                lambda: self.extract_model(full_text, texts)
            ]
        )
        
        # === HORSE POWER ===
        results['horse_power'] = self._extract_with_fallback(
            'horse_power',
            [
                lambda: self._extract_hp_from_table(table_data),
                lambda: self._extract_from_kv(kv_pairs, 'horse_power'),
                lambda: self.extract_hp(full_text)
            ]
        )
        
        # === ASSET COST ===
        results['asset_cost'] = self._extract_with_fallback(
            'asset_cost',
            [
                lambda: self._extract_cost_from_table(table_data),
                lambda: self._extract_from_kv(kv_pairs, 'asset_cost'),
                lambda: self.extract_cost(full_text)
            ]
        )
        
        return results
    
    def _extract_with_fallback(
        self,
        field: str,
        strategies: List[callable]
    ) -> Tuple[Any, float]:
        """Try multiple extraction strategies, return best result."""
        best_result = (None, 0.0)
        
        for strategy in strategies:
            try:
                result = strategy()
                if result and result[0] is not None and result[1] > best_result[1]:
                    best_result = result
                    if result[1] >= 0.9:  # High confidence, stop searching
                        break
            except Exception:
                continue
        
        return best_result
    
    def _detect_key_value_pairs(
        self,
        text_elements: List[Tuple[str, List[int]]]
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Detect key-value pairs using spatial proximity analysis.
        
        Strategy:
        - Find text elements containing key words
        - Look for values to the right or below the key
        - Use spatial distance to determine association
        """
        kv_pairs = defaultdict(list)
        
        if not text_elements:
            return kv_pairs
        
        for i, (text, box) in enumerate(text_elements):
            text_lower = text.lower().strip()
            
            # Check if this text contains a key
            for field, keywords in self.KEY_PATTERNS.items():
                if any(kw in text_lower for kw in keywords):
                    # Found a key, look for value
                    value, conf = self._find_value_near_key(
                        text, box, text_elements, field
                    )
                    if value:
                        kv_pairs[field].append((value, conf))
        
        return kv_pairs
    
    def _find_value_near_key(
        self,
        key_text: str,
        key_box: List[int],
        text_elements: List[Tuple[str, List[int]]],
        field: str
    ) -> Tuple[Optional[str], float]:
        """
        Find value associated with a key based on spatial position.
        
        Looks for values:
        1. On the same line, to the right of the key
        2. On the next line, below the key
        3. After colon/separator in the same text
        """
        # Check if value is in the same text (after colon)
        if ':' in key_text:
            parts = key_text.split(':', 1)
            if len(parts) > 1 and parts[1].strip():
                return (parts[1].strip(), 0.9)
        
        key_x1, key_y1, key_x2, key_y2 = key_box
        key_cx = (key_x1 + key_x2) / 2
        key_cy = (key_y1 + key_y2) / 2
        key_height = key_y2 - key_y1
        
        candidates = []
        
        for text, box in text_elements:
            if text.lower().strip() == key_text.lower().strip():
                continue
            
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            # Check if on same line (y overlap) and to the right
            if abs(cy - key_cy) < key_height * 0.8 and x1 > key_x2 - 10:
                distance = x1 - key_x2
                if distance < 300:  # Max horizontal distance
                    candidates.append((text, 0.85, distance))
            
            # Check if on next line and aligned
            elif y1 > key_y2 and y1 - key_y2 < key_height * 2:
                if abs(cx - key_cx) < 100:  # Aligned below
                    candidates.append((text, 0.75, y1 - key_y2))
        
        # Return closest candidate
        if candidates:
            candidates.sort(key=lambda x: x[2])  # Sort by distance
            best = candidates[0]
            return (best[0], best[1])
        
        return (None, 0.0)
    
    def _extract_from_kv(
        self,
        kv_pairs: Dict,
        field: str
    ) -> Tuple[Any, float]:
        """Extract field from detected key-value pairs."""
        values = kv_pairs.get(field, [])
        if not values:
            return (None, 0.0)
        
        # Get highest confidence value
        best = max(values, key=lambda x: x[1])
        raw_value, conf = best
        
        # Post-process based on field type
        if field == 'horse_power':
            hp = self._extract_hp_from_text(raw_value)
            if hp:
                return (hp, conf)
        elif field == 'asset_cost':
            cost = self._parse_indian_number(raw_value)
            if cost and 50000 <= cost <= 5000000:
                return (cost, conf)
        else:
            return (raw_value.strip(), conf)
        
        return (None, 0.0)
    
    def _extract_hp_from_text(self, text: str) -> Optional[int]:
        """Extract HP number from text."""
        numbers = re.findall(r'\d+', text)
        for num in numbers:
            hp = int(num)
            if 15 <= hp <= 150:
                return hp
        return None
    
    def _detect_table_structure(
        self,
        text_elements: List[Tuple[str, List[int]]]
    ) -> Dict:
        """
        Detect table structures in document.
        
        Strategy:
        - Group text elements by rows (similar y-coordinates)
        - Identify column alignment
        - Extract header-value relationships
        """
        if not text_elements or len(text_elements) < 3:
            return {'rows': [], 'columns': [], 'cells': {}}
        
        # Group by rows
        rows = self._group_by_rows(text_elements)
        
        # Identify potential table regions
        table_data = {
            'rows': rows,
            'header_row': None,
            'data_rows': [],
            'cells': {}
        }
        
        # Find header row (often contains keywords)
        table_keywords = ['model', 'hp', 'power', 'price', 'amount', 'description',
                         'मॉडल', 'मूल्य', 'मॉडেલ', 'કિંમત']
        
        for i, row in enumerate(rows):
            row_text = ' '.join([t for t, _ in row]).lower()
            if any(kw in row_text for kw in table_keywords):
                table_data['header_row'] = i
                table_data['data_rows'] = rows[i+1:min(i+6, len(rows))]
                break
        
        return table_data
    
    def _group_by_rows(
        self,
        text_elements: List[Tuple[str, List[int]]],
        y_tolerance: int = 15
    ) -> List[List[Tuple[str, List[int]]]]:
        """Group text elements into rows based on y-coordinate."""
        if not text_elements:
            return []
        
        # Sort by y then x
        sorted_elements = sorted(text_elements, key=lambda x: (x[1][1], x[1][0]))
        
        rows = []
        current_row = [sorted_elements[0]]
        current_y = sorted_elements[0][1][1]
        
        for text, box in sorted_elements[1:]:
            y = box[1]
            if abs(y - current_y) <= y_tolerance:
                current_row.append((text, box))
            else:
                rows.append(sorted(current_row, key=lambda x: x[1][0]))
                current_row = [(text, box)]
                current_y = y
        
        if current_row:
            rows.append(sorted(current_row, key=lambda x: x[1][0]))
        
        return rows
    
    def _extract_model_from_table(self, table_data: Dict) -> Tuple[Optional[str], float]:
        """Extract model name from table structure."""
        for row in table_data.get('data_rows', []):
            row_text = ' '.join([t for t, _ in row])
            
            # Try model patterns on row text
            for pattern in self._model_compiled:
                matches = pattern.findall(row_text)
                if matches:
                    model = re.sub(r'\s+', ' ', matches[0]).strip()
                    return (self._normalize_model(model), 0.92)
        
        return (None, 0.0)
    
    def _extract_hp_from_table(self, table_data: Dict) -> Tuple[Optional[int], float]:
        """Extract HP from table structure."""
        for row in table_data.get('data_rows', []):
            row_text = ' '.join([t for t, _ in row])
            
            # Look for HP patterns
            hp_match = re.search(r'(\d{2,3})\s*(?:HP|hp|H\.P\.)', row_text)
            if hp_match:
                hp = int(hp_match.group(1))
                if 15 <= hp <= 150:
                    return (hp, 0.92)
        
        return (None, 0.0)
    
    def _extract_cost_from_table(self, table_data: Dict) -> Tuple[Optional[int], float]:
        """Extract cost from table, looking for totals in last rows."""
        # Check data rows in reverse (totals usually at bottom)
        for row in reversed(table_data.get('data_rows', [])):
            row_text = ' '.join([t for t, _ in row])
            
            # Look for total patterns
            if any(kw in row_text.lower() for kw in ['total', 'grand', 'कुल', 'કુલ']):
                numbers = re.findall(r'[\d,\.]+', row_text)
                for num in reversed(numbers):  # Usually last number
                    cost = self._parse_indian_number(num)
                    if cost and 50000 <= cost <= 5000000:
                        return (cost, 0.92)
        
        return (None, 0.0)
    
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
