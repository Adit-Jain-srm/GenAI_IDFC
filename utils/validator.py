"""
Validator Module
Field validation, fuzzy matching, and cross-validation

Features:
- Fuzzy matching against master lists
- HP-Model cross-validation
- Range and sanity checks
"""

from typing import Dict, List, Optional, Tuple

try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


class Validator:
    """
    Validate and normalize extracted fields.
    """
    
    # Comprehensive Model-HP mapping for cross-validation
    MODEL_HP_MAP = {
        # Mahindra
        "MAHINDRA 265": 30, "MAHINDRA 275": 39, "MAHINDRA 475": 42,
        "MAHINDRA 575": 50, "MAHINDRA 585": 50, "MAHINDRA 605": 57,
        "MAHINDRA 415": 40, "MAHINDRA 555": 55, "MAHINDRA 595": 60,
        "ARJUN 555": 55, "ARJUN 605": 60, "ARJUN NOVO": 57,
        # John Deere
        "JOHN DEERE 5042": 42, "JOHN DEERE 5050": 50, "JOHN DEERE 5055": 55,
        "JOHN DEERE 5210": 42, "JOHN DEERE 5310": 55, "JOHN DEERE 5405": 63,
        # TAFE
        "TAFE 5900": 59, "TAFE 7250": 72, "TAFE 7515": 75,
        "TAFE 35": 35, "TAFE 45": 45,
        # Swaraj
        "SWARAJ 717": 17, "SWARAJ 724": 26, "SWARAJ 735": 39,
        "SWARAJ 744": 48, "SWARAJ 855": 52, "SWARAJ 963": 65,
        # Sonalika
        "SONALIKA 35": 35, "SONALIKA 42": 42, "SONALIKA 50": 50,
        "SONALIKA 60": 60, "SONALIKA 750": 50, "SONALIKA DI": 50,
        # Massey Ferguson
        "MASSEY FERGUSON 241": 42, "MASSEY FERGUSON 1035": 39,
        "MASSEY FERGUSON 7250": 50, "MASSEY FERGUSON 9500": 60,
        # New Holland
        "NEW HOLLAND 3230": 42, "NEW HOLLAND 3630": 55, "NEW HOLLAND 4710": 47,
        # Kubota
        "KUBOTA MU4501": 45, "KUBOTA MU5501": 55,
        # Eicher
        "EICHER 241": 24, "EICHER 312": 31, "EICHER 380": 38,
        "EICHER 485": 48, "EICHER 557": 55,
        # Escorts/Farmtrac/Powertrac
        "FARMTRAC 45": 45, "FARMTRAC 60": 60,
        "POWERTRAC 425": 37, "POWERTRAC 434": 39, "POWERTRAC 439": 39,
    }
    
    # Expanded default dealer list
    DEFAULT_DEALERS = [
        # Generic patterns
        "Tractors Pvt Ltd", "Motors Private Limited", "Auto Agencies",
        "Automobile Dealers", "Tractors and Automobiles",
        # Common name patterns
        "Sharma Tractors", "Gupta Motors", "Singh Auto", "Patel Agencies",
        "Kumar Tractors", "Reddy Auto Works", "Joshi Motors",
        "Verma Automobiles", "Agarwal Tractors", "Choudhary Motors",
        # Brand authorized dealers
        "Mahindra Authorized Dealer", "John Deere Dealership",
        "Swaraj Tractors", "TAFE Motors", "Sonalika Dealer",
    ]
    
    # Expanded model list
    DEFAULT_MODELS = [
        # Mahindra
        "Mahindra 265 DI", "Mahindra 275 DI", "Mahindra 275 XP Plus",
        "Mahindra 415 DI", "Mahindra 475 DI", "Mahindra 555 DI",
        "Mahindra 575 DI", "Mahindra 585 DI", "Mahindra 595 DI",
        "Mahindra 605 DI", "Mahindra Arjun 555 DI", "Mahindra Arjun 605 DI",
        "Mahindra Arjun Novo 605", "Mahindra JIVO",
        # John Deere
        "John Deere 5042D", "John Deere 5050D", "John Deere 5055E",
        "John Deere 5210", "John Deere 5310", "John Deere 5405",
        # TAFE
        "TAFE 35 DI", "TAFE 45 DI", "TAFE 5900", "TAFE 7250", "TAFE 7515",
        # Swaraj
        "Swaraj 717", "Swaraj 724 FE", "Swaraj 735 FE", "Swaraj 735 XT",
        "Swaraj 744 FE", "Swaraj 744 XT", "Swaraj 855 FE", "Swaraj 963 FE",
        # Sonalika
        "Sonalika DI 35", "Sonalika DI 42", "Sonalika DI 50", "Sonalika DI 60",
        "Sonalika DI 750 III", "Sonalika GT 26", "Sonalika RX",
        # Massey Ferguson
        "Massey Ferguson 241 DI", "Massey Ferguson 1035 DI",
        "Massey Ferguson 7250", "Massey Ferguson 9500",
        # New Holland
        "New Holland 3032", "New Holland 3230", "New Holland 3630",
        "New Holland 4710", "New Holland 5500",
        # Kubota
        "Kubota MU4501", "Kubota MU5501", "Kubota L4508",
        # Eicher
        "Eicher 241", "Eicher 312", "Eicher 380", "Eicher 485", "Eicher 557",
        # Escorts
        "Farmtrac 45", "Farmtrac 60", "Powertrac 425", "Powertrac 434", "Powertrac 439",
    ]
    
    def __init__(
        self,
        dealer_master: Optional[str] = None,
        model_master: Optional[str] = None,
        fuzzy_threshold: int = 90
    ):
        """
        Initialize validator.
        
        Args:
            dealer_master: Path to dealer CSV (optional)
            model_master: Path to model CSV (optional)
            fuzzy_threshold: Minimum fuzzy match score (0-100)
        """
        self.fuzzy_threshold = fuzzy_threshold
        
        # Load or use defaults
        self.dealers = self._load_csv(dealer_master) if dealer_master else self.DEFAULT_DEALERS
        self.models = self._load_csv(model_master) if model_master else self.DEFAULT_MODELS
    
    def _load_csv(self, path: str) -> List[str]:
        """Load list from CSV first column."""
        try:
            import csv
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                return [row[0] for row in reader if row and row[0].strip()]
        except Exception:
            return []
    
    def validate_all(self, fields: Dict) -> Dict:
        """
        Validate all extracted fields.
        
        Args:
            fields: Dict with (value, confidence) tuples
            
        Returns:
            Dict with validated values and '_confidence' score
        """
        validated = {}
        confidences = []
        
        # === Dealer Name ===
        dealer_val, dealer_conf = fields.get('dealer_name', (None, 0))
        if dealer_val:
            matched, score = self.fuzzy_match(dealer_val, self.dealers, threshold=85)
            if matched and score >= self.fuzzy_threshold:
                validated['dealer_name'] = matched
                confidences.append(score / 100)
            else:
                # Keep original but lower confidence if no match
                validated['dealer_name'] = dealer_val
                confidences.append(min(dealer_conf, 0.7))
        else:
            validated['dealer_name'] = None
            confidences.append(0)
        
        # === Model Name ===
        model_val, model_conf = fields.get('model_name', (None, 0))
        if model_val:
            matched, score = self.fuzzy_match(model_val, self.models, threshold=80)
            if matched:
                validated['model_name'] = matched
                confidences.append(score / 100)
            else:
                validated['model_name'] = model_val
                confidences.append(min(model_conf, 0.75))
        else:
            validated['model_name'] = None
            confidences.append(0)
        
        # === Horse Power ===
        hp_val, hp_conf = fields.get('horse_power', (None, 0))
        if hp_val is not None and isinstance(hp_val, (int, float)):
            hp_val = int(hp_val)
            if 15 <= hp_val <= 150:
                validated['horse_power'] = hp_val
                
                # Cross-validate with model
                expected_hp = self._get_expected_hp(validated.get('model_name', ''))
                if expected_hp:
                    hp_diff = abs(hp_val - expected_hp)
                    if hp_diff <= 3:
                        confidences.append(min(1.0, hp_conf + 0.1))  # Boost
                    elif hp_diff <= 10:
                        confidences.append(hp_conf)
                    else:
                        confidences.append(hp_conf * 0.7)  # Penalty
                else:
                    confidences.append(hp_conf)
            else:
                validated['horse_power'] = None
                confidences.append(0)
        else:
            validated['horse_power'] = None
            confidences.append(0)
        
        # === Asset Cost ===
        cost_val, cost_conf = fields.get('asset_cost', (None, 0))
        if cost_val is not None:
            try:
                cost_val = int(cost_val)
                if 50000 <= cost_val <= 5000000:
                    validated['asset_cost'] = cost_val
                    confidences.append(cost_conf)
                else:
                    validated['asset_cost'] = None
                    confidences.append(0)
            except (ValueError, TypeError):
                validated['asset_cost'] = None
                confidences.append(0)
        else:
            validated['asset_cost'] = None
            confidences.append(0)
        
        # Calculate overall confidence
        validated['_confidence'] = sum(confidences) / len(confidences) if confidences else 0
        
        return validated
    
    def fuzzy_match(
        self,
        query: str,
        choices: List[str],
        threshold: int = None
    ) -> Tuple[Optional[str], int]:
        """
        Fuzzy match query against choices.
        
        Returns: (best_match, score) or (None, 0)
        """
        threshold = threshold if threshold is not None else self.fuzzy_threshold
        
        if not query or not choices:
            return (None, 0)
        
        query = str(query).strip()
        
        if RAPIDFUZZ_AVAILABLE:
            # Try different scorers
            result = process.extractOne(
                query, choices,
                scorer=fuzz.WRatio,  # Weighted ratio handles partial matches better
                score_cutoff=threshold
            )
            if result:
                return (result[0], int(result[1]))
        else:
            # Fallback: simple substring matching
            query_lower = query.lower()
            best_match = None
            best_score = 0
            
            for choice in choices:
                choice_lower = choice.lower()
                
                # Exact match
                if query_lower == choice_lower:
                    return (choice, 100)
                
                # Substring match
                if query_lower in choice_lower or choice_lower in query_lower:
                    score = 85
                    if score > best_score:
                        best_score = score
                        best_match = choice
                
                # Word overlap
                query_words = set(query_lower.split())
                choice_words = set(choice_lower.split())
                overlap = len(query_words & choice_words)
                if overlap >= 2:
                    score = min(90, 60 + overlap * 10)
                    if score > best_score:
                        best_score = score
                        best_match = choice
            
            if best_match and best_score >= threshold:
                return (best_match, best_score)
        
        return (None, 0)
    
    def _get_expected_hp(self, model: str) -> Optional[int]:
        """Get expected HP for a model from mapping."""
        if not model:
            return None
        
        model_upper = model.upper()
        
        # Try exact key match first
        for key, hp in self.MODEL_HP_MAP.items():
            if key in model_upper:
                return hp
        
        # Try extracting brand and number
        import re
        for key, hp in self.MODEL_HP_MAP.items():
            # Extract numbers from both
            key_nums = re.findall(r'\d+', key)
            model_nums = re.findall(r'\d+', model_upper)
            
            if key_nums and model_nums and key_nums[0] == model_nums[0]:
                # Same model number
                key_brand = key.split()[0]
                if key_brand in model_upper:
                    return hp
        
        return None
    
    def calculate_document_accuracy(
        self,
        prediction: Dict,
        ground_truth: Dict,
        hp_tolerance: float = 0.05,
        cost_tolerance: float = 0.05
    ) -> Dict:
        """
        Calculate accuracy against ground truth.
        
        Returns: Dict with per-field and overall accuracy
        """
        results = {'fields': {}, 'all_correct': True}
        
        # Text fields - fuzzy match
        for field in ['dealer_name', 'model_name']:
            pred_val = prediction.get('fields', {}).get(field)
            gt_val = ground_truth.get('fields', {}).get(field)
            
            if not gt_val:
                results['fields'][field] = True
                continue
            
            if not pred_val:
                results['fields'][field] = False
                results['all_correct'] = False
                continue
            
            threshold = 90 if field == 'dealer_name' else 95
            _, score = self.fuzzy_match(str(pred_val), [str(gt_val)], threshold=0)
            results['fields'][field] = score >= threshold
            if not results['fields'][field]:
                results['all_correct'] = False
        
        # Numeric fields - tolerance check
        for field, tol in [('horse_power', hp_tolerance), ('asset_cost', cost_tolerance)]:
            pred_val = prediction.get('fields', {}).get(field)
            gt_val = ground_truth.get('fields', {}).get(field)
            
            if gt_val is None:
                results['fields'][field] = pred_val is None
            elif pred_val is None:
                results['fields'][field] = False
                results['all_correct'] = False
            else:
                try:
                    diff = abs(float(pred_val) - float(gt_val))
                    results['fields'][field] = diff <= float(gt_val) * tol
                except (ValueError, TypeError):
                    results['fields'][field] = False
                
                if not results['fields'][field]:
                    results['all_correct'] = False
        
        return results
