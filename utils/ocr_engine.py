"""
OCR Engine Module
PaddleOCR wrapper with multilingual support (English, Hindi, Gujarati)

Key Features:
- Multi-language OCR with automatic detection
- Layout-aware text extraction
- Confidence scoring
"""

from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import numpy as np
from PIL import Image
from loguru import logger

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    logger.warning("PaddleOCR not available. Install: pip install paddleocr paddlepaddle")


class OCREngine:
    """
    Multilingual OCR Engine using PaddleOCR.
    
    Supports: English, Hindi (Devanagari), Gujarati, and mixed text.
    Uses multiple OCR passes for best accuracy.
    """
    
    # PaddleOCR language codes
    LANG_CODES = {
        'english': 'en',
        'hindi': 'devanagari',  # Devanagari script for Hindi
        'gujarati': 'gu',
        'multilingual': 'ch'   # Chinese model handles mixed scripts well
    }
    
    def __init__(
        self,
        use_gpu: bool = True,
        use_angle_cls: bool = True,
        drop_score: float = 0.4,
        enable_multilingual: bool = True
    ):
        """
        Initialize OCR engine with multilingual support.
        
        Args:
            use_gpu: Use GPU acceleration
            use_angle_cls: Enable angle classification for rotated text
            drop_score: Minimum confidence threshold
            enable_multilingual: Enable Hindi/Gujarati in addition to English
        """
        if not PADDLE_AVAILABLE:
            raise ImportError("PaddleOCR required. Install: pip install paddleocr paddlepaddle")
        
        self.drop_score = drop_score
        self.enable_multilingual = enable_multilingual
        
        # Initialize OCR engines
        self.ocr_engines = {}
        
        # Primary: English OCR (good for alphanumeric, model names, prices)
        logger.info("Loading English OCR model...")
        self.ocr_engines['en'] = PaddleOCR(
            lang='en',
            use_angle_cls=use_angle_cls,
            use_gpu=use_gpu,
            show_log=False,
            det_db_thresh=0.3,
            det_db_box_thresh=0.5
        )
        
        # Secondary: Devanagari OCR (Hindi)
        if enable_multilingual:
            try:
                logger.info("Loading Devanagari (Hindi) OCR model...")
                self.ocr_engines['hi'] = PaddleOCR(
                    lang='devanagari',
                    use_angle_cls=use_angle_cls,
                    use_gpu=use_gpu,
                    show_log=False
                )
            except Exception as e:
                logger.warning(f"Hindi OCR model not loaded: {e}")
        
        logger.info(f"OCR Engine initialized with {len(self.ocr_engines)} language(s)")
    
    def extract_text(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        languages: List[str] = None
    ) -> Dict:
        """
        Extract text from image with multi-language support.
        
        Runs OCR in multiple languages and merges results for best coverage.
        
        Args:
            image: Input image (PIL, numpy array, or path)
            languages: List of languages to try ['en', 'hi']. Default: all available
            
        Returns:
            Dict with 'texts', 'boxes', 'confidences', 'full_text', 'language'
        """
        img_array = self._to_numpy(image)
        languages = languages or list(self.ocr_engines.keys())
        
        all_results = {}
        
        # Run OCR for each language
        for lang in languages:
            if lang in self.ocr_engines:
                try:
                    result = self.ocr_engines[lang].ocr(img_array, cls=True)
                    parsed = self._parse_results(result)
                    all_results[lang] = parsed
                except Exception as e:
                    logger.warning(f"OCR failed for {lang}: {e}")
        
        # Merge results from all languages
        merged = self._merge_multilingual_results(all_results)
        
        # Detect primary language
        merged['language'] = self.detect_language(merged.get('full_text', ''))
        
        return merged
    
    def _parse_results(self, result: List) -> Dict:
        """Parse PaddleOCR raw results."""
        texts = []
        boxes = []
        confidences = []
        
        if result and result[0]:
            for line in result[0]:
                if line:
                    try:
                        box, (text, conf) = line
                        if conf >= self.drop_score and text.strip():
                            texts.append(text.strip())
                            boxes.append(self._normalize_box(box))
                            confidences.append(float(conf))
                    except (ValueError, TypeError) as e:
                        continue
        
        return {
            'texts': texts,
            'boxes': boxes,
            'confidences': confidences,
            'full_text': '\n'.join(texts),
            'avg_confidence': np.mean(confidences) if confidences else 0.0,
            'count': len(texts)
        }
    
    def _merge_multilingual_results(self, all_results: Dict) -> Dict:
        """
        Merge OCR results from multiple languages.
        
        Strategy: 
        - Use English results as base (better for numbers, model names)
        - Add unique Hindi/Gujarati text not captured by English
        - Avoid duplicate detections using IoU
        """
        if not all_results:
            return {
                'texts': [], 'boxes': [], 'confidences': [],
                'full_text': '', 'avg_confidence': 0.0
            }
        
        # Start with English results (best for structured data)
        if 'en' in all_results:
            merged = {
                'texts': list(all_results['en']['texts']),
                'boxes': list(all_results['en']['boxes']),
                'confidences': list(all_results['en']['confidences'])
            }
        else:
            merged = {'texts': [], 'boxes': [], 'confidences': []}
        
        # Add non-duplicate results from other languages
        for lang, result in all_results.items():
            if lang == 'en':
                continue
            
            for i, (text, box, conf) in enumerate(zip(
                result['texts'], result['boxes'], result['confidences']
            )):
                # Check if this box overlaps with existing
                is_duplicate = False
                for existing_box in merged['boxes']:
                    if self._calculate_iou(box, existing_box) > 0.5:
                        is_duplicate = True
                        break
                
                # Add if unique and contains non-ASCII (likely Hindi/Gujarati)
                if not is_duplicate and self._has_non_ascii(text):
                    merged['texts'].append(text)
                    merged['boxes'].append(box)
                    merged['confidences'].append(conf)
        
        # Sort by position
        if merged['boxes']:
            sorted_data = self._sort_by_position(
                merged['texts'], merged['boxes'], merged['confidences']
            )
            merged['texts'], merged['boxes'], merged['confidences'] = sorted_data
        
        merged['full_text'] = '\n'.join(merged['texts'])
        merged['avg_confidence'] = np.mean(merged['confidences']) if merged['confidences'] else 0.0
        
        return merged
    
    def _normalize_box(self, box: List) -> List[int]:
        """Convert polygon to [x1, y1, x2, y2] format."""
        box = np.array(box)
        return [
            int(min(box[:, 0])),
            int(min(box[:, 1])),
            int(max(box[:, 0])),
            int(max(box[:, 1]))
        ]
    
    def _sort_by_position(
        self,
        texts: List[str],
        boxes: List[List[int]],
        confidences: List[float]
    ) -> Tuple[List, List, List]:
        """Sort by reading order (top-to-bottom, left-to-right)."""
        y_tolerance = 20
        
        items = list(zip(texts, boxes, confidences))
        items.sort(key=lambda x: (x[1][1] // y_tolerance, x[1][0]))
        
        return (
            [i[0] for i in items],
            [i[1] for i in items],
            [i[2] for i in items]
        )
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculate Intersection over Union."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x1 >= x2 or y1 >= y2:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _has_non_ascii(self, text: str) -> bool:
        """Check if text contains non-ASCII characters (Hindi/Gujarati)."""
        return any(ord(c) > 127 for c in text)
    
    def _to_numpy(self, image: Union[Image.Image, np.ndarray, str, Path]) -> np.ndarray:
        """Convert various image formats to numpy array."""
        if isinstance(image, np.ndarray):
            return image
        elif isinstance(image, Image.Image):
            return np.array(image)
        else:
            return np.array(Image.open(image))
    
    def detect_language(self, text: str) -> str:
        """
        Detect primary language based on script analysis.
        
        Returns: 'english', 'hindi', 'gujarati', or 'mixed'
        """
        if not text:
            return 'english'
        
        # Count characters by script
        devanagari = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        gujarati = sum(1 for c in text if '\u0A80' <= c <= '\u0AFF')
        ascii_alpha = sum(1 for c in text if c.isascii() and c.isalpha())
        
        total = devanagari + gujarati + ascii_alpha
        if total == 0:
            return 'english'
        
        # Determine primary language
        hi_ratio = devanagari / total
        gu_ratio = gujarati / total
        en_ratio = ascii_alpha / total
        
        if hi_ratio > 0.4:
            return 'hindi' if en_ratio < 0.3 else 'mixed'
        elif gu_ratio > 0.4:
            return 'gujarati' if en_ratio < 0.3 else 'mixed'
        elif en_ratio > 0.6:
            return 'english'
        else:
            return 'mixed'
    
    def get_text_in_region(
        self,
        ocr_result: Dict,
        region: Tuple[int, int, int, int]
    ) -> List[str]:
        """Get text elements within a specific image region."""
        rx1, ry1, rx2, ry2 = region
        texts_in_region = []
        
        for text, box in zip(ocr_result.get('texts', []), ocr_result.get('boxes', [])):
            # Check if box center is in region
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                texts_in_region.append(text)
        
        return texts_in_region
