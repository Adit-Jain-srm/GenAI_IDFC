"""
VLM Extractor Module
Vision Language Model extraction using Qwen2.5-VL or OpenAI API fallback
"""

import os
import json
import base64
from io import BytesIO
from typing import Dict, Optional, Union
from pathlib import Path
import numpy as np
from PIL import Image
from loguru import logger

# Lazy imports for resource efficiency
QWEN_AVAILABLE = False
OPENAI_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    pass


class VLMExtractor:
    """
    Vision Language Model extractor for document field extraction.
    Supports local Qwen2.5-VL or OpenAI GPT-4o-mini fallback.
    """
    
    def __init__(
        self,
        provider: str = 'openai',
        model_name: str = 'gpt-4o-mini',
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.1
    ):
        self.provider = provider
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        self.model = None
        self.processor = None
        self.client = None
        self._tokens_used = 0
        
        if provider == 'openai':
            self._init_openai(api_key)
        elif provider == 'qwen':
            self._init_qwen()
    
    def _init_openai(self, api_key: Optional[str]):
        """Initialize OpenAI client."""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package required: pip install openai")
        
        api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY required")
        
        self.client = OpenAI(api_key=api_key)
    
    def _init_qwen(self):
        """Initialize Qwen model (lazy load)."""
        global QWEN_AVAILABLE
        try:
            import torch
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            QWEN_AVAILABLE = True
            
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map='auto'
            )
            self.processor = AutoProcessor.from_pretrained(self.model_name)
        except Exception as e:
            logger.warning(f"Qwen init failed: {e}. Use OpenAI fallback.")
            raise
    
    def extract_fields(self, image: Union[Image.Image, np.ndarray, str, Path]) -> Dict:
        """
        Extract invoice fields from image.
        
        Returns:
            Dict with dealer_name, model_name, horse_power, asset_cost
        """
        prompt = self._get_extraction_prompt()
        
        if self.provider == 'openai':
            return self._extract_openai(image, prompt)
        else:
            return self._extract_qwen(image, prompt)
    
    def _get_extraction_prompt(self) -> str:
        """Get the extraction prompt."""
        return """Analyze this invoice/quotation document and extract:

1. dealer_name: The dealer/seller company name (from header/letterhead)
2. model_name: Tractor model (e.g., "Mahindra 575 DI", "John Deere 5050D")
3. horse_power: HP value as integer (e.g., "50 HP" → 50)
4. asset_cost: Total price as integer without currency (e.g., "₹5,25,000" → 525000)

Return ONLY valid JSON:
{
  "dealer_name": "extracted name or null",
  "model_name": "extracted model or null",
  "horse_power": number or null,
  "asset_cost": number or null,
  "confidence": 0.0 to 1.0
}

Rules:
- horse_power must be number 15-150 or null
- asset_cost must be number 50000-5000000 or null
- confidence reflects overall extraction quality
- Return ONLY the JSON, no other text"""

    def _extract_openai(self, image: Union[Image.Image, np.ndarray, str, Path], prompt: str) -> Dict:
        """Extract using OpenAI API."""
        pil_image = self._to_pil(image)
        base64_image = self._image_to_base64(pil_image)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}},
                        {"type": "text", "text": prompt}
                    ]
                }],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            self._tokens_used = response.usage.total_tokens
            output = response.choices[0].message.content
            return self._parse_json(output)
            
        except Exception as e:
            logger.error(f"OpenAI extraction failed: {e}")
            return {"error": str(e), "confidence": 0.0}
    
    def _extract_qwen(self, image: Union[Image.Image, np.ndarray, str, Path], prompt: str) -> Dict:
        """Extract using local Qwen model."""
        import torch
        
        pil_image = self._to_pil(image)
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt}
            ]
        }]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[pil_image], padding=True, return_tensors="pt").to('cuda')
        
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens, temperature=self.temperature)
        
        output = self.processor.batch_decode(output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return self._parse_json(output)
    
    def _parse_json(self, text: str) -> Dict:
        """Parse JSON from response."""
        try:
            text = text.strip()
            
            # Extract JSON block
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0]
            elif '```' in text:
                text = text.split('```')[1].split('```')[0]
            
            # Find JSON object
            start = text.find('{')
            end = text.rfind('}') + 1
            
            if start >= 0 and end > start:
                return json.loads(text[start:end])
            
            return {"error": "No JSON found", "confidence": 0.0}
            
        except json.JSONDecodeError as e:
            return {"error": f"JSON parse error: {e}", "confidence": 0.0}
    
    def _to_pil(self, image: Union[Image.Image, np.ndarray, str, Path]) -> Image.Image:
        """Convert to PIL Image."""
        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image)
        else:
            return Image.open(image)
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert to base64."""
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=90)
        return base64.b64encode(buffered.getvalue()).decode()
    
    @property
    def tokens_used(self) -> int:
        """Get tokens used in last call."""
        return self._tokens_used
