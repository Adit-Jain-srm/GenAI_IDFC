"""
VLM Extractor Module
Vision Language Model extraction using Qwen2.5-VL or OpenAI API fallback

Prompt Engineering Best Practices Applied:
- Domain-specific persona (IDFC Bank document processing expert)
- Chain-of-thought reasoning for complex extractions
- Few-shot examples for consistent output
- Multilingual awareness (English, Hindi, Gujarati)
- Structured output with validation rules
- Confidence scoring guidelines
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
AZURE_OPENAI_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    pass

try:
    from openai import AzureOpenAI
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    pass


class VLMExtractor:
    """
    Vision Language Model extractor for document field extraction.
    Supports local Qwen2.5-VL or OpenAI GPT-4o-mini fallback.
    
    Optimized for IDFC Bank's tractor loan quotation processing with:
    - Professional prompt engineering
    - Multilingual document handling
    - High accuracy field extraction
    """
    
    # System prompt establishing expert persona and domain context
    SYSTEM_PROMPT = """You are an expert Document AI specialist working for IDFC First Bank's tractor loan processing division. Your role is to accurately extract structured data from tractor dealer quotations and invoices to support loan disbursal decisions.

DOMAIN EXPERTISE:
- You understand Indian tractor brands: Mahindra, John Deere, TAFE, Swaraj, Sonalika, Massey Ferguson, New Holland, Kubota, Eicher, Escorts (Farmtrac/Powertrac)
- You can read documents in English, Hindi (Devanagari script), and Gujarati
- You understand Indian currency formats (₹, Rs., lakhs notation like 5,25,000)
- You recognize dealer letterheads, quotation formats, and invoice structures

ACCURACY STANDARDS:
- IDFC Bank requires ≥95% document-level accuracy
- Each field extraction directly impacts loan approval decisions
- When uncertain, indicate lower confidence rather than guessing"""

    def __init__(
        self,
        provider: str = 'openai',
        model_name: str = 'gpt-4o-mini',
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        azure_api_version: str = '2024-02-15-preview',
        max_tokens: int = 1500,
        temperature: float = 0.1
    ):
        self.provider = provider
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Azure-specific settings
        self.azure_endpoint = azure_endpoint
        self.azure_deployment = azure_deployment
        self.azure_api_version = azure_api_version
        
        self.model = None
        self.processor = None
        self.client = None
        self._tokens_used = 0
        
        if provider == 'openai':
            self._init_openai(api_key)
        elif provider == 'azure':
            self._init_azure(api_key)
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
    
    def _init_azure(self, api_key: Optional[str]):
        """Initialize Azure OpenAI client."""
        if not AZURE_OPENAI_AVAILABLE:
            raise ImportError("OpenAI package required: pip install openai>=1.0.0")
        
        # Get credentials from params or environment
        api_key = api_key or os.environ.get('AZURE_OPENAI_API_KEY')
        endpoint = self.azure_endpoint or os.environ.get('AZURE_OPENAI_ENDPOINT')
        deployment = self.azure_deployment or os.environ.get('AZURE_OPENAI_DEPLOYMENT')
        api_version = self.azure_api_version or os.environ.get('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
        
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY required (env var or --azure_api_key)")
        if not endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT required (env var or --azure_endpoint)")
        if not deployment:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT required (env var or --azure_deployment)")
        
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
        # Use deployment name as model name for Azure
        self.model_name = deployment
        logger.info(f"Azure OpenAI initialized: endpoint={endpoint}, deployment={deployment}")
    
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
        
        if self.provider in ('openai', 'azure'):
            return self._extract_openai(image, prompt)
        else:
            return self._extract_qwen(image, prompt)
    
    def _get_extraction_prompt(self) -> str:
        """
        Get the optimized extraction prompt with best practices:
        - Chain-of-thought reasoning
        - Few-shot examples
        - Structured output format
        - Validation guidelines
        """
        return """TASK: Extract key fields from this tractor dealer quotation/invoice for IDFC Bank loan processing.

===== VISUAL LAYOUT ANALYSIS =====

First, analyze the document structure:

**DOCUMENT REGIONS:**
- TOP (0-20%): Usually contains dealer letterhead, logo, company name, contact info
- MIDDLE (20-75%): Product details, specifications table, pricing breakdown
- BOTTOM (75-100%): Totals, signatures, stamps, terms & conditions

**COMMON LAYOUTS:**
1. **Letterhead Style**: Dealer name prominent at top, items listed below
2. **Table Format**: Rows with Description | Specs | Price columns
3. **Form Style**: Key: Value pairs aligned vertically
4. **Mixed**: Combination of above

**KEY-VALUE PAIRS**: Look for patterns like:
- "Label: Value" (colon-separated)
- Label on LEFT, value on RIGHT (same row)
- Label ABOVE, value BELOW (stacked)

===== STEP-BY-STEP EXTRACTION PROCESS =====

1. **DEALER NAME** - Look for:
   - Company name in letterhead/header (TOP 20% of document)
   - Text after "From:", "Dealer:", "Authorized Dealer"
   - Business registration details, GST numbers nearby
   - Look for logos with company names
   - Hindi: "डीलर", "विक्रेता" | Gujarati: "ડીલર"

2. **MODEL NAME** - Look for:
   - Tractor model in product description section (MIDDLE region)
   - Often in a table row or as a prominent item
   - Format: [Brand] [Number] [Variant] (e.g., "Mahindra 575 DI XP Plus")
   - Common brands: Mahindra, John Deere, Swaraj, Sonalika, TAFE, Massey Ferguson, New Holland, Kubota, Eicher, Farmtrac, Powertrac
   - May be in bold or larger font
   - Hindi: "मॉडल", "ट्रैक्टर" | Gujarati: "મોડલ", "ટ્રેક્ટર"

3. **HORSE POWER** - Look for:
   - Often in a specifications table or near model name
   - Number followed by "HP", "hp", "Horse Power", "BHP", "H.P."
   - May be in a column labeled "Power" or "Engine"
   - Valid range: 15-150 HP for tractors
   - **OFTEN HANDWRITTEN** - Look for handwritten numbers near "H.P." or "HP" text
   - May appear inline with model: "SWARAJ 744 FE TRACTOR .....25..... H.P."
   - Common values: 25, 30, 35, 40, 45, 50, 55, 60, 65, 75, 90 HP
   - Hindi: "अश्वशक्ति", "एचपी" | Gujarati: "એચપી", "અશ્વશક્તિ"

4. **ASSET COST** - Look for:
   - Final/Total amount (BOTTOM region, often last row of table)
   - Usually the LARGEST number on the document
   - Labels: "Total", "Grand Total", "Net Amount", "Ex-Showroom", "On-Road Price"
   - Indian format: ₹5,25,000 or Rs. 5,25,000/- means 525000
   - Valid range: ₹50,000 - ₹50,00,000
   - Often bold or highlighted
   - Look for handwritten amounts - they are often the final price
   - Hindi: "कुल", "राशि", "मूल्य" | Gujarati: "કુલ", "કિંમત"
   
   ⚠️ **CRITICAL - DO NOT CONFUSE WITH PIN CODES:**
   - Indian PIN codes are 6-digit numbers (e.g., 306115, 382010)
   - PIN codes appear in ADDRESS sections near: "Dist.", "District", city names, "Rajasthan", "Gujarat", etc.
   - Example: "RANI - 306115, Dist.-Pali (Raj.)" → 306115 is a PIN code, NOT a price
   - Prices have ₹/Rs. prefix and appear near "Total", "Amount", "Price" labels
   - When in doubt, prefer the larger number with price formatting (commas like 5,50,000)

===== FEW-SHOT EXAMPLES =====

Example 1 - Clear English Invoice (Letterhead Style):
Input: Document shows "Sharma Tractors Pvt Ltd" header at top, "Mahindra 575 DI" in description, "50 HP" in specs, "Total: ₹5,25,000/-"
Output: {"dealer_name": "Sharma Tractors Pvt Ltd", "model_name": "Mahindra 575 DI", "horse_power": 50, "asset_cost": 525000, "confidence": 0.95}

Example 2 - Hindi Document:
Input: Document shows "गुप्ता ट्रैक्टर्स" header, "जॉन डियर 5050D" model, "50 एचपी", "कुल राशि: ₹6,50,000"
Output: {"dealer_name": "Gupta Tractors", "model_name": "John Deere 5050D", "horse_power": 50, "asset_cost": 650000, "confidence": 0.90}

Example 3 - Table Format Invoice:
Input: Document has a table with columns [Item, Description, Specs, Price]. Row 1: "Swaraj 744 XT | 48 HP Tractor | 48 HP | Rs.6,15,000". Footer shows "Grand Total: Rs.6,50,000/-"
Output: {"dealer_name": null, "model_name": "Swaraj 744 XT", "horse_power": 48, "asset_cost": 650000, "confidence": 0.85, "extraction_notes": "Dealer name not found in visible area"}

Example 4 - Gujarati Document (Mixed Script):
Input: Document shows "પટેલ ટ્રેક્ટર્સ" at top, model "Sonalika DI 50" in body, "50 એચપી", total "કુલ: ₹4,75,000"
Output: {"dealer_name": "Patel Tractors", "model_name": "Sonalika DI 50", "horse_power": 50, "asset_cost": 475000, "confidence": 0.88}

Example 5 - Partial Information (Low Quality):
Input: Scanned document with dealer name and model clearly visible, but HP section is blurry and multiple price values shown without clear total
Output: {"dealer_name": "ABC Motors", "model_name": "Swaraj 744 FE", "horse_power": null, "asset_cost": null, "confidence": 0.55, "extraction_notes": "HP illegible, cost ambiguous - multiple values without clear total"}

Example 6 - PIN Code vs Price (IMPORTANT):
Input: Header shows "KISSAN TRACTOR AGENCY, Kenpura Road, RANI - 306115, Dist.-Pali (Raj.)". Body shows "SWARAJ 744 FE TRACTOR ...25... H.P." with handwritten "25". Price section shows "Rs. 5,50,600" and "Total Amount Rs.: 5,50,000/-"
Output: {"dealer_name": "Kissan Tractor Agency", "model_name": "Swaraj 744 FE", "horse_power": 25, "asset_cost": 550000, "confidence": 0.90, "extraction_notes": "306115 is PIN code in address, not price. Used handwritten total amount."}
WRONG: {"asset_cost": 306115} ← This is the PIN code from address, NOT the price!

===== OUTPUT FORMAT =====

Return ONLY a valid JSON object with this exact structure:
```json
{
  "dealer_name": "string or null",
  "model_name": "string or null (include brand + model + variant)",
  "horse_power": integer or null (15-150 range only),
  "asset_cost": integer or null (50000-5000000 range, no currency symbols),
  "confidence": float (0.0-1.0),
  "extraction_notes": "brief note on any issues or uncertainties"
}
```

===== CONFIDENCE SCORING GUIDELINES =====
- 0.90-1.00: All fields clearly visible and extracted with high certainty
- 0.75-0.89: Most fields clear, 1-2 minor uncertainties
- 0.50-0.74: Some fields unclear or potentially incorrect
- 0.25-0.49: Multiple fields uncertain, low quality document
- 0.00-0.24: Unable to extract reliably, severe quality issues

===== CRITICAL RULES =====
1. DO NOT hallucinate or guess - use null for uncertain fields
2. Transliterate Hindi/Gujarati names to English (e.g., "गुप्ता" → "Gupta")
3. Always normalize cost to plain integer (5,25,000 → 525000)
4. Include full model name with variant (e.g., "Mahindra 575 DI XP Plus" not just "575")
5. Return ONLY the JSON - no explanations before or after

Now analyze the provided document image and extract the fields:"""

    def _extract_openai(self, image: Union[Image.Image, np.ndarray, str, Path], prompt: str) -> Dict:
        """
        Extract using OpenAI API with optimized prompting.
        
        Uses:
        - System message for domain expertise persona
        - High detail image analysis
        - Low temperature for consistent output
        """
        pil_image = self._to_pil(image)
        base64_image = self._image_to_base64(pil_image)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": self.SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"}  # Enforce JSON output
            )
            
            self._tokens_used = response.usage.total_tokens
            output = response.choices[0].message.content
            result = self._parse_json(output)
            
            # Post-process and validate
            return self._validate_extraction(result)
            
        except Exception as e:
            logger.error(f"OpenAI extraction failed: {e}")
            return {"error": str(e), "confidence": 0.0}
    
    def _extract_qwen(self, image: Union[Image.Image, np.ndarray, str, Path], prompt: str) -> Dict:
        """
        Extract using local Qwen model with optimized prompting.
        
        Combines system prompt with user prompt for Qwen's chat format.
        """
        import torch
        
        pil_image = self._to_pil(image)
        
        # Combine system and user prompts for Qwen format
        combined_prompt = f"{self.SYSTEM_PROMPT}\n\n{prompt}"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": combined_prompt}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[pil_image], padding=True, return_tensors="pt").to('cuda')
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=False  # Greedy decoding for consistency
            )
        
        output = self.processor.batch_decode(output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        result = self._parse_json(output)
        
        # Post-process and validate
        return self._validate_extraction(result)
    
    def _parse_json(self, text: str) -> Dict:
        """
        Robust JSON parsing from LLM response.
        
        Handles:
        - Markdown code blocks
        - Extra text before/after JSON
        - Common formatting issues
        """
        try:
            text = text.strip()
            
            # Extract JSON block from markdown
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0]
            elif '```' in text:
                parts = text.split('```')
                for part in parts:
                    if '{' in part and '}' in part:
                        text = part
                        break
            
            # Find JSON object boundaries
            start = text.find('{')
            end = text.rfind('}') + 1
            
            if start >= 0 and end > start:
                json_str = text[start:end]
                # Fix common issues
                json_str = json_str.replace("'", '"')  # Single to double quotes
                json_str = json_str.replace('None', 'null')  # Python None to JSON null
                json_str = json_str.replace('True', 'true').replace('False', 'false')
                return json.loads(json_str)
            
            return {"error": "No JSON found", "confidence": 0.0}
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            return {"error": f"JSON parse error: {e}", "confidence": 0.0}
    
    def _validate_extraction(self, result: Dict) -> Dict:
        """
        Post-process and validate extraction results.
        
        Applies:
        - Type coercion
        - Range validation
        - Confidence adjustment
        """
        if "error" in result:
            return result
        
        validated = {}
        
        # Dealer name - clean and normalize
        dealer = result.get('dealer_name')
        if dealer and isinstance(dealer, str):
            dealer = dealer.strip()
            # Remove common suffixes for consistency
            validated['dealer_name'] = dealer if len(dealer) > 2 else None
        else:
            validated['dealer_name'] = None
        
        # Model name - ensure proper format
        model = result.get('model_name')
        if model and isinstance(model, str):
            model = model.strip()
            validated['model_name'] = model if len(model) > 3 else None
        else:
            validated['model_name'] = None
        
        # Horse power - validate range
        hp = result.get('horse_power')
        if hp is not None:
            try:
                hp = int(float(hp))
                validated['horse_power'] = hp if 15 <= hp <= 150 else None
            except (ValueError, TypeError):
                validated['horse_power'] = None
        else:
            validated['horse_power'] = None
        
        # Asset cost - validate range and clean
        # Note: PIN code filtering is NOT applied here because:
        # 1. VLM prompt explicitly instructs to distinguish PIN codes from prices
        # 2. Valid tractor prices can be ₹3-4 lakhs (300000-400000) for entry-level models
        # 3. Threshold-based filtering incorrectly rejects valid prices
        # The VLM's contextual understanding is trusted for this distinction
        cost = result.get('asset_cost')
        if cost is not None:
            try:
                # Handle string numbers with commas
                if isinstance(cost, str):
                    cost = cost.replace(',', '').replace('₹', '').replace('Rs', '').strip()
                cost = int(float(cost))
                
                # Validate range only - trust VLM for PIN vs price distinction
                validated['asset_cost'] = cost if 50000 <= cost <= 5000000 else None
            except (ValueError, TypeError):
                validated['asset_cost'] = None
        else:
            validated['asset_cost'] = None
        
        # Confidence - validate and adjust
        conf = result.get('confidence', 0.5)
        try:
            conf = float(conf)
            conf = max(0.0, min(1.0, conf))
            
            # Adjust confidence based on extraction completeness
            fields_found = sum(1 for v in [
                validated['dealer_name'],
                validated['model_name'],
                validated['horse_power'],
                validated['asset_cost']
            ] if v is not None)
            
            # Penalize if many fields are missing
            if fields_found < 2:
                conf = min(conf, 0.4)
            elif fields_found < 3:
                conf = min(conf, 0.7)
            
            validated['confidence'] = round(conf, 2)
        except (ValueError, TypeError):
            validated['confidence'] = 0.3
        
        # Include extraction notes if provided
        if 'extraction_notes' in result:
            validated['extraction_notes'] = result['extraction_notes']
        
        return validated
    
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
