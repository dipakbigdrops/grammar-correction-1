"""
Fixed Grammar Correction Processor
All indentation and syntax errors resolved
"""
import gc
import os
import json
import re
import threading
import time
import base64
from io import BytesIO
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw
from typing import Tuple, List, Dict, Optional, Any
import logging
import torch

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from app.config import settings
from app.robust_model_loader import load_robust_model, test_model_inference

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-compiled patterns (compiled once at import time, reused per request)
# ---------------------------------------------------------------------------
_NORMALIZE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Transpositions / short OCR swaps
    (re.compile(r'\bteh\b', re.IGNORECASE), 'the'),
    (re.compile(r'\badn\b', re.IGNORECASE), 'and'),
    (re.compile(r'\btaht\b', re.IGNORECASE), 'that'),
    (re.compile(r'\bwihch\b', re.IGNORECASE), 'which'),
    # Common misspellings the model may not catch
    (re.compile(r'\bgrammer\b', re.IGNORECASE), 'grammar'),
    (re.compile(r'\brecieve\b', re.IGNORECASE), 'receive'),
    (re.compile(r'\boccured\b', re.IGNORECASE), 'occurred'),
    (re.compile(r'\bseperate\b', re.IGNORECASE), 'separate'),
    (re.compile(r'\bdefinately\b', re.IGNORECASE), 'definitely'),
    (re.compile(r'\bdefinatly\b', re.IGNORECASE), 'definitely'),
    (re.compile(r'\bcorection\b', re.IGNORECASE), 'correction'),
    (re.compile(r'\bneccessary\b', re.IGNORECASE), 'necessary'),
    (re.compile(r'\bnecesary\b', re.IGNORECASE), 'necessary'),
    (re.compile(r'\baccomodate\b', re.IGNORECASE), 'accommodate'),
    (re.compile(r'\brecomend\b', re.IGNORECASE), 'recommend'),
    (re.compile(r'\breccomend\b', re.IGNORECASE), 'recommend'),
    (re.compile(r'\bthier\b', re.IGNORECASE), 'their'),
    (re.compile(r'\bwierd\b', re.IGNORECASE), 'weird'),
    (re.compile(r'\bbeleive\b', re.IGNORECASE), 'believe'),
    (re.compile(r'\bbelive\b', re.IGNORECASE), 'believe'),
    (re.compile(r'\bgoverment\b', re.IGNORECASE), 'government'),
    (re.compile(r'\bgovernement\b', re.IGNORECASE), 'government'),
    (re.compile(r'\bexistance\b', re.IGNORECASE), 'existence'),
    (re.compile(r'\bwritting\b', re.IGNORECASE), 'writing'),
    (re.compile(r'\buntill\b', re.IGNORECASE), 'until'),
    (re.compile(r'\btruely\b', re.IGNORECASE), 'truly'),
    (re.compile(r'\bfourty\b', re.IGNORECASE), 'forty'),
    (re.compile(r'\bpriveledge\b', re.IGNORECASE), 'privilege'),
    (re.compile(r'\bprivlege\b', re.IGNORECASE), 'privilege'),
    (re.compile(r'\bhieght\b', re.IGNORECASE), 'height'),
    (re.compile(r'\bheigth\b', re.IGNORECASE), 'height'),
    (re.compile(r'\bcommittment\b', re.IGNORECASE), 'commitment'),
    (re.compile(r'\boccasionaly\b', re.IGNORECASE), 'occasionally'),
    (re.compile(r'\bappologize\b', re.IGNORECASE), 'apologize'),
    (re.compile(r'\bknowlege\b', re.IGNORECASE), 'knowledge'),
    (re.compile(r'\bknoweldge\b', re.IGNORECASE), 'knowledge'),
    (re.compile(r'\benviroment\b', re.IGNORECASE), 'environment'),
    (re.compile(r'\benviornment\b', re.IGNORECASE), 'environment'),
    (re.compile(r'\bconvinient\b', re.IGNORECASE), 'convenient'),
    (re.compile(r'\bconvienient\b', re.IGNORECASE), 'convenient'),
    (re.compile(r'\bmanagment\b', re.IGNORECASE), 'management'),
    (re.compile(r'\bmanagament\b', re.IGNORECASE), 'management'),
    (re.compile(r'\bpersonell\b', re.IGNORECASE), 'personnel'),
    (re.compile(r'\bconsious\b', re.IGNORECASE), 'conscious'),
    (re.compile(r'\bconcious\b', re.IGNORECASE), 'conscious'),
    (re.compile(r'\brecognise\b', re.IGNORECASE), 'recognize'),
    (re.compile(r'\bpronouciation\b', re.IGNORECASE), 'pronunciation'),
    (re.compile(r'\bpersistance\b', re.IGNORECASE), 'persistence'),
    (re.compile(r'\boccurence\b', re.IGNORECASE), 'occurrence'),
]

# Safe contractions: words that are NEVER standalone valid English words
_CONTRACTION_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r'\bdont\b', re.IGNORECASE), "don't"),
    (re.compile(r'\bdoesnt\b', re.IGNORECASE), "doesn't"),
    (re.compile(r'\bdidnt\b', re.IGNORECASE), "didn't"),
    (re.compile(r'\bhavent\b', re.IGNORECASE), "haven't"),
    (re.compile(r'\bhasnt\b', re.IGNORECASE), "hasn't"),
    (re.compile(r'\bhadnt\b', re.IGNORECASE), "hadn't"),
    (re.compile(r'\bisnt\b', re.IGNORECASE), "isn't"),
    (re.compile(r'\bwasnt\b', re.IGNORECASE), "wasn't"),
    (re.compile(r'\bwerent\b', re.IGNORECASE), "weren't"),
    (re.compile(r'\bwouldnt\b', re.IGNORECASE), "wouldn't"),
    (re.compile(r'\bcouldnt\b', re.IGNORECASE), "couldn't"),
    (re.compile(r'\bshouldnt\b', re.IGNORECASE), "shouldn't"),
    (re.compile(r'\barent\b', re.IGNORECASE), "aren't"),
    (re.compile(r'\bwouldve\b', re.IGNORECASE), "would've"),
    (re.compile(r'\bcouldve\b', re.IGNORECASE), "could've"),
    (re.compile(r'\bshouldve\b', re.IGNORECASE), "should've"),
    (re.compile(r'\bmustve\b', re.IGNORECASE), "must've"),
    (re.compile(r'\bmightve\b', re.IGNORECASE), "might've"),
]

_FALLBACK_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Spelling
    (re.compile(r'\bgrammer\b', re.IGNORECASE), 'grammar'),
    (re.compile(r'\bteh\b', re.IGNORECASE), 'the'),
    (re.compile(r'\badn\b', re.IGNORECASE), 'and'),
    (re.compile(r'\bthier\b', re.IGNORECASE), 'their'),
    (re.compile(r'\brecieve\b', re.IGNORECASE), 'receive'),
    (re.compile(r'\boccured\b', re.IGNORECASE), 'occurred'),
    (re.compile(r'\bseperate\b', re.IGNORECASE), 'separate'),
    (re.compile(r'\bdefinately\b', re.IGNORECASE), 'definitely'),
    (re.compile(r'\bcorection\b', re.IGNORECASE), 'correction'),
    (re.compile(r'\bneccessary\b', re.IGNORECASE), 'necessary'),
    (re.compile(r'\bnecesary\b', re.IGNORECASE), 'necessary'),
    (re.compile(r'\baccomodate\b', re.IGNORECASE), 'accommodate'),
    (re.compile(r'\brecomend\b', re.IGNORECASE), 'recommend'),
    (re.compile(r'\breccomend\b', re.IGNORECASE), 'recommend'),
    (re.compile(r'\bwierd\b', re.IGNORECASE), 'weird'),
    (re.compile(r'\bbeleive\b', re.IGNORECASE), 'believe'),
    (re.compile(r'\bexistance\b', re.IGNORECASE), 'existence'),
    (re.compile(r'\benviroment\b', re.IGNORECASE), 'environment'),
    # Contractions (safe ones only)
    (re.compile(r'\bdont\b', re.IGNORECASE), "don't"),
    (re.compile(r'\bdoesnt\b', re.IGNORECASE), "doesn't"),
    (re.compile(r'\bdidnt\b', re.IGNORECASE), "didn't"),
    (re.compile(r'\bhavent\b', re.IGNORECASE), "haven't"),
    (re.compile(r'\bhasnt\b', re.IGNORECASE), "hasn't"),
    (re.compile(r'\bhadnt\b', re.IGNORECASE), "hadn't"),
    (re.compile(r'\bisnt\b', re.IGNORECASE), "isn't"),
    (re.compile(r'\bwasnt\b', re.IGNORECASE), "wasn't"),
    (re.compile(r'\bwerent\b', re.IGNORECASE), "weren't"),
    (re.compile(r'\bwouldnt\b', re.IGNORECASE), "wouldn't"),
    (re.compile(r'\bcouldnt\b', re.IGNORECASE), "couldn't"),
    (re.compile(r'\bshouldnt\b', re.IGNORECASE), "shouldn't"),
    (re.compile(r'\barent\b', re.IGNORECASE), "aren't"),
    # OCR artefacts
    (re.compile(r'\btaht\b', re.IGNORECASE), 'that'),
    (re.compile(r'\bt eh\b', re.IGNORECASE), 'the'),
    (re.compile(r'\bwi th\b', re.IGNORECASE), 'with'),
    (re.compile(r'\bfo r\b', re.IGNORECASE), 'for'),
    (re.compile(r'\bint he\b', re.IGNORECASE), 'in the'),
    (re.compile(r'\bont he\b', re.IGNORECASE), 'on the'),
    (re.compile(r'\bwit h\b', re.IGNORECASE), 'with'),
    (re.compile(r'\bfrorn\b', re.IGNORECASE), 'from'),
    (re.compile(r'\bwhic h\b', re.IGNORECASE), 'which'),
    (re.compile(r'\bso me\b', re.IGNORECASE), 'some'),
    (re.compile(r'\bhav e\b', re.IGNORECASE), 'have'),
    (re.compile(r'\brn\b'), 'm'),
    (re.compile(r'\bvv\b'), 'w'),
]

# Capitalization helpers (compiled once)
_STANDALONE_I_RE = re.compile(r'\bi\b')
_SENTENCE_START_RE = re.compile(r'([.!?])\s+([a-z])')
# Matches words that contain a lowercase letter immediately followed by an uppercase letter
# anywhere after position 0 — the signature of OCR/typo mid-word stray caps ("heLLo", "tHe").
# All-caps words (acronyms: "NATO", "SQL") are explicitly excluded by the handler.
_MID_WORD_CAPS_RE = re.compile(r'\b[a-zA-Z]*[a-z][A-Z][a-zA-Z]*\b')


def _lower_unless_acronym(m: re.Match) -> str:
    """Used by _fix_capitalization to lowercase irregular-cased words while preserving acronyms."""
    word = m.group(0)
    return word if word.isupper() else word.lower()


# HTML elements that are true content-bearing leaf blocks (not structural containers).
# Only the innermost matching element is corrected to avoid double-processing nested content.
_HTML_BLOCK_TAGS = frozenset({
    'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'li', 'td', 'th', 'dt', 'dd',
    'blockquote', 'figcaption', 'caption', 'label',
})
# Tags whose text content should never be grammar-corrected.
_HTML_SKIP_TAGS = frozenset({'script', 'style', 'code', 'pre', 'kbd', 'var', 'samp'})


class GrammarCorrectionProcessor:
    """Fixed grammar correction processor with singleton pattern"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if GrammarCorrectionProcessor._initialized:
            return

        self.model = None
        self.tokenizer = None
        self.ocr_reader = None
        self._ocr_lock = threading.Lock()
        self._ocr_last_used = 0.0
        self.spell_checker = None
        self._last_used = 0.0
        self._in_use_count = 0
        self._lock = threading.Lock()
        self._initialize_spell_checker()

        GrammarCorrectionProcessor._initialized = True
        logger.info("GrammarCorrectionProcessor initialized (singleton, model and OCR load on first use)")

    def ensure_loaded(self):
        with self._lock:
            self._in_use_count += 1
            if self.model is None or self.tokenizer is None:
                self._load_model()
            self._last_used = time.time()

    def release(self):
        with self._lock:
            if self._in_use_count > 0:
                self._in_use_count -= 1
            self._last_used = time.time()

    def unload_model_if_idle(self):
        timeout = getattr(settings, "MODEL_IDLE_UNLOAD_SECONDS", 300) or 300
        with self._lock:
            if self._in_use_count > 0:
                return
            if self.model is None:
                return
            if time.time() - self._last_used < timeout:
                return
            self.model = None
            self.tokenizer = None
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("Model unloaded (idle %ds); RAM freed", timeout)

    def unload_ocr_if_idle(self):
        timeout = getattr(settings, "OCR_IDLE_UNLOAD_SECONDS", 180) or 180
        with self._ocr_lock:
            if self.ocr_reader is None:
                return
            if time.time() - self._ocr_last_used < timeout:
                return
            self.ocr_reader = None
        gc.collect()
        logger.info("OCR unloaded (idle %ds); RAM freed", timeout)

    def _load_model(self):
        """Load model with ultimate robust error handling"""
        try:
            model_path = settings.MODEL_PATH
            if not os.path.isabs(model_path):
                if not os.path.exists(model_path):
                    app_dir = os.path.dirname(os.path.abspath(__file__))
                    project_root = os.path.dirname(app_dir)
                    resolved = os.path.join(project_root, model_path.replace("\\", "/").lstrip("./"))
                    if os.path.exists(resolved):
                        model_path = os.path.abspath(resolved)
                        logger.info("Resolved MODEL_PATH to %s", model_path)
                else:
                    model_path = os.path.abspath(model_path)
            model_id = (getattr(settings, "MODEL_ID", "") or "").strip() or None

            if not os.path.exists(model_path) and model_id:
                logger.info(" Model path not found: %s, downloading from Hugging Face: %s", model_path, model_id)
                try:
                    from huggingface_hub import snapshot_download
                    hf_token = getattr(settings, 'HF_TOKEN', None)
                    os.makedirs(model_path, exist_ok=True)
                    snapshot_download(repo_id=model_id, local_dir=model_path, token=hf_token if hf_token else None)
                    logger.info(" Model downloaded successfully from Hugging Face")
                except Exception as download_error:
                    logger.error(" Failed to download model from Hugging Face: %s", download_error)
                    self.model = None
                    self.tokenizer = None
                    return
            
            if os.path.exists(model_path) or model_id:
                # Determine what to pass to load_robust_model
                # If model_path exists, use it; otherwise use MODEL_ID (HF repo ID)
                model_source = model_path if os.path.exists(model_path) else model_id
                logger.info(" Loading model from %s", model_source)

                # Get model info first (if model exists locally)
                if os.path.exists(model_path):
                    from app.robust_model_loader import get_model_info
                    model_info = get_model_info(model_path)
                    logger.info("Model info: %s", model_info)

                # Try to load with ultimate robust loader (can accept local path or HF repo ID)
                hf_token = getattr(settings, 'HF_TOKEN', None)
                self.model, self.tokenizer = load_robust_model(model_source, hf_token=hf_token)

                if self.model is None or self.tokenizer is None:
                    logger.warning("load_robust_model returned None; check model files at %s or MODEL_ID", model_source)

                if self.model is not None and self.tokenizer is not None:
                    logger.info("Model loaded successfully with ultimate robust loader")
                    self.model.eval()
                    self.model.to('cpu')
                    for param in self.model.parameters():
                        param.requires_grad = False

                    # INT8 dynamic quantization: ~1.5-2x faster on CPU, ~50% less RAM, minimal accuracy loss
                    try:
                        self.model = torch.quantization.quantize_dynamic(
                            self.model, {torch.nn.Linear}, dtype=torch.qint8
                        )
                        logger.info("Model quantized to INT8 (dynamic quantization applied)")
                    except Exception as q_err:
                        logger.warning("INT8 quantization skipped (using float32): %s", q_err)

                    if not getattr(settings, "SKIP_MODEL_TEST", False):
                        try:
                            test_result = test_model_inference(self.model, self.tokenizer, "This is a test.")
                            logger.info("Model test successful: '%s'", test_result)
                        except (RuntimeError, AttributeError) as test_e:
                            logger.warning("Model test failed but model loaded: %s", test_e)
                    gc.collect()
                else:
                    logger.warning(" Model loading failed, using fallback")
                    self.model = None
                    self.tokenizer = None
            else:
                logger.warning(" Model path not found: %s and MODEL_ID not set", model_path)
                self.model = None
                self.tokenizer = None
        except (OSError, RuntimeError, ImportError) as e:
            logger.error(" Error loading model: %s", e)
            self.model = None
            self.tokenizer = None

    def _ensure_ocr(self):
        """Lazy-load OCR reader on first image request to reduce peak memory."""
        with self._ocr_lock:
            if self.ocr_reader is not None:
                return
            try:
                import easyocr  # pylint: disable=import-outside-toplevel
                model_dir = getattr(settings, 'OCR_MODEL_DIR', '/app/.EasyOCR/model')
                os.makedirs(model_dir, exist_ok=True)
                os.environ['OMP_NUM_THREADS'] = '1'
                os.environ['MKL_NUM_THREADS'] = '1'
                self.ocr_reader = easyocr.Reader(
                    ['en'],
                    model_storage_directory=model_dir,
                    gpu=False,
                    verbose=False
                )
                self._ocr_last_used = time.time()
                logger.info("OCR initialized (lazy) with model directory: %s", model_dir)
            except (ImportError, OSError, RuntimeError) as e:
                logger.warning("OCR not available: %s", e)
                self.ocr_reader = None

    def _initialize_ocr(self):
        """Legacy entry point; use _ensure_ocr() for lazy init."""
        self._ensure_ocr()

    def _initialize_spell_checker(self):
        """Initialize spell checker for catching spelling errors"""
        try:
            from spellchecker import SpellChecker  # type: ignore[import-untyped]
            self.spell_checker = SpellChecker(language='en')
            logger.info("Spell checker initialized")
        except (ImportError, Exception) as e:
            logger.warning("Spell checker not available: %s", e)
            self.spell_checker = None

    def _preprocess_image_for_ocr(self, img_array: Any) -> Optional[Any]:
        """Preprocess image to improve OCR: grayscale, contrast (CLAHE), light denoise."""
        try:
            import numpy as np
            import cv2
            if img_array is None or (hasattr(img_array, 'shape') and len(img_array.shape) < 2):
                return None
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = np.asarray(img_array, dtype=np.uint8)
                if gray.ndim == 3:
                    gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            denoised = cv2.bilateralFilter(enhanced, 5, 50, 50)
            return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            logger.debug("OCR preprocessing skipped: %s", e)
            return None

    def is_ready(self) -> Dict[str, bool]:
        """Check readiness"""
        return {
            "model_loaded": self.model is not None,
            "ocr_available": self.ocr_reader is not None,
            "spell_checker_available": self.spell_checker is not None
        }

    def handle_input(self, input_source_path: str) -> Tuple[Optional[Any], str]:
        """
        Handle input file and determine its type.

        Args:
            input_source_path: Path to the input file

        Returns:
            Tuple of (content/path, input_type) or (None, error_type)
        """
        if not os.path.isfile(input_source_path):
            logger.error("File not found at %s", input_source_path)
            return None, 'file_not_found'

        file_extension = os.path.splitext(input_source_path)[1].lower()

        if file_extension in settings.ALLOWED_IMAGE_EXTENSIONS:
            return input_source_path, 'image'
        if file_extension in settings.ALLOWED_HTML_EXTENSIONS:
            # Try multiple encodings to handle different file formats
            encodings = ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'latin-1', 'cp1252']

            for encoding in encodings:
                try:
                    with open(input_source_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    logger.info("Successfully read HTML file with %s encoding", encoding)
                    return content, 'html'
                except UnicodeDecodeError:
                    continue
                except OSError as e:
                    logger.warning("Error reading HTML file with %s: %s", encoding, e)
                    continue

            # If all encodings fail, try reading as binary and decode with error handling
            try:
                with open(input_source_path, 'rb') as f:
                    raw_content = f.read()
                # Try to detect BOM and remove it
                if raw_content.startswith(b'\xff\xfe'):
                    content = raw_content[2:].decode('utf-16-le', errors='ignore')
                elif raw_content.startswith(b'\xfe\xff'):
                    content = raw_content[2:].decode('utf-16-be', errors='ignore')
                elif raw_content.startswith(b'\xef\xbb\xbf'):
                    content = raw_content[3:].decode('utf-8', errors='ignore')
                else:
                    content = raw_content.decode('utf-8', errors='ignore')
                logger.info("Successfully read HTML file with error handling")
                return content, 'html'
            except (OSError, IOError) as e:
                logger.error("Error reading HTML file with all methods: %s", e)
                return None, 'html_read_error'
        logger.error("Unsupported file type: %s", file_extension)
        return None, 'unknown_file_type'

    def extract_text(self, content: Any, input_type: str) -> Tuple[Any, Any]:
        """
        Extract text from image or HTML content.

        Args:
            content: Image path (str) or HTML content (str)
            input_type: Type of input ('image' or 'html')

        Returns:
            Tuple of (extracted_text, metadata)
        """
        if input_type == 'image':
            self._ensure_ocr()
            if not self.ocr_reader:
                logger.error("OCR reader not available")
                return [], []

            try:
                import numpy as np
                max_dimension = getattr(settings, "OCR_MAX_DIMENSION", 1024)

                img = Image.open(content)
                original_size = img.size
                if max(img.size) > max_dimension:
                    ratio = max_dimension / max(img.size)
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    logger.info("Resized image from %s to %s for memory efficiency", original_size, new_size)
                img_array = np.array(img)
                img.close()
                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)
                ocr_input = self._preprocess_image_for_ocr(img_array)
                if ocr_input is None:
                    ocr_input = img_array
                self._ocr_last_used = time.time()
                results = self.ocr_reader.readtext(ocr_input)
                self._ocr_last_used = time.time()
                extracted_texts = [item[1] for item in results]
                gc.collect()
                return extracted_texts, results
            except (OSError, ValueError, AttributeError) as e:
                logger.error("Error during OCR: %s", e)
                gc.collect()
                return [], []

        if input_type == 'html':
            # For HTML, we need to preserve the structure while extracting text for correction
            # Store the original HTML string to preserve formatting exactly
            original_html_string = content
            soup = BeautifulSoup(content, 'html.parser')

            # Extract ALL text content for grammar correction (not just specific elements)
            # This ensures we catch errors in any HTML element (td, th, etc.)
            # Use get_text() to extract all text, preserving structure for reconstruction
            extracted_text = soup.get_text(separator=' ', strip=False)
            
            extracted_text = re.sub(r'\s+', ' ', extracted_text)
            extracted_text = extracted_text.strip()

            # Return extracted text, soup object, and original HTML string for reconstruction
            return extracted_text, (soup, original_html_string)

        return None, None

    def _normalize_known_spellings(self, text: str) -> str:
        """Pre-pass: fix known misspellings so the model sees cleaner input. Uses pre-compiled patterns."""
        out = text
        for pattern, repl in _NORMALIZE_PATTERNS:
            out = pattern.sub(repl, out)
        return out

    def _fix_contractions(self, text: str) -> str:
        """Post-correction pass: restore contractions that spell checker or model may have left bare.
        Only applies patterns that are never standalone valid English words."""
        out = text
        for pattern, repl in _CONTRACTION_PATTERNS:
            out = pattern.sub(repl, out)
        return out

    def _fix_capitalization(self, text: str) -> str:
        """Fix capitalization errors the model may miss:
        1. Mid-word stray uppercase (OCR artefacts: 'heLLo' -> 'hello', 'tHe' -> 'the')
        2. Standalone 'i' -> 'I'
        3. First character of the entire text
        4. First letter after sentence-ending punctuation (. ! ?)

        Step 1 runs before steps 3-4 so that lowercased words are re-capitalised
        correctly at sentence boundaries.
        """
        if not text:
            return text

        # 1. Fix mid-word stray uppercase (e.g. "heLLo" -> "hello", "tHe" -> "the")
        text = _MID_WORD_CAPS_RE.sub(_lower_unless_acronym, text)

        # 2. Standalone pronoun 'i' -> 'I'
        text = _STANDALONE_I_RE.sub('I', text)

        # 3. Capitalize the very first letter if it is lowercase
        if text[0].islower():
            text = text[0].upper() + text[1:]

        # 4. Capitalize first letter after '. ' / '! ' / '? '
        text = _SENTENCE_START_RE.sub(lambda m: m.group(1) + ' ' + m.group(2).upper(), text)

        return text

    def _correct_html_blocks(self, soup, full_text: str) -> Tuple[str, List[Dict]]:
        """
        Correct each leaf block-level HTML element independently so the model always
        sees a coherent grammatical unit instead of a mixed-context text blob.

        Returns (reassembled_corrected_text, combined_corrections_list).
        Falls back to flat correction when no qualifying blocks are found.

        CPU cost: roughly equivalent to flat correction — same total token count,
        but split across N sequential calls (each one is shorter and therefore faster).
        Blocks that are too short (<10 chars) are skipped to avoid wasted inference.
        """
        # Collect leaf block elements: block-level tags that contain no nested block children
        candidate_blocks: List[Tuple[Any, str]] = []
        for el in soup.find_all(_HTML_BLOCK_TAGS):
            # Skip elements inside code/script/style
            if any(p.name in _HTML_SKIP_TAGS for p in el.parents):
                continue
            # Only process leaf blocks — skip if any child is itself a block-level element
            if el.find(_HTML_BLOCK_TAGS):
                continue
            text = el.get_text(strip=True)
            if len(text) < 40:
                continue
            candidate_blocks.append((el, text))

        if not candidate_blocks:
            # Fallback: correct the full extracted text as one unit
            corrected = self.correct_grammar(full_text)
            return corrected, self.identify_corrections(full_text, corrected)

        orig_parts: List[str] = []
        corr_parts: List[str] = []
        all_corrections: List[Dict] = []

        for _el, block_text in candidate_blocks:
            corrected_block = self.correct_grammar(block_text)
            orig_parts.append(block_text)
            corr_parts.append(corrected_block)
            all_corrections.extend(self.identify_corrections(block_text, corrected_block))

        return " ".join(corr_parts), all_corrections

    def correct_grammar(self, text: str) -> str:
        """Correct grammar with chunked processing and spell checking"""
        if not self.model or not self.tokenizer:
            logger.warning(
                "Model not available (model=%s, tokenizer=%s), using fallback. Check MODEL_PATH and MODEL_ID, or see startup logs for load errors.",
                self.model is not None,
                self.tokenizer is not None,
            )
            result = self._fallback_correction(text)
            result = self._fix_contractions(result)
            return self._fix_capitalization(result)

        try:
            text = text.strip()
            if not text:
                return text
            text = self._normalize_known_spellings(text)

            # Pre-spell-check: fix OCR character-level errors before the grammar model sees
            # them. Without this, garbled words like "cooiing" or "uindoivs" are simply
            # deleted by the T5 model rather than corrected.
            if self.spell_checker:
                text = self._apply_spell_checking(text)

            max_tokens_per_chunk = getattr(settings, "MODEL_CHUNK_MAX_TOKENS", 128)
            estimated_tokens = len(text.split()) * 1.33
            logger.info("correct_grammar: %d estimated tokens, input[:100]=%r", int(estimated_tokens), text[:100])

            if estimated_tokens <= max_tokens_per_chunk:
                model_corrected = self._correct_grammar_chunk(text)
            else:
                logger.info("Text is long (%d estimated tokens), processing in chunks", int(estimated_tokens))
                model_corrected = self._correct_grammar_chunked(text, max_tokens_per_chunk)

            logger.info("model output[:100]=%r", model_corrected[:100])

            if self.spell_checker:
                result = self._apply_spell_checking(model_corrected)
            else:
                result = model_corrected
            result = self._fix_contractions(result)
            final = self._fix_capitalization(result)
            if final.strip() == text.strip():
                logger.info("correct_grammar: model+pipeline made NO changes to text")
            else:
                logger.info("correct_grammar: changes detected, output[:100]=%r", final[:100])
            return final

        except Exception as e:
            logger.error("Error in correct_grammar: %s", e, exc_info=True)
            return self._fallback_correction(text)
    
    def _correct_grammar_chunk(self, text: str) -> str:
        """Correct grammar for a single chunk of text"""
        try:
            device = torch.device("cpu")
            max_len = getattr(settings, "MODEL_MAX_LENGTH", 256)
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            num_beams = getattr(settings, "MODEL_NUM_BEAMS", 2)
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_len,
                    num_beams=num_beams,
                    early_stopping=(num_beams > 1),
                    do_sample=False,
                    num_return_sequences=1,
                )

            # Decode the generated IDs to text
            corrected_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            # If result is empty, it's a model failure - use fallback
            if not corrected_text or corrected_text.strip() == "":
                logger.warning("Model returned empty result for chunk, using fallback")
                return self._fallback_correction(text)

            # Clean up the corrected text
            corrected_text = corrected_text.strip()
            return corrected_text

        except Exception as e:
            logger.error("Error correcting chunk: %s", e)
            return self._fallback_correction(text)

    def _correct_grammar_batched(self, chunks: List[str]) -> List[str]:
        """Process multiple chunks in one model forward pass. Memory use scales with batch size."""
        if not chunks or not self.model or not self.tokenizer:
            return [self._fallback_correction(c) for c in chunks] if chunks else []
        try:
            device = torch.device("cpu")
            max_len = getattr(settings, "MODEL_MAX_LENGTH", 128)
            num_beams = getattr(settings, "MODEL_NUM_BEAMS", 2)
            inputs = self.tokenizer(
                chunks,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_len,
                    num_beams=num_beams,
                    early_stopping=(num_beams > 1),
                    do_sample=False,
                    num_return_sequences=1,
                )
            corrected = []
            for i in range(generated_ids.size(0)):
                decoded = self.tokenizer.decode(generated_ids[i], skip_special_tokens=True).strip()
                if not decoded:
                    decoded = self._fallback_correction(chunks[i])
                corrected.append(decoded)
            del input_ids, attention_mask, generated_ids
            gc.collect()
            return corrected
        except Exception as e:
            logger.warning("Batch inference failed, falling back to sequential: %s", e)
            return [self._correct_grammar_chunk(c) for c in chunks]

    def _correct_grammar_chunked(self, text: str, max_tokens_per_chunk: int = 128) -> str:
        """Process long text in chunks and combine results"""
        # Split text into sentences for better chunking
        # This preserves sentence boundaries which helps the model
        sentences = re.split(r'([.!?]\s+)', text)
        
        # Recombine sentences with their punctuation
        sentence_pairs = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                sentence_pairs.append(sentences[i] + sentences[i + 1])
            else:
                sentence_pairs.append(sentences[i])
        if len(sentences) % 2 == 1:
            sentence_pairs.append(sentences[-1])
        
        # Group sentences into chunks that fit within token limit
        chunks = []
        current_chunk = ""
        
        for sentence in sentence_pairs:
            # Estimate tokens for current chunk + new sentence
            test_chunk = (current_chunk + " " + sentence).strip()
            estimated_tokens = len(test_chunk.split()) * 1.33
            
            if estimated_tokens <= max_tokens_per_chunk and current_chunk:
                # Add to current chunk
                current_chunk = test_chunk
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        batch_size = max(1, getattr(settings, "MODEL_BATCH_CHUNKS", 1))
        logger.info("Split text into %d chunks, processing in batches of %d", len(chunks), batch_size)

        corrected_chunks = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            if len(batch) == 1 and batch_size == 1:
                corrected_chunks.append(self._correct_grammar_chunk(batch[0]))
            else:
                corrected_chunks.extend(self._correct_grammar_batched(batch))

        corrected_text = " ".join(corrected_chunks)
        gc.collect()

        if corrected_text.strip() == text.strip():
            logger.info("No grammar errors found after chunked processing")
        else:
            logger.info("Grammar correction applied via chunked processing: '%s...' -> '%s...'", 
                       text[:50], corrected_text[:50])
        
        return corrected_text

    def _check_model_changes(self, original: str, corrected: str) -> bool:
        """Check if the model actually made changes"""
        return original.strip() != corrected.strip()
    
    def _apply_spell_checking(self, text: str) -> str:
        """Apply spell checking to catch errors the model missed. Batches unknown() for speed."""
        if not self.spell_checker:
            return text
        try:
            tokens = re.findall(r"(\b[\w']+\b|\W+)", text)
            word_pattern = re.compile(r"^[\w']+$")

            # Collect indices and normalised forms of candidate words in one pass
            candidate_indices: List[int] = []
            candidates: List[str] = []
            for i, token in enumerate(tokens):
                if not word_pattern.match(token):
                    continue
                normalised = token.lower().replace("'", "")
                if len(normalised) < 3:
                    continue
                candidate_indices.append(i)
                candidates.append(normalised)

            if not candidates:
                return text

            # Single batch call - much cheaper than N individual calls
            unknown_set = self.spell_checker.unknown(candidates)

            corrected = list(tokens)
            corrections_made = 0
            for idx, normalised in zip(candidate_indices, candidates):
                if normalised not in unknown_set:
                    continue
                suggestion = self.spell_checker.correction(normalised)
                if not suggestion or suggestion == normalised:
                    continue
                original_token = tokens[idx]
                if original_token.isupper():
                    suggestion = suggestion.upper()
                elif original_token[0].isupper():
                    suggestion = suggestion.capitalize()
                corrected[idx] = suggestion
                corrections_made += 1
                logger.info("Spell checker: '%s' -> '%s'", original_token, suggestion)

            if corrections_made > 0:
                logger.info("Spell checker applied %d corrections", corrections_made)
                return ''.join(corrected)
            return text
        except Exception as e:
            logger.error("Error in spell checking: %s", e, exc_info=True)
            return text

    def _fallback_correction(self, text: str) -> str:
        """Fallback corrections when model is unavailable. Uses pre-compiled patterns."""
        corrected_text = text
        corrections_made = 0
        for pattern, replacement in _FALLBACK_PATTERNS:
            new_text = pattern.sub(replacement, corrected_text)
            if new_text != corrected_text:
                corrections_made += 1
            corrected_text = new_text
        if corrections_made > 0:
            logger.info("Fallback correction applied %d fixes", corrections_made)
        return corrected_text

    def identify_corrections(self, original_text: str, corrected_text: str, context_words: int = 3) -> List[Dict[str, str]]:
        """
        Compares original and corrected text to identify changed words using sequence matching.
        Matches googlecolab.py exactly.
        """
        from difflib import Differ

        # Quick check: if texts are identical, no corrections needed
        if original_text.strip() == corrected_text.strip():
            logger.info("identify_corrections: texts are identical, no corrections")
            return []
        logger.info("identify_corrections: orig_len=%d corrected_len=%d", len(original_text), len(corrected_text))

        # Tokenize including punctuation as separate tokens (exactly like googlecolab.py)
        original_tokens_with_sep = re.findall(r'(\b\w+\b|\W+)', original_text)
        corrected_tokens_with_sep = re.findall(r'(\b\w+\b|\W+)', corrected_text)

        # Create lists of only words for diffing
        original_words = [token.lower() for token in original_tokens_with_sep if re.fullmatch(r'\b\w+\b', token)]
        corrected_words = [token.lower() for token in corrected_tokens_with_sep if re.fullmatch(r'\b\w+\b', token)]

        differ = Differ()
        # Diff based on words only for identifying changes
        diff = list(differ.compare(original_words, corrected_words))

        corrections = []
        original_buffer = []
        corrected_buffer = []

        # Keep track of the index in the original_words and corrected_words lists
        original_word_index = 0
        corrected_word_index = 0

        for item in diff:
            code = item[0]
            token = item[2:]  # This is a word from original_words or corrected_words

            if code == '?':
                # Skip difference markers
                continue

            if code == '-':
                original_buffer.append(token)
                original_word_index += 1
            elif code == '+':
                corrected_buffer.append(token)
                corrected_word_index += 1
            elif code == ' ':
                # If tokens are the same, process any buffered changes before this
                while original_buffer or corrected_buffer:
                    orig = original_buffer.pop(0) if original_buffer else ''
                    corr = corrected_buffer.pop(0) if corrected_buffer else ''

                    # Only add to corrections if there's a change or non-empty insertion/deletion
                    if orig != corr or (orig == '' and corr != '') or (orig != '' and corr == ''):
                        # Find the index of the original word in the original_words list
                        try:
                            if orig:
                                # Find the index of the *last* occurrence of the original word in the original_words list before the current index
                                orig_index_in_words = original_word_index - len(original_buffer) - 1 if original_buffer else original_word_index - 1
                                orig_index_in_words = max(0, orig_index_in_words)

                                # Get original context words
                                original_context_start = max(0, orig_index_in_words - context_words)
                                original_context_end = min(len(original_words), orig_index_in_words + len([orig]) + context_words)
                                original_context = " ".join(original_words[original_context_start:original_context_end])
                            else:
                                # For insertions, context is based on the corrected text position
                                corr_index_in_words = corrected_word_index - len(corrected_buffer) - 1 if corrected_buffer else corrected_word_index - 1
                                corr_index_in_words = max(0, corr_index_in_words)

                                corrected_context_start = max(0, corr_index_in_words - context_words)
                                corrected_context_end = min(len(corrected_words), corr_index_in_words + len([corr]) + context_words)
                                original_context = " ".join(corrected_words[corrected_context_start:corrected_context_end])
                        except (IndexError, ValueError) as e:
                            logger.error("Error getting original context for %s: %s", orig, e)
                            original_context = ""

                        # Get corrected context words
                        try:
                            if corr:
                                corr_index_in_words = corrected_word_index - len(corrected_buffer) - 1 if corrected_buffer else corrected_word_index - 1
                                corr_index_in_words = max(0, corr_index_in_words)

                                corrected_context_start = max(0, corr_index_in_words - context_words)
                                corrected_context_end = min(len(corrected_words), corr_index_in_words + len([corr]) + context_words)
                                corrected_context = " ".join(corrected_words[corrected_context_start:corrected_context_end])
                            else:
                                # For deletions, context is based on the original text position
                                orig_index_in_words = original_word_index - len(original_buffer) - 1 if original_buffer else original_word_index - 1
                                orig_index_in_words = max(0, orig_index_in_words)

                                original_context_start = max(0, orig_index_in_words - context_words)
                                original_context_end = min(len(original_words), orig_index_in_words + len([orig]) + context_words)
                                corrected_context = " ".join(original_words[original_context_start:original_context_end])
                        except (IndexError, ValueError) as e:
                            logger.error("Error getting corrected context for %s: %s", corr, e)
                            corrected_context = ""

                        if (orig.strip() != corr.strip() and
                                (orig.strip() != '' or corr.strip() != '')):
                            corrections.append({
                                'original_word': orig.strip(),
                                'corrected_word': corr.strip(),
                                'original_context': original_context,
                                'corrected_context': corrected_context
                            })

                # Move indices forward for the matched token
                original_word_index += 1
                corrected_word_index += 1
                # Reset buffers
                original_buffer = []
                corrected_buffer = []

        # Process any remaining buffered changes at the end
        while original_buffer or corrected_buffer:
            orig = original_buffer.pop(0) if original_buffer else ''
            corr = corrected_buffer.pop(0) if corrected_buffer else ''

            if orig != corr or (orig == '' and corr != '') or (orig != '' and corr == ''):
                try:
                    if orig:
                        orig_index_in_words = original_word_index - len(original_buffer) - 1 if original_buffer else original_word_index - 1
                        orig_index_in_words = max(0, orig_index_in_words)

                        original_context_start = max(0, orig_index_in_words - context_words)
                        original_context_end = min(len(original_words), orig_index_in_words + len([orig]) + context_words)
                        original_context = " ".join(original_words[original_context_start:original_context_end])
                    else:
                        corr_index_in_words = corrected_word_index - len(corrected_buffer) - 1 if corrected_buffer else corrected_word_index - 1
                        corr_index_in_words = max(0, corr_index_in_words)

                        corrected_context_start = max(0, corr_index_in_words - context_words)
                        corrected_context_end = min(len(corrected_words), corr_index_in_words + len([corr]) + context_words)
                        original_context = " ".join(corrected_words[corrected_context_start:corrected_context_end])
                except (IndexError, ValueError) as e:
                    logger.error("Error getting original context for %s at end: %s", orig, e)
                    original_context = ""

                try:
                    if corr:
                        corr_index_in_words = corrected_word_index - len(corrected_buffer) - 1 if corrected_buffer else corrected_word_index - 1
                        corr_index_in_words = max(0, corr_index_in_words)

                        corrected_context_start = max(0, corr_index_in_words - context_words)
                        corrected_context_end = min(len(corrected_words), corr_index_in_words + len([corr]) + context_words)
                        corrected_context = " ".join(corrected_words[corrected_context_start:corrected_context_end])
                    else:
                        orig_index_in_words = original_word_index - len(original_buffer) - 1 if original_buffer else original_word_index - 1
                        orig_index_in_words = max(0, orig_index_in_words)

                        original_context_start = max(0, orig_index_in_words - context_words)
                        original_context_end = min(len(original_words), orig_index_in_words + len([orig]) + context_words)
                        corrected_context = " ".join(original_words[original_context_start:original_context_end])
                except (IndexError, ValueError) as e:
                    logger.error("Error getting corrected context for %s at end: %s", corr, e)
                    corrected_context = ""

                if (orig.strip() != corr.strip() and
                        (orig.strip() != '' or corr.strip() != '')):
                    corrections.append({
                        'original_word': orig.strip(),
                        'corrected_word': corr.strip(),
                        'original_context': original_context,
                        'corrected_context': corrected_context
                    })

        # Build a set of original words that were flagged as unknown by the spell checker.
        # Corrections where the original word was already correct in the dictionary are
        # likely model rephrasings (false positives) and are excluded.
        original_words_in_corrections = {
            c['original_word'] for c in corrections if c['original_word'].strip()
        }
        if self.spell_checker and original_words_in_corrections:
            unknown_originals = self.spell_checker.unknown(list(original_words_in_corrections))
        else:
            unknown_originals = original_words_in_corrections

        cleaned_corrections = []
        for corr_dict in corrections:
            orig_word = corr_dict['original_word']
            corr_word = corr_dict['corrected_word']
            if orig_word == corr_word:
                continue
            if not orig_word.strip() and not corr_word.strip():
                continue

            # The grammar model sometimes deletes OCR-garbled words instead of correcting them,
            # producing corrected_word="". In that case, ask the spell checker what the word
            # should be and use that as the corrected word.
            if orig_word.strip() and not corr_word.strip():
                if self.spell_checker:
                    suggestion = self.spell_checker.correction(orig_word)
                    if suggestion and suggestion.lower() != orig_word.lower():
                        corr_dict = dict(corr_dict)
                        corr_dict['corrected_word'] = suggestion
                        cleaned_corrections.append(corr_dict)
                # Do not report a pure deletion with no spell-checker alternative
                continue

            # For substitutions, only report if the original word was not in the dictionary
            # (i.e., it was genuinely misspelled), to avoid reporting model rephrasings.
            if orig_word.strip() and orig_word not in unknown_originals:
                continue

            cleaned_corrections.append(corr_dict)

        cleaned_corrections = [
            c for c in cleaned_corrections
            if c.get('original_word', '').strip() and c.get('corrected_word', '').strip()
        ]
        logger.info("Filtered corrections: %d -> %d meaningful corrections", len(corrections), len(cleaned_corrections))

        return cleaned_corrections

    def reconstruct_with_highlighting(self, original_content: Any, input_type: str, corrected_text: str, corrections: List[Dict], original_ocr_results: Optional[List] = None) -> Optional[Any]:
        """
        Reconstructs the original content with highlighted corrections at the word level.
        Uses regex-based word matching for better accuracy (from googlecolab.py).
        """
        # Explicitly handle the case where no corrections are found
        if not corrections and input_type == 'image':
            logger.info("No corrections identified for image. Returning original image.")
            try:
                return Image.open(original_content).convert("RGB")
            except (OSError, IOError) as e:
                logger.error("Error loading original image for return: %s", e)
                return None
        if not corrections and input_type == 'html':
            logger.info("No corrections identified for %s. Returning original content.", input_type)
            if isinstance(original_content, tuple) and len(original_content) == 2:
                return original_content[1]
            return str(original_content) if original_content else None

        # Proceed with highlighting only if corrections exist
        if input_type == 'image':
            if original_ocr_results is None:
                logger.error("Error: original_ocr_results is required for image input.")
                return None

            try:
                # Load the original image using the path
                # Memory optimization: Load image efficiently
                img = Image.open(original_content).convert("RGB")
                original_size = img.size  # Store original size before resizing
                
                # Resize if too large to prevent memory issues
                max_dimension = 2048
                if max(img.size) > max_dimension:
                    ratio = max_dimension / max(img.size)
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    logger.info("Resized image for reconstruction from %s to %s", original_size, new_size)
                
                draw = ImageDraw.Draw(img)

                # Create a set of original words that were corrected for quick lookup
                original_corrected_words_set = {
                    corr_dict['original_word'].lower()
                    for corr_dict in corrections
                    if corr_dict['original_word'] != corr_dict['corrected_word']
                }

                confidence_threshold = getattr(
                    settings, "OCR_CONFIDENCE_THRESHOLD", 0.5
                )

                for (bbox, text, confidence) in original_ocr_results:
                    if confidence >= confidence_threshold:
                        # Get the bounding box coordinates as integers
                        x_coords = [int(p[0]) for p in bbox]
                        y_coords = [int(p[1]) for p in bbox]
                        x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)

                        # Attempt word-level highlighting within the bounding box
                        block_text = text  # Use the original text from OCR
                        block_text_lower = block_text.lower()

                        # Iterate through the original words that were corrected
                        for original_word_lower in original_corrected_words_set:
                            # Find all occurrences of the original word within the block text
                            # Use regex to find whole words
                            for match in re.finditer(r'\b' + re.escape(original_word_lower) + r'\b', block_text_lower):
                                start_index = match.start()
                                word_length = len(match.group(0))

                                # Basic approximation for word position within the block
                                block_width = x2 - x1
                                char_width_approx = block_width / len(block_text) if len(block_text) > 0 else 0

                                word_x1 = x1 + (start_index * char_width_approx)
                                word_y1 = y1
                                word_x2 = word_x1 + (word_length * char_width_approx)
                                word_y2 = y2

                                # Draw a highlight (red rectangle border) around the approximate word bounding box
                                draw.rectangle([(word_x1, word_y1), (word_x2, word_y2)], outline='red', width=2)

                gc.collect()
                return img

            except (OSError, IOError, ValueError) as e:
                logger.error("Error processing image for highlighting: %s", e)
                gc.collect()
                return None

        if input_type == 'html':
            try:
                # original_content is a tuple of (soup, original_html_string) from extract_text
                if isinstance(original_content, tuple) and len(original_content) == 2:
                    soup, html_string = original_content
                else:
                    # Fallback for old format
                    if hasattr(original_content, 'find_all'):
                        soup = original_content
                        html_string = str(soup)
                    else:
                        html_string = str(original_content)
                        soup = BeautifulSoup(html_string, 'html.parser')

                # Build a mapping of original words to their corrected versions
                word_corrections = {}
                for corr_dict in corrections:
                    orig_word = corr_dict.get('original_word', '')
                    corr_word = corr_dict.get('corrected_word', '')
                    if orig_word and orig_word != corr_word:
                        word_corrections[orig_word.lower()] = {
                            'original': orig_word,
                            'corrected': corr_word
                        }
                
                logger.info("HTML reconstruction: Processing %d corrections: %s", 
                           len(word_corrections), 
                           list(word_corrections.keys()))

                if not word_corrections:
                    # No corrections, return original HTML string
                    logger.warning("HTML reconstruction: No corrections to apply")
                    return html_string

                # DOM-based replacement: Traverse text nodes and insert real <u> Tag nodes
                # This approach is robust and handles all edge cases correctly
                from bs4 import NavigableString, Tag, Comment
                
                # Get all text nodes, excluding script, style, and comments
                text_nodes = []
                for element in soup.descendants:
                    if isinstance(element, NavigableString):
                        parent = element.parent
                        if parent:
                            parent_name = parent.name.lower() if parent.name else None
                            # Skip script, style, and comments
                            if parent_name not in ['script', 'style']:
                                # Check if parent is a comment or if element is inside a comment
                                is_comment = isinstance(parent, Comment)
                                # Also check if the string itself is a comment
                                if not is_comment and not isinstance(element, Comment):
                                    text_nodes.append(element)
                
                # Process each text node
                for text_node in text_nodes:
                    original_text = str(text_node)
                    if not original_text.strip():
                        continue
                    
                    # Find all words that need to be wrapped (with their positions)
                    # Use a set to track positions to avoid overlapping matches
                    word_matches = []
                    used_positions = set()
                    
                    for orig_lower, corr_data in word_corrections.items():
                        original_word = corr_data.get('original', '')
                        if not original_word:
                            continue
                        
                        # Escape special regex characters in the word
                        escaped_word = re.escape(original_word)
                        
                        # Use simple word boundary pattern - \b works well for most cases
                        # This is more reliable than the complex negative lookahead/lookbehind
                        word_pattern = r'\b' + escaped_word + r'\b'
                        
                        # Find all matches in this text node (case-insensitive)
                        for match in re.finditer(word_pattern, original_text, re.IGNORECASE | re.UNICODE):
                            start = match.start()
                            end = match.end()
                            
                            # Skip if this position range overlaps with a previous match
                            if any(start < prev_end and end > prev_start 
                                   for prev_start, prev_end, _ in word_matches):
                                continue
                            
                            # Get the actual word at this position (preserve original case)
                            actual_word = original_text[start:end]
                            
                            # Verify it's the same word (case-insensitive comparison)
                            if actual_word.lower() != orig_lower:
                                continue
                            
                            # Additional safety: verify it's not part of a larger word
                            # Check character before (if exists)
                            if start > 0:
                                char_before = original_text[start - 1]
                                # If it's a word character, skip (part of larger word)
                                if char_before.isalnum() or char_before == '_':
                                    continue
                            
                            # Check character after (if exists)
                            if end < len(original_text):
                                char_after = original_text[end]
                                # If it's a word character, skip (part of larger word)
                                if char_after.isalnum() or char_after == '_':
                                    continue
                            
                            word_matches.append((start, end, actual_word))
                            used_positions.add((start, end))
                    
                    if not word_matches:
                        continue
                    
                    # Sort matches by position (ascending) and remove any overlapping ones
                    word_matches.sort(key=lambda x: x[0])
                    
                    # Remove overlapping matches (keep first occurrence)
                    non_overlapping = []
                    for start, end, word in word_matches:
                        if not any(start < prev_end and end > prev_start 
                                  for prev_start, prev_end, _ in non_overlapping):
                            non_overlapping.append((start, end, word))
                    word_matches = non_overlapping
                    
                    # Build new content: split text and insert <u> tags
                    parent = text_node.parent
                    if not parent:
                        continue
                    
                    new_elements = []
                    last_pos = 0
                    
                    for start, end, word in word_matches:
                        # Add text before this match
                        if start > last_pos:
                            before_text = original_text[last_pos:start]
                            if before_text:
                                new_elements.append(NavigableString(before_text))
                        
                        # Create <u> tag with the word
                        u_tag = soup.new_tag('u')
                        u_tag.string = word
                        new_elements.append(u_tag)
                        
                        last_pos = end
                    
                    # Add remaining text after last match
                    if last_pos < len(original_text):
                        after_text = original_text[last_pos:]
                        if after_text:
                            new_elements.append(NavigableString(after_text))
                    
                    # Replace the original text node with new elements
                    if new_elements:
                        # Replace with first element
                        text_node.replace_with(new_elements[0])
                        # Insert remaining elements after the first
                        current = new_elements[0]
                        for elem in new_elements[1:]:
                            current.insert_after(elem)
                            current = elem
                
                # Convert soup back to string, preserving formatting
                # BeautifulSoup's str() preserves structure but may normalize some whitespace
                # For maximum preservation, we could use prettify with formatter=None,
                # but str() is sufficient and faster
                html_output = str(soup)
                
                # Ensure no escaped <u> tags (shouldn't happen with DOM-based approach, but verify)
                if '&lt;u&gt;' in html_output or '&lt;/u&gt;' in html_output:
                    logger.warning("Found escaped <u> tags in output, replacing")
                    html_output = html_output.replace('&lt;u&gt;', '<u>').replace('&lt;/u&gt;', '</u>')
                
                return html_output
                
            except (ValueError, AttributeError, re.error) as e:
                logger.error("HTML processing error: %s", e, exc_info=True)
                # Return original HTML string on error
                if isinstance(original_content, tuple) and len(original_content) == 2:
                    return original_content[1]
                return str(original_content) if original_content else None

        return None

    def generate_output(self, reconstructed_content: Any, input_type: str, corrections: List[Dict], output_dir: str = "/tmp") -> Tuple[Optional[str], str]:
        """
        Generate output - returns base64 for images, HTML string for HTML.

        Args:
            reconstructed_content: Processed content (Image or HTML)
            input_type: Type of input ('image' or 'html')
            corrections: List of correction dictionaries
            output_dir: Output directory (unused, kept for compatibility)

        Returns:
            Tuple of (content_output, json_output_string)
        """
        content_output = None

        if input_type == 'image':
            if isinstance(reconstructed_content, Image.Image):
                try:
                    # Memory optimization: Use JPEG instead of PNG for smaller size
                    # Convert image to base64 instead of saving to disk
                    buffered = BytesIO()
                    # Use JPEG with quality 85 for smaller file size and less memory
                    if reconstructed_content.mode == 'RGBA':
                        reconstructed_content = reconstructed_content.convert('RGB')
                    reconstructed_content.save(buffered, format="JPEG", quality=85, optimize=True)
                    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    content_output = f"data:image/jpeg;base64,{img_base64}"
                    # Close image and clear buffer to free memory
                    reconstructed_content.close()
                    buffered.close()
                    logger.info("Image converted to base64 successfully")
                except (OSError, IOError) as e:
                    logger.error("Error converting image to base64: %s", e)
                    content_output = "Error converting image to base64"

        elif input_type == 'html':
            # reconstructed_content is already a properly formatted HTML string
            # with <u> tags and all original formatting preserved
            if isinstance(reconstructed_content, str):
                content_output = reconstructed_content
            elif hasattr(reconstructed_content, 'prettify'):
                # Fallback: convert soup to string (shouldn't happen with new logic)
                content_output = str(reconstructed_content)
            else:
                content_output = str(reconstructed_content) if reconstructed_content else None

        try:
            json_output_string = json.dumps(corrections, indent=4)
        except (TypeError, ValueError) as e:
            logger.error("Error generating JSON: %s", e)
            json_output_string = "[]"

        return content_output, json_output_string

    def process_input(self, input_source_path: str, output_dir: str = "/tmp") -> Dict[str, Any]:
        """
        Process input end-to-end.

        Args:
            input_source_path: Path to input file
            output_dir: Output directory (unused, kept for compatibility)

        Returns:
            Dictionary with processing results
        """
        start_time = time.time()

        try:
            # 1. Handle input
            original_content, input_type = self.handle_input(input_source_path)

            if original_content is None:
                return {
                    "success": False,
                    "error": f"Failed to handle input: {input_type}",
                    "input_type": input_type
                }

            # 2. Extract text
            if input_type == 'image':
                extracted_texts, original_ocr_results = self.extract_text(original_content, input_type)
                text_to_correct = " ".join(extracted_texts) if extracted_texts else ""
                original_content_for_reconstruct = original_content
            elif input_type == 'html':
                extracted_text, soup_and_html = self.extract_text(original_content, input_type)
                text_to_correct = extracted_text if extracted_text else ""
                original_ocr_results = None
                # Pass both soup object and original HTML string for reconstruction
                original_content_for_reconstruct = soup_and_html
            else:
                return {
                    "success": False,
                    "error": "Unsupported input type",
                    "input_type": input_type
                }

            if not text_to_correct:
                return {
                    "success": True,
                    "input_type": input_type,
                    "original_text": "",
                    "corrected_text": "",
                    "corrections": [],
                    "corrections_count": 0,
                    "output_file": None,
                    "processing_time_seconds": time.time() - start_time
                }

            # 3. Correct grammar + 4. Identify corrections
            if input_type == 'html':
                # Per-block correction: each paragraph/cell/heading is corrected
                # independently so the model sees coherent grammatical units.
                soup_obj, _ = original_content_for_reconstruct
                corrected_text, corrections = self._correct_html_blocks(soup_obj, text_to_correct)
            else:
                corrected_text = self.correct_grammar(text_to_correct)
                original_text_for_comparison = (
                    " ".join(extracted_texts) if input_type == 'image' else text_to_correct
                )
                corrections = self.identify_corrections(original_text_for_comparison, corrected_text)

            # 5. Reconstruct with highlighting
            reconstructed_content = self.reconstruct_with_highlighting(
                original_content_for_reconstruct,
                input_type,
                corrected_text,
                corrections,
                original_ocr_results=original_ocr_results if input_type == 'image' else None
            )

            # 6. Generate output
            content_output, json_output = self.generate_output(
                reconstructed_content,
                input_type,
                corrections,
                output_dir=output_dir
            )

            processing_time = time.time() - start_time
            gc.collect()

            return {
                "success": True,
                "input_type": input_type,
                "original_text": text_to_correct,
                "corrected_text": corrected_text,
                "corrections": corrections,
                "corrections_count": len(corrections),
                "output_content": content_output,  # Contains base64 image or HTML string
                "processing_time_seconds": round(processing_time, 2)
            }

        except (OSError, RuntimeError, ValueError) as e:
            logger.error("Error in process_input: %s", e, exc_info=True)
            gc.collect()
            return {
                "success": False,
                "error": str(e),
                "input_type": "unknown"
            }


# Global processor instance
_processor = None

def get_processor() -> GrammarCorrectionProcessor:
    """Get or create global processor instance"""
    global _processor
    if _processor is None:
        _processor = GrammarCorrectionProcessor()
    return _processor
