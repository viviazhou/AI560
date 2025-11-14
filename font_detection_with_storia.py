"""
font_detection_with_storia.py - Font Detection Tool with Storia.ai

Uses Storia font-classify-onnx model from Hugging Face:
https://huggingface.co/storia/font-classify-onnx

This is a SEPARATE project from Tesserae.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from collections import Counter
from typing import List, Dict, Tuple, Optional
import torch
from pathlib import Path


class StoriaFontDetector:
    """
    Font detection using Storia's font-classify-onnx model.
    """
    
    def __init__(self):
        """Initialize the Storia font detector."""
        self.model = None
        self.model_config = None
        self.fonts_mapping = None
        self.input_size = (320, 320)
        self.google_fonts_database = None
        
        print("🔧 Initializing Storia Font Detector...")
        self._load_storia_model()
        self._load_google_fonts_database()
    
    def _load_storia_model(self):
        """
        Load Storia font classification model from Hugging Face.
        Uses ONNX Runtime for inference.
        """
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            import yaml
            
            print("📦 Loading Storia font-classify-onnx model...")
            
            # Download model files
            model_path = hf_hub_download(
                repo_id="storia/font-classify-onnx",
                filename="model.onnx"
            )
            
            config_path = hf_hub_download(
                repo_id="storia/font-classify-onnx",
                filename="model_config.yaml"
            )
            
            fonts_mapping_path = hf_hub_download(
                repo_id="storia/font-classify-onnx",
                filename="fonts_mapping.yaml"
            )
            
            # Load ONNX model
            self.model = ort.InferenceSession(model_path)
            
            # Load configs
            with open(config_path, 'r') as f:
                self.model_config = yaml.safe_load(f)
            
            with open(fonts_mapping_path, 'r') as f:
                fonts_mapping_dict = yaml.safe_load(f)
            
            # Convert to index-based list (order matches model output)
            self.fonts_mapping = list(fonts_mapping_dict.keys())
            
            # Store input dimensions
            self.input_size = (320, 320)  # Storia expects 320x320 images
            
            print(f"✅ Model loaded with {len(self.fonts_mapping)} fonts")
            
        except Exception as e:
            print(f"❌ Error loading Storia model: {e}")
            print("   Make sure onnxruntime and pyyaml are installed")
            self.model = None
            self.fonts_mapping = None
    
    def _load_google_fonts_database(self):
        """
        Load Google Fonts metadata for matching.
        """
        # Expanded Google Fonts database organized by characteristics
        self.google_fonts_database = {
            'serif': {
                'elegant': ['Playfair Display', 'Cormorant Garamond', 'Crimson Text'],
                'traditional': ['Merriweather', 'Lora', 'PT Serif', 'Source Serif Pro'],
                'modern': ['IBM Plex Serif', 'Bitter', 'Arvo'],
                'display': ['Abril Fatface', 'Yeseva One', 'Cinzel']
            },
            'sans-serif': {
                'geometric': ['Montserrat', 'Poppins', 'Raleway', 'Quicksand'],
                'humanist': ['Open Sans', 'Lato', 'Source Sans Pro', 'Nunito'],
                'grotesque': ['Roboto', 'Inter', 'Work Sans', 'DM Sans'],
                'modern': ['Lexend', 'Manrope', 'Plus Jakarta Sans'],
                'condensed': ['Oswald', 'Bebas Neue', 'Barlow Condensed']
            },
            'monospace': {
                'coding': ['Roboto Mono', 'Fira Code', 'Source Code Pro', 'JetBrains Mono'],
                'typewriter': ['Courier Prime', 'IBM Plex Mono', 'Space Mono']
            },
            'display': {
                'bold': ['Anton', 'Righteous', 'Permanent Marker'],
                'decorative': ['Pacifico', 'Dancing Script', 'Satisfy'],
                'modern': ['Archivo Black', 'Alfa Slab One']
            }
        }
    
    def detect_font(self, image_path: str) -> Dict:
        """
        Main font detection function using Storia model.
        
        Args:
            image_path: Path to image containing text
            
        Returns:
            dict: Detection results with Storia predictions and Google Fonts matches
        """
        print(f"\n🔍 Analyzing image: {image_path}")
        
        if self.model is None:
            return {
                'success': False,
                'error': 'Storia model not loaded',
                'suggestion': 'Install transformers: pip install transformers'
            }
        
        # Load image
        img = Image.open(image_path)
        
        # Step 1: Detect and extract text regions
        print("  📍 Detecting text regions...")
        text_regions = self._detect_text_regions_opencv(img)
        
        if not text_regions:
            return {
                'success': False,
                'error': 'No clear text regions detected',
                'suggestion': 'Try an image with larger, clearer text'
            }
        
        print(f"  ✅ Found {len(text_regions)} text region(s)")
        
        # Step 2: Run Storia model on each text region
        print("  🤖 Running Storia font classification...")
        storia_predictions = []
        
        for i, region in enumerate(text_regions[:3]):  # Analyze top 3 regions
            region_img = self._extract_region(img, region)
            prediction = self._classify_with_storia(region_img)
            
            if prediction:
                storia_predictions.append({
                    'region_index': i,
                    'prediction': prediction,
                    'region_info': region
                })
        
        if not storia_predictions:
            return {
                'success': False,
                'error': 'Storia classification failed',
                'suggestion': 'Try a different image with clearer text'
            }
        
        # Step 3: Map Storia predictions to Google Fonts
        print("  🔎 Matching to Google Fonts...")
        google_matches = self._map_to_google_fonts(storia_predictions)
        
        # Step 4: Compile results
        results = {
            'success': True,
            'text_regions_detected': len(text_regions),
            'storia_predictions': storia_predictions,
            'google_fonts_suggestions': google_matches,
            'confidence': self._calculate_overall_confidence(storia_predictions)
        }
        
        return results
    
    def _detect_text_regions_opencv(self, img: Image.Image) -> List[Dict]:
        """
        Detect text regions using OpenCV.
        """
        # Convert PIL to OpenCV
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Use MSER for text detection
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        
        text_regions = []
        height, width = img_cv.shape[:2]
        
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            
            # Filter for text-like regions
            aspect_ratio = w / float(h) if h > 0 else 0
            area = w * h
            
            if (0.1 < aspect_ratio < 20 and
                100 < area < (width * height * 0.5) and
                h > 15 and w > 15):  # Minimum size for Storia
                
                text_regions.append({
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'area': area
                })
        
        # Sort by area (largest first) and return top regions
        text_regions.sort(key=lambda r: r['area'], reverse=True)
        return text_regions[:5]
    
    def _extract_region(self, img: Image.Image, region: Dict) -> Image.Image:
        """
        Extract and prepare a text region for classification.
        """
        x, y, w, h = region['x'], region['y'], region['width'], region['height']
        
        # Add padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.width - x, w + 2 * padding)
        h = min(img.height - y, h + 2 * padding)
        
        # Crop region
        region_img = img.crop((x, y, x + w, y + h))
        
        # Resize if too small (Storia works better with larger images)
        min_size = 224
        if region_img.width < min_size or region_img.height < min_size:
            scale = max(min_size / region_img.width, min_size / region_img.height)
            new_size = (int(region_img.width * scale), int(region_img.height * scale))
            region_img = region_img.resize(new_size, Image.Resampling.LANCZOS)
        
        return region_img
    
    def _classify_with_storia(self, img: Image.Image) -> Optional[Dict]:
        """
        Classify font using Storia ONNX model.
        """
        try:
            # Resize to Storia's expected input size (320x320)
            img_resized = img.resize(self.input_size, Image.Resampling.LANCZOS)
            
            # Convert to RGB if needed
            if img_resized.mode != 'RGB':
                img_resized = img_resized.convert('RGB')
            
            # Convert to numpy array and normalize
            img_array = np.array(img_resized).astype(np.float32)
            
            # Normalize to [0, 1] range
            img_array = img_array / 255.0
            
            # Transpose to CHW format (channels, height, width)
            img_array = np.transpose(img_array, (2, 0, 1))
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            # Run inference
            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name
            
            outputs = self.model.run([output_name], {input_name: img_array})
            logits = outputs[0][0]  # Remove batch dimension
            
            # Apply softmax to get probabilities
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / np.sum(exp_logits)
            
            # Get top 5 predictions
            top_k = 5
            top_indices = np.argsort(probabilities)[-top_k:][::-1]
            
            predictions = []
            for idx in top_indices:
                # Get font name from mapping
                # Get font name from list (using index)
                if 0 <= idx < len(self.fonts_mapping):
                    font_name = self.fonts_mapping[idx]
                else:
                    font_name = f"Unknown_Font_{idx}"
                
                predictions.append({
                    'font_id': int(idx),
                    'font_name': font_name,
                    'confidence': float(probabilities[idx])
                })
            
            return {
                'top_prediction': predictions[0],
                'all_predictions': predictions
            }
            
        except Exception as e:
            print(f"    ⚠️  Classification error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _map_to_google_fonts(self, storia_predictions: List[Dict]) -> List[Dict]:
        """
        Map Storia predictions to Google Fonts.
        
        Note: This is a heuristic mapping. You'll need to create a proper
        mapping between Storia's font IDs and Google Fonts based on characteristics.
        """
        google_matches = []
        
        # For now, we'll use heuristics based on the predictions
        # In production, you'd have a lookup table: Storia font ID -> Google Font
        
        for pred_data in storia_predictions:
            prediction = pred_data['prediction']
            top_pred = prediction['top_prediction']
            confidence = top_pred['confidence']
            
            # Heuristic: suggest fonts from different categories based on confidence
            if confidence > 0.7:
                # High confidence - suggest specific category
                category = self._determine_category(top_pred)
                fonts = self._get_fonts_from_category(category)
            else:
                # Lower confidence - suggest variety
                fonts = self._get_diverse_fonts()
            
            for font in fonts[:3]:  # Top 3 from this region
                google_matches.append({
                    'font_name': font['name'],
                    'category': font['category'],
                    'subcategory': font['subcategory'],
                    'confidence': confidence,
                    'source_region': pred_data['region_index'],
                    'google_fonts_url': f"https://fonts.google.com/specimen/{font['name'].replace(' ', '+')}",
                    'css_import': f"@import url('https://fonts.googleapis.com/css2?family={font['name'].replace(' ', '+')}');"
                })
        
        # Remove duplicates and sort by confidence
        seen = set()
        unique_matches = []
        for match in google_matches:
            if match['font_name'] not in seen:
                seen.add(match['font_name'])
                unique_matches.append(match)
        
        unique_matches.sort(key=lambda x: x['confidence'], reverse=True)
        
        return unique_matches[:10]  # Return top 10 unique matches
    
    def _determine_category(self, prediction: Dict) -> str:
        """
        Determine font category from Storia prediction.
        This is a placeholder - you'll need to map Storia's output properly.
        """
        # This is where you'd use Storia's actual category labels
        # For now, return a default
        return 'sans-serif.humanist'
    
    def _get_fonts_from_category(self, category: str) -> List[Dict]:
        """
        Get fonts from a specific category.
        """
        parts = category.split('.')
        main_cat = parts[0] if parts else 'sans-serif'
        sub_cat = parts[1] if len(parts) > 1 else 'humanist'
        
        fonts = []
        if main_cat in self.google_fonts_database:
            if sub_cat in self.google_fonts_database[main_cat]:
                for font_name in self.google_fonts_database[main_cat][sub_cat]:
                    fonts.append({
                        'name': font_name,
                        'category': main_cat,
                        'subcategory': sub_cat
                    })
        
        return fonts if fonts else self._get_diverse_fonts()
    
    def _get_diverse_fonts(self) -> List[Dict]:
        """
        Get a diverse set of popular fonts.
        """
        diverse = [
            {'name': 'Inter', 'category': 'sans-serif', 'subcategory': 'grotesque'},
            {'name': 'Playfair Display', 'category': 'serif', 'subcategory': 'elegant'},
            {'name': 'Montserrat', 'category': 'sans-serif', 'subcategory': 'geometric'},
            {'name': 'Merriweather', 'category': 'serif', 'subcategory': 'traditional'},
            {'name': 'Roboto', 'category': 'sans-serif', 'subcategory': 'grotesque'}
        ]
        return diverse
    
    def _calculate_overall_confidence(self, predictions: List[Dict]) -> float:
        """
        Calculate overall detection confidence.
        """
        if not predictions:
            return 0.0
        
        confidences = [p['prediction']['top_prediction']['confidence'] 
                      for p in predictions]
        return float(np.mean(confidences))


# Testing function
def test_storia_detector(image_path: str):
    """
    Test the Storia-based font detector.
    """
    detector = StoriaFontDetector()
    results = detector.detect_font(image_path)
    
    if results['success']:
        print("\n" + "="*60)
        print("STORIA FONT DETECTION RESULTS")
        print("="*60)
        
        print(f"\n📊 Analysis Summary:")
        print(f"  • Text regions found: {results['text_regions_detected']}")
        print(f"  • Overall confidence: {results['confidence']:.0%}")
        
        print(f"\n🤖 Storia Predictions:")
        for pred_data in results['storia_predictions']:
            pred = pred_data['prediction']['top_prediction']
            print(f"  Region {pred_data['region_index']}:")
            print(f"    • Font ID: {pred['font_id']}")
            print(f"    • Confidence: {pred['confidence']:.0%}")
        
        print(f"\n🎨 Google Fonts Suggestions:")
        for i, match in enumerate(results['google_fonts_suggestions'][:5], 1):
            print(f"\n  {i}. {match['font_name']}")
            print(f"     Category: {match['category']} / {match['subcategory']}")
            print(f"     Confidence: {match['confidence']:.0%}")
            print(f"     URL: {match['google_fonts_url']}")
        
        print("\n" + "="*60)
    else:
        print(f"\n❌ Detection failed: {results.get('error')}")
        print(f"   Suggestion: {results.get('suggestion')}")
    
    return results