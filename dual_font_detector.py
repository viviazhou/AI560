# dual_font_detector.py - DUAL MODEL FONT DETECTION

import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, AutoModelForImageClassification
import yaml
import numpy as np
from PIL import Image
import cv2
import torch

class DualModelFontDetector:
    
    def __init__(self):
        """Initialize both Storia and Gaborcselle models"""
        # Load Storia (comprehensive, 3,475 fonts)
        print("Loading Storia model...")
        model_path = hf_hub_download(repo_id="storia/font-classify-onnx", filename="model.onnx")
        config_path = hf_hub_download(repo_id="storia/font-classify-onnx", filename="model_config.yaml")
        fonts_mapping_path = hf_hub_download(repo_id="storia/font-classify-onnx", filename="fonts_mapping.yaml")
    
        self.storia_session = ort.InferenceSession(model_path)
        
        with open(config_path, 'r') as f:
            self.model_config = yaml.safe_load(f)
        with open(fonts_mapping_path, 'r') as f:
            self.fonts_mapping = yaml.safe_load(f)
        
        # CREATE INDEX-TO-FONT MAPPING
        self.index_to_font = {i: font_name for i, font_name in enumerate(self.fonts_mapping.keys())}
        
        # Load Gaborcselle (fast, 48 common fonts)
        print("Loading Gaborcselle model...")
        self.gabor_processor = AutoImageProcessor.from_pretrained("gaborcselle/font-identifier")
        self.gabor_model = AutoModelForImageClassification.from_pretrained("gaborcselle/font-identifier")
        self.gabor_labels = self.gabor_model.config.id2label
        
        print(f"✅ Dual model initialized: Storia ({len(self.fonts_mapping)} fonts) + Gaborcselle ({len(self.gabor_labels)} fonts)")
    
    def detect_with_gabor(self, image_path):
        """Fast detection with Gaborcselle (48 common fonts)"""
        image = Image.open(image_path).convert('RGB')
        inputs = self.gabor_processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.gabor_model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get top 5 predictions
        top5_prob, top5_idx = torch.topk(probabilities, 5)
        
        predictions = []
        for prob, idx in zip(top5_prob[0], top5_idx[0]):
            predictions.append({
                'font_name': self.gabor_labels[idx.item()],
                'confidence': prob.item()
            })
        
        return {
            'success': True,
            'top_prediction': predictions[0],
            'all_predictions': predictions
        }
    
    def detect_with_storia(self, image_path):
        """Comprehensive detection with Storia (3,475 fonts)"""
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        
        # Detect text regions
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        regions = self._detect_text_regions(gray)
        
        if not regions:
            regions = [(0, 0, image_array.shape[1], image_array.shape[0])]
        
        all_predictions = []
        for x, y, w, h in regions:
            region = image_array[y:y+h, x:x+w]
            region_pil = Image.fromarray(region)
            
            # Preprocess for Storia
            region_resized = region_pil.resize((320, 320))
            region_array = np.array(region_resized).astype(np.float32) / 255.0
            region_array = np.transpose(region_array, (2, 0, 1))
            region_array = np.expand_dims(region_array, axis=0)
            
            # Run Storia inference
            outputs = self.storia_session.run(None, {'input': region_array})
            logits = outputs[0][0]
            probabilities = self._softmax(logits)
            
            # Get top 5 predictions
            top5_indices = np.argsort(probabilities)[-5:][::-1]
            
            predictions = []
            for idx in top5_indices:
                predictions.append({
                    'font_id': int(idx),
                    'font_name': self.index_to_font.get(int(idx), f"Unknown_{idx}"),
                    'confidence': float(probabilities[idx])
                })
            
            all_predictions.append({
                'region': (x, y, w, h),
                'prediction': {
                    'top_prediction': predictions[0],
                    'all_predictions': predictions
                }
            })
        
        return {
            'success': True,
            'storia_predictions': all_predictions
        }
    
    def detect_font_smart(self, image_path, confidence_threshold=0.75):
        """
        Smart detection: Try Gaborcselle first (fast), fall back to Storia if needed
        """
        print("\n🔍 Starting smart font detection...")
        
        # Step 1: Try Gaborcselle (fast, common fonts)
        print("  📍 Trying Gaborcselle (48 common fonts)...")
        gabor_result = self.detect_with_gabor(image_path)
        
        top_confidence = gabor_result['top_prediction']['confidence']
        print(f"  ✓ Gaborcselle top result: {gabor_result['top_prediction']['font_name']} ({top_confidence:.1%})")
        
        # If high confidence, use Gaborcselle result
        if top_confidence >= confidence_threshold:
            print(f"  ✅ High confidence! Using Gaborcselle result.")
            return {
                'success': True,
                'source': 'gaborcselle',
                'confidence': 'high',
                'primary_prediction': gabor_result['top_prediction'],
                'all_predictions': gabor_result['all_predictions']
            }
        
        # Step 2: Low confidence, use Storia for comprehensive search
        print(f"  ⚠️  Confidence below {confidence_threshold:.0%}, checking Storia (3,475 fonts)...")
        storia_result = self.detect_with_storia(image_path)
        
        # Get the best Storia prediction
        best_storia = storia_result['storia_predictions'][0]['prediction']['top_prediction']
        print(f"  ✓ Storia top result: {best_storia['font_name']} ({best_storia['confidence']:.1%})")
        
        return {
            'success': True,
            'source': 'storia',
            'confidence': 'medium',
            'primary_prediction': best_storia,
            'gabor_alternative': gabor_result['top_prediction'],
            'storia_predictions': storia_result['storia_predictions']
        }
    
    def _detect_text_regions(self, gray_image):
        """Detect text regions in image"""
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
        dilated = cv2.dilate(binary, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 and h > 20:
                regions.append((x, y, w, h))
        
        return regions
    
    def _softmax(self, x):
        """Compute softmax values"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()


def test_dual_detector(image_path):
    """Test the dual model detector"""
    detector = DualModelFontDetector()
    
    print("\n" + "="*60)
    print("DUAL MODEL FONT DETECTION TEST")
    print("="*60)
    
    results = detector.detect_font_smart(image_path, confidence_threshold=0.75)
    
    print("\n📊 RESULTS:")
    print(f"  Source: {results['source'].upper()}")
    print(f"  Confidence: {results['confidence'].upper()}")
    print(f"\n🎯 PRIMARY PREDICTION:")
    print(f"  {results['primary_prediction']['font_name']} ({results['primary_prediction']['confidence']:.1%})")
    
    if 'gabor_alternative' in results:
        print(f"\n💡 GABORCSELLE ALTERNATIVE:")
        print(f"  {results['gabor_alternative']['font_name']} ({results['gabor_alternative']['confidence']:.1%})")
    
    return results