"""
font_detection_standalone.py - Standalone Font Detection Tool

Uses Storia.ai font repository and image analysis to:
1. Detect fonts in images using OCR + metadata analysis
2. Match to Storia.ai font database
3. Find similar fonts in Google Fonts library
4. Provide confidence scores and alternatives

This is separate from Tesserae for now - will integrate later.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2
from collections import Counter
from typing import List, Dict, Tuple, Optional
import requests
import json
from pathlib import Path


class FontDetector:
    """
    Standalone font detection tool.
    Analyzes images to detect fonts and suggest similar alternatives.
    """
    
    def __init__(self, storia_api_key: Optional[str] = None):
        """
        Initialize font detector.
        
        Args:
            storia_api_key: API key for Storia.ai (if available)
        """
        self.storia_api_key = storia_api_key
        self.google_fonts_cache = None
        
        # Font feature vectors for matching
        self.font_features = {
            'x-height': 0.0,      # Ratio of lowercase height to cap height
            'width-variation': 0.0,  # Width variance between characters
            'weight': 0.0,         # Stroke weight/thickness
            'contrast': 0.0,       # Thick/thin stroke contrast
            'serif': False,        # Has serifs
            'slant': 0.0,         # Italic angle
            'terminal': '',       # Terminal style (ball, wedge, etc.)
            'aperture': 0.0       # Openness of letterforms
        }
        
        # Load Google Fonts metadata
        self._load_google_fonts_database()
    
    def detect_font(self, image_path: str, use_ocr: bool = True) -> Dict:
        """
        Main font detection function.
        
        Args:
            image_path: Path to image containing text
            use_ocr: Whether to use OCR for text extraction
            
        Returns:
            dict: Detection results with matches and confidence
        """
        print(f"🔍 Analyzing image: {image_path}")
        
        # Load image
        img = Image.open(image_path)
        img_cv = cv2.imread(str(image_path))
        
        if img_cv is None:
            img_rgb = img.convert('RGB')
            img_cv = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
        
        # Step 1: Detect text regions
        print("  📍 Detecting text regions...")
        text_regions = self._detect_text_regions(img_cv)
        
        if not text_regions:
            return {
                'success': False,
                'error': 'No text regions detected in image',
                'suggestion': 'Try an image with clearer text'
            }
        
        print(f"  ✅ Found {len(text_regions)} text region(s)")
        
        # Step 2: Extract font features from text
        print("  🔬 Extracting font features...")
        features = self._extract_font_features(img_cv, text_regions)
        
        # Step 3: Match against Storia.ai if available
        storia_matches = []
        if self.storia_api_key:
            print("  🔗 Querying Storia.ai database...")
            storia_matches = self._query_storia(features)
        else:
            print("  ⚠️  No Storia.ai API key - skipping Storia database")
        
        # Step 4: Find similar fonts in Google Fonts
        print("  🔎 Matching to Google Fonts...")
        google_matches = self._match_google_fonts(features)
        
        # Step 5: Compile results
        results = {
            'success': True,
            'text_regions_detected': len(text_regions),
            'extracted_features': features,
            'storia_matches': storia_matches,
            'google_fonts_suggestions': google_matches,
            'confidence': self._calculate_confidence(features, text_regions)
        }
        
        return results
    
    def _detect_text_regions(self, img: np.ndarray) -> List[Dict]:
        """
        Detect regions containing text using advanced techniques.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Method 1: MSER (Maximally Stable Extremal Regions) - great for text
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        
        text_regions = []
        height, width = img.shape[:2]
        
        for region in regions:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(region)
            
            # Filter for text-like characteristics
            aspect_ratio = w / float(h) if h > 0 else 0
            area = w * h
            
            if (0.1 < aspect_ratio < 20 and  # Text aspect ratio
                50 < area < (width * height * 0.5) and  # Reasonable size
                h > 8 and w > 8):  # Minimum dimensions
                
                # Calculate character density (how much is filled vs empty)
                roi = gray[y:y+h, x:x+w]
                _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                density = np.sum(binary > 0) / (w * h)
                
                if 0.1 < density < 0.8:  # Text typically has medium density
                    text_regions.append({
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'density': density
                    })
        
        # Sort by size (largest first)
        text_regions.sort(key=lambda r: r['area'], reverse=True)
        
        return text_regions[:10]  # Return top 10 regions
    
    def _extract_font_features(self, img: np.ndarray, text_regions: List[Dict]) -> Dict:
        """
        Extract detailed font features from text regions.
        This is the core of font identification.
        """
        features = {
            'stroke_width': [],
            'stroke_width_variance': 0.0,
            'x_height_ratio': 0.0,
            'character_width_variance': 0.0,
            'has_serifs': False,
            'serif_confidence': 0.0,
            'slant_angle': 0.0,
            'contrast_ratio': 0.0,
            'weight_class': 'regular',
            'character_spacing': 0.0
        }
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Analyze the largest/clearest text region
        for region in text_regions[:3]:  # Analyze top 3 regions
            x, y, w, h = region['x'], region['y'], region['width'], region['height']
            roi = gray[y:y+h, x:x+w]
            
            # Enhance contrast
            roi = cv2.equalizeHist(roi)
            
            # Binarize
            _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # 1. Stroke Width Analysis
            # Use distance transform to measure stroke thickness
            dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            stroke_widths = dist_transform[dist_transform > 0]
            
            if len(stroke_widths) > 0:
                features['stroke_width'].extend(stroke_widths.tolist())
        
        # Calculate stroke statistics
        if features['stroke_width']:
            stroke_array = np.array(features['stroke_width'])
            mean_stroke = np.mean(stroke_array)
            std_stroke = np.std(stroke_array)
            
            features['stroke_width_variance'] = float(std_stroke / mean_stroke) if mean_stroke > 0 else 0
            features['contrast_ratio'] = float(np.max(stroke_array) / np.min(stroke_array)) if np.min(stroke_array) > 0 else 1.0
            
            # Determine weight class based on stroke width
            avg_stroke = float(mean_stroke)
            if avg_stroke < 2:
                features['weight_class'] = 'light'
            elif avg_stroke < 3:
                features['weight_class'] = 'regular'
            elif avg_stroke < 4.5:
                features['weight_class'] = 'medium'
            else:
                features['weight_class'] = 'bold'
        
        # 2. Serif Detection
        # Look for small projections at character edges
        serif_score = self._detect_serifs(img, text_regions[0] if text_regions else None)
        features['has_serifs'] = serif_score > 0.3
        features['serif_confidence'] = float(serif_score)
        
        # 3. Slant/Italic Detection
        slant = self._detect_slant(img, text_regions[0] if text_regions else None)
        features['slant_angle'] = float(slant)
        
        return features
    
    def _detect_serifs(self, img: np.ndarray, region: Optional[Dict]) -> float:
        """
        Detect presence of serifs.
        Returns confidence score 0-1.
        """
        if region is None:
            return 0.0
        
        x, y, w, h = region['x'], region['y'], region['width'], region['height']
        roi = img[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Look for small protrusions at letter edges (serif characteristic)
        # This is a simplified heuristic
        edge_density = np.sum(edges > 0) / (w * h)
        
        # Serifs typically create more edge complexity
        # Sans-serif fonts have cleaner edges
        if edge_density > 0.15:
            return min(edge_density / 0.3, 1.0)
        
        return 0.0
    
    def _detect_slant(self, img: np.ndarray, region: Optional[Dict]) -> float:
        """
        Detect italic slant angle.
        Returns angle in degrees.
        """
        if region is None:
            return 0.0
        
        x, y, w, h = region['x'], region['y'], region['width'], region['height']
        roi = img[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Detect lines using Hough transform
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=10, maxLineGap=5)
        
        if lines is None:
            return 0.0
        
        # Calculate average angle of vertical strokes
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Only consider roughly vertical lines
            if abs(x2 - x1) < w * 0.3:  # Mostly vertical
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)
        
        if angles:
            median_angle = np.median(angles)
            # Normalize to 0-90 range
            return float(abs(median_angle - 90))
        
        return 0.0
    
    def _query_storia(self, features: Dict) -> List[Dict]:
        """
        Query Storia.ai API for font matches.
        
        Note: This is a placeholder - you'll need to implement
        actual Storia.ai API integration based on their docs.
        """
        if not self.storia_api_key:
            return []
        
        # TODO: Implement actual Storia.ai API call
        # This is pseudocode showing what the integration would look like:
        
        # try:
        #     response = requests.post(
        #         'https://api.storia.ai/v1/font/identify',
        #         headers={'Authorization': f'Bearer {self.storia_api_key}'},
        #         json={
        #             'features': features,
        #             'max_results': 10
        #         }
        #     )
        #     
        #     if response.status_code == 200:
        #         return response.json()['matches']
        # except Exception as e:
        #     print(f"Storia.ai query failed: {e}")
        
        return []
    
    def _load_google_fonts_database(self):
        """
        Load Google Fonts metadata.
        Uses Google Fonts API to get available fonts and their characteristics.
        """
        # Simplified Google Fonts database
        # In production, you'd fetch this from Google Fonts API
        self.google_fonts_cache = {
            'serif': {
                'light': ['Lora', 'Crimson Text', 'PT Serif'],
                'regular': ['Merriweather', 'Playfair Display', 'Source Serif Pro'],
                'medium': ['Bitter', 'Arvo', 'Rokkitt'],
                'bold': ['Cardo', 'Copse', 'Neuton']
            },
            'sans-serif': {
                'light': ['Raleway', 'Nunito', 'Quicksand'],
                'regular': ['Open Sans', 'Roboto', 'Lato', 'Montserrat', 'Inter'],
                'medium': ['Poppins', 'Work Sans', 'Mulish'],
                'bold': ['Oswald', 'Bebas Neue', 'Anton']
            },
            'monospace': {
                'light': ['Source Code Pro'],
                'regular': ['Roboto Mono', 'Fira Code', 'IBM Plex Mono'],
                'medium': ['JetBrains Mono'],
                'bold': ['Courier Prime']
            }
        }
    
    def _match_google_fonts(self, features: Dict) -> List[Dict]:
        """
        Match extracted features to Google Fonts library.
        """
        matches = []
        
        # Determine category (serif vs sans-serif)
        category = 'serif' if features.get('has_serifs', False) else 'sans-serif'
        
        # Get weight class
        weight = features.get('weight_class', 'regular')
        
        # Get fonts for this category and weight
        if category in self.google_fonts_cache and weight in self.google_fonts_cache[category]:
            fonts = self.google_fonts_cache[category][weight]
            
            for font in fonts:
                confidence = self._calculate_font_confidence(features, category, weight, font)
                
                matches.append({
                    'font_name': font,
                    'category': category,
                    'weight': weight,
                    'confidence': confidence,
                    'google_fonts_url': f"https://fonts.google.com/specimen/{font.replace(' ', '+')}",
                    'css_import': f"@import url('https://fonts.googleapis.com/css2?family={font.replace(' ', '+')}');"
                })
        
        # Sort by confidence
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        
        return matches[:5]  # Return top 5 matches
    
    def _calculate_font_confidence(self, features: Dict, category: str, weight: str, font: str) -> float:
        """
        Calculate confidence score for a specific font match.
        """
        confidence = 0.5  # Base confidence
        
        # Boost confidence for correct category
        if category == 'serif' and features.get('has_serifs'):
            confidence += features.get('serif_confidence', 0) * 0.3
        elif category == 'sans-serif' and not features.get('has_serifs'):
            confidence += (1 - features.get('serif_confidence', 0)) * 0.3
        
        # Weight match adds confidence
        if weight == features.get('weight_class'):
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _calculate_confidence(self, features: Dict, text_regions: List[Dict]) -> float:
        """
        Calculate overall detection confidence.
        """
        confidence = 0.3  # Base confidence
        
        # More text regions = higher confidence
        if len(text_regions) > 0:
            confidence += min(len(text_regions) * 0.1, 0.3)
        
        # Clear stroke detection = higher confidence
        if features.get('stroke_width'):
            confidence += 0.2
        
        # Serif detection (clear yes or no) = higher confidence
        serif_conf = features.get('serif_confidence', 0)
        if serif_conf > 0.7 or serif_conf < 0.3:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def export_results(self, results: Dict, output_path: str = 'font_detection_results.json'):
        """
        Export detection results to JSON file.
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results exported to {output_path}")


# Standalone testing function
def test_font_detection(image_path: str, storia_key: Optional[str] = None):
    """
    Test the font detector on an image.
    """
    detector = FontDetector(storia_api_key=storia_key)
    results = detector.detect_font(image_path)
    
    if results['success']:
        print("\n" + "="*60)
        print("FONT DETECTION RESULTS")
        print("="*60)
        
        print(f"\n📊 Analysis Summary:")
        print(f"  • Text regions found: {results['text_regions_detected']}")
        print(f"  • Overall confidence: {results['confidence']:.0%}")
        
        print(f"\n🔬 Detected Features:")
        features = results['extracted_features']
        print(f"  • Category: {'Serif' if features.get('has_serifs') else 'Sans-serif'}")
        print(f"  • Weight: {features.get('weight_class', 'unknown').title()}")
        print(f"  • Slant: {features.get('slant_angle', 0):.1f}°")
        print(f"  • Stroke variance: {features.get('stroke_width_variance', 0):.3f}")
        
        if results['google_fonts_suggestions']:
            print(f"\n🎨 Google Fonts Suggestions:")
            for i, match in enumerate(results['google_fonts_suggestions'], 1):
                print(f"\n  {i}. {match['font_name']}")
                print(f"     Confidence: {match['confidence']:.0%}")
                print(f"     Category: {match['category']}")
                print(f"     Weight: {match['weight']}")
                print(f"     URL: {match['google_fonts_url']}")
        
        if results['storia_matches']:
            print(f"\n🏛️  Storia.ai Matches:")
            for match in results['storia_matches']:
                print(f"  • {match}")
        
        print("\n" + "="*60)
    else:
        print(f"\n❌ Detection failed: {results.get('error')}")
        print(f"   Suggestion: {results.get('suggestion')}")
    
    return results