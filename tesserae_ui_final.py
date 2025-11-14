"""
tesserae_ui_final.py - Complete Tesserae Interface

Full design system extraction:
- Color palettes
- Pattern recognition (spacing, sizing, grid)
- Typography analysis and font suggestions
- Complete design token generation
"""

import gradio as gr
from pathlib import Path
from PIL import Image
import json
from image_upload import ImageUploadHandler
from color_extraction import ColorExtractor
from pattern_recognition import PatternRecognizer
from typography_analysis import TypographyAnalyzer


def create_tesserae_interface():
    """Create complete Tesserae interface with all features."""
    
    handler = ImageUploadHandler()
    color_extractor = ColorExtractor(n_colors=8)
    pattern_recognizer = PatternRecognizer()
    typography_analyzer = TypographyAnalyzer()
    
    # Store results for export
    current_results = {
        'palette': None,
        'patterns': None,
        'typography': None,
        'color_tokens': None,
        'pattern_tokens': None,
        'typography_tokens': None,
        'file_id': None
    }
    
    def process_moodboard(image, user_consent):
        """Process uploaded moodboard with complete analysis."""
        if not user_consent:
            return (
                "❌ Please review and accept the privacy policy before uploading.",
                None, "", "", "", "", "", None
            )
        
        if image is None:
            return "❌ Please upload an image first.", None, "", "", "", "", "", None
        
        try:
            # Save temporary file
            temp_path = Path("./temp_gradio_upload.png")
            image.save(temp_path)
            
            # Process upload
            result = handler.process_upload(temp_path, user_id="gradio_test_user")
            temp_path.unlink()
            
            if not result['success']:
                return f"❌ Upload failed: {result['error']}", None, "", "", "", "", "", None
            
            file_path = Path(result['path'])
            processed_img = handler.get_image_for_processing(result['file_id'])
            
            # Extract colors
            print("🎨 Extracting colors...")
            palette = color_extractor.extract_palette(str(file_path))
            color_tokens = color_extractor.generate_design_tokens(palette)
            
            # Recognize patterns
            print("🔍 Recognizing patterns...")
            patterns = pattern_recognizer.analyze_patterns(str(file_path))
            pattern_tokens = pattern_recognizer.generate_design_tokens(patterns)
            
            # Analyze typography
            print("🔤 Analyzing typography...")
            typography = typography_analyzer.analyze_typography(str(file_path))
            typography_tokens = typography_analyzer.generate_design_tokens(typography)
            
            # Store for export
            current_results['palette'] = palette
            current_results['patterns'] = patterns
            current_results['typography'] = typography
            current_results['color_tokens'] = color_tokens
            current_results['pattern_tokens'] = pattern_tokens
            current_results['typography_tokens'] = typography_tokens
            current_results['file_id'] = result['file_id']
            
            # Format upload metadata
            upload_metadata = f"""
✅ **Upload Successful!**

**File ID:** `{result['file_id']}`  
**Dimensions:** {result['dimensions']}  
**Format:** {result['format']}  
**Auto-delete Date:** {result['deletion_date'][:10]}
            """
            
            # Format color palette
            color_display = "## 🎨 Color Palette\n\n"
            for i, color in enumerate(palette['colors'][:5], 1):
                color_display += f"**{i}. {color['name'].title()}**  \n"
                color_display += f"• Hex: `{color['hex']}` • Coverage: {color['percentage']}% • Role: `{color['role']}`  \n\n"
            
            # Format patterns
            patterns_display = "## 📐 Spacing & Layout\n\n"
            patterns_display += "**Spacing Scale:**  \n"
            spacing_scale = patterns['spacing']['spacing_scale']
            for size, value in list(spacing_scale.items())[:4]:
                patterns_display += f"• **{size}**: {value}px  \n"
            
            patterns_display += f"\n**Grid:** {patterns['grid']['columns']} columns, {patterns['grid']['gutter']}px gutter  \n"
            
            # Format typography
            typography_display = "## 🔤 Typography\n\n"
            
            if typography['text_detected']:
                typography_display += f"**Text regions detected:** {typography['text_regions']}  \n\n"
                
                typography_display += "**Suggested Font Pairings:**\n\n"
                for pairing in typography['font_pairings'][:2]:
                    typography_display += f"**{pairing['name']}**  \n"
                    typography_display += f"• Heading: {pairing['heading']}  \n"
                    typography_display += f"• Body: {pairing['body']}  \n"
                    typography_display += f"• Use: {pairing['use_case']}  \n\n"
            else:
                typography_display += "*No text detected in image*  \n"
                typography_display += "Using recommended typography system  \n\n"
                
                typography_display += "**Recommended Fonts:**\n"
                for role, data in typography['font_suggestions'].items():
                    typography_display += f"• **{role.title()}**: {', '.join(data['fonts'][:2])}  \n"
            
            # Format complete analysis
            analysis = palette['analysis']
            analysis_display = f"""
## 📊 Complete Analysis

### Colors
- Scheme: {analysis['color_scheme'].title()}  
- Temperature: {analysis['temperature'].title()}  
- Accessible pairs: {len(analysis['accessibility'])}

### Patterns
- Spacing base: {patterns['spacing']['base_unit']}px  
- Grid: {patterns['grid']['columns']} columns  
- Regions analyzed: {patterns['spacing']['total_gaps_analyzed']}

### Typography
- Font scale: 8 sizes (xs to 4xl)  
- Weight: {typography['analysis']['dominant_weight']}  
- Text regions: {typography['text_regions']}
"""
            
            # Create color swatches
            swatches_html = create_color_swatches(palette['colors'])
            
            status = f"✅ Complete analysis! Extracted colors, patterns, and typography from your moodboard."
            
            return (
                status,
                processed_img,
                upload_metadata,
                color_display,
                patterns_display,
                typography_display,
                analysis_display,
                swatches_html
            )
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error: {error_details}")
            return f"❌ Error: {str(e)}", None, "", "", "", "", "", None
    
    def create_color_swatches(colors):
        """Create HTML color swatches."""
        html = '<div style="display: flex; flex-wrap: wrap; gap: 10px; padding: 10px;">'
        
        for color in colors[:8]:
            hex_color = color['hex']
            name = color['name']
            percentage = color['percentage']
            text_color = '#000' if color['hsl']['l'] > 50 else '#fff'
            
            html += f'''
            <div style="
                background-color: {hex_color};
                color: {text_color};
                padding: 15px;
                border-radius: 8px;
                min-width: 100px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
                <div style="font-weight: bold; font-size: 0.9em;">{hex_color}</div>
                <div style="font-size: 0.85em; margin-top: 5px;">{name}</div>
                <div style="font-size: 0.75em; margin-top: 3px; opacity: 0.8;">{percentage}%</div>
            </div>
            '''
        
        html += '</div>'
        return html
    
    def export_tokens_json():
        """Export complete design tokens as JSON."""
        if current_results['color_tokens'] is None:
            return "❌ No tokens to export. Process an image first."
        
        # Combine all tokens
        complete_tokens = {
            'color': current_results['color_tokens']['color'],
            'spacing': current_results['pattern_tokens']['spacing'],
            'sizing': current_results['pattern_tokens']['sizing'],
            'borderRadius': current_results['pattern_tokens']['borderRadius'],
            'boxShadow': current_results['pattern_tokens']['boxShadow'],
            'grid': current_results['pattern_tokens']['grid'],
            'typography': {
                'fontFamily': current_results['typography_tokens']['fontFamily'],
                'fontSize': current_results['typography_tokens']['fontSize'],
                'fontWeight': current_results['typography_tokens']['fontWeight'],
                'lineHeight': current_results['typography_tokens']['lineHeight'],
                'letterSpacing': current_results['typography_tokens']['letterSpacing']
            },
            'metadata': {
                **current_results['color_tokens']['metadata'],
                'generated_by': 'Tesserae v3.0 - Complete Design System Generator',
                'features': ['color', 'spacing', 'sizing', 'typography', 'grid', 'accessibility']
            }
        }
        
        return json.dumps(complete_tokens, indent=2)
    
    def export_css():
        """Export complete design tokens as CSS."""
        if current_results['color_tokens'] is None:
            return "❌ No tokens to export. Process an image first."
        
        color_css = color_extractor.export_css_variables(current_results['color_tokens'])
        pattern_css = pattern_recognizer.export_css_variables(current_results['pattern_tokens'])
        typography_css = typography_analyzer.export_css_variables(current_results['typography_tokens'])
        
        # Combine
        combined_css = "/* TESSERAE COMPLETE DESIGN SYSTEM */\n"
        combined_css += "/* Generated from moodboard analysis */\n\n"
        combined_css += "/* ==================== */\n"
        combined_css += "/* COLOR TOKENS */\n"
        combined_css += "/* ==================== */\n"
        combined_css += color_css
        combined_css += "\n\n/* ==================== */\n"
        combined_css += "/* SPACING & LAYOUT */\n"
        combined_css += "/* ==================== */\n"
        combined_css += pattern_css
        combined_css += "\n\n/* ==================== */\n"
        combined_css += "/* TYPOGRAPHY */\n"
        combined_css += "/* ==================== */\n"
        combined_css += typography_css
        
        return combined_css
    
    def check_expired_images():
        """Clean up expired images."""
        result = handler.cleanup_expired()
        if result['deleted_count'] > 0:
            return f"🗑️ Deleted {result['deleted_count']} expired image(s)."
        else:
            return "✅ No expired images to clean up."
    
    privacy_text = """
## 📋 Privacy & Data Policy

- ✅ Images analyzed by AI to extract complete design systems
- ✅ **Automatic deletion after 30 days**
- ✅ EXIF metadata stripped for privacy
- ✅ Stored locally, never shared with third parties
- ✅ Do not upload confidential client work without authorization

**By checking the box, you agree to these terms.**
    """
    
    # Create Gradio interface
    with gr.Blocks(title="Tesserae - Complete Design System Generator", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("# 🎨 Tesserae - AI Design System Generator")
        gr.Markdown("*Extract complete design systems from moodboards: colors, spacing, typography, and more*")
        
        with gr.Tab("🖼️ Upload & Analyze"):
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(privacy_text)
                    
                    consent_checkbox = gr.Checkbox(
                        label="I agree to the privacy policy",
                        value=False
                    )
                    
                    image_input = gr.Image(
                        label="Upload Your Moodboard",
                        type="pil",
                        height=350
                    )
                    
                    upload_btn = gr.Button("🚀 Analyze Moodboard", variant="primary", size="lg")
                    
                    status_output = gr.Textbox(
                        label="Status",
                        placeholder="Upload an image to get started...",
                        interactive=False
                    )
                    
                    upload_metadata = gr.Markdown("*Upload info will appear here*")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Processing Results")
                    
                    processed_image = gr.Image(
                        label="Processed Image",
                        type="pil"
                    )
                    
                    color_swatches = gr.HTML(label="Color Palette")
            
            with gr.Row():
                with gr.Column():
                    color_palette_output = gr.Markdown("*Colors*")
                
                with gr.Column():
                    patterns_output = gr.Markdown("*Patterns*")
                
                with gr.Column():
                    typography_output = gr.Markdown("*Typography*")
            
            with gr.Row():
                analysis_output = gr.Markdown("*Complete analysis*")
        
        with gr.Tab("📤 Export Design System"):
            gr.Markdown("## Export Complete Design System")
            gr.Markdown("Download your design tokens ready for Figma, web projects, and design tools")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📦 JSON Format")
                    gr.Markdown("*Complete design tokens - all systems combined*")
                    
                    json_output = gr.Code(
                        label="Design Tokens (JSON)",
                        language="json",
                        lines=30
                    )
                    
                    export_json_btn = gr.Button("📋 Generate JSON", variant="secondary")
                
                with gr.Column():
                    gr.Markdown("### 🎨 CSS Format")
                    gr.Markdown("*Production-ready CSS custom properties*")
                    
                    css_output = gr.Code(
                        label="CSS Custom Properties",
                        language="css",
                        lines=30
                    )
                    
                    export_css_btn = gr.Button("📋 Generate CSS", variant="secondary")
            
            gr.Markdown("""
### 💡 Using Your Design System

**What You Get:**
- 🎨 **Colors**: Palette with semantic roles (primary, accent, etc.)
- 📏 **Spacing**: 8px grid system (xs through 4xl)
- 📐 **Sizing**: Element dimension scale
- 🔘 **Border Radius**: Rounded corner values
- 🌑 **Shadows**: Elevation/depth system
- 📊 **Grid**: Column and gutter specifications
- 🔤 **Typography**: Font families, sizes, weights, line heights

**How to Use:**
- **Figma**: Import JSON with Tokens Studio plugin
- **Web**: Copy CSS variables directly into your stylesheet
- **React/Vue**: Use as design token foundation
- **Style Dictionary**: Transform tokens to any platform
            """)
        
        with gr.Tab("🔧 System"):
            gr.Markdown("## 🔧 System Management")
            
            cleanup_btn = gr.Button("🗑️ Clean Up Expired Images", variant="secondary")
            cleanup_output = gr.Textbox(label="Cleanup Status", interactive=False)
            
            gr.Markdown("""
### 📝 Storage Information
- Automatic deletion after 30 days
- Manual cleanup available anytime
- All processing happens locally
- Privacy-first architecture
            """)
        
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
## 🎯 About Tesserae

**Tesserae** is a complete AI-powered design system generator that extracts production-ready design tokens from moodboard images.

### Complete Feature Set

✅ **Color System**
- K-means clustering for dominant color extraction
- Semantic naming and role detection
- WCAG 2.1 accessibility compliance checking
- Color harmony analysis

✅ **Pattern Recognition**
- Spacing system detection (8px grid)
- Element sizing patterns
- Border radius values
- Grid system analysis (12/16/24 column)
- Shadow/elevation system

✅ **Typography Analysis**
- Text region detection
- Font characteristic analysis
- Google Fonts suggestions
- Font pairing recommendations
- Complete typography scale generation

✅ **Export Formats**
- JSON design tokens (Style Dictionary compatible)
- CSS custom properties
- Ready for Figma, web, mobile, and design tools

### Technical Implementation

**Computer Vision:**
- K-means clustering (n=8) for color extraction
- Canny edge detection for spacing/sizing
- Contour analysis for pattern recognition
- Text region detection for typography

**AI & ML:**
- Unsupervised learning for color clustering
- Statistical analysis for pattern normalization
- Heuristic algorithms for design token generation

**Design Theory:**
- Color harmony and temperature analysis
- Modular scale for typography (1.25 ratio)
- 8px spacing grid system
- WCAG 2.1 accessibility standards

### Ethics & Privacy

- Privacy-first with 30-day auto-deletion
- EXIF metadata stripping
- No training on user data without consent
- Transparent AI decision-making
- Human oversight for all final decisions
- Accessibility built into core functionality

### Project Information

**Creator:** Vivian Zhou  
**Course:** Applied AI Design and Development Lab  
**Instructor:** Dan Bartlett  
**Date:** October 2025

**Built with:** Python, OpenCV, scikit-learn, Gradio  

---

*Tesserae transforms moodboards into production-ready design systems in seconds.*
            """)
        
        # Connect event handlers
        upload_btn.click(
            fn=process_moodboard,
            inputs=[image_input, consent_checkbox],
            outputs=[
                status_output,
                processed_image,
                upload_metadata,
                color_palette_output,
                patterns_output,
                typography_output,
                analysis_output,
                color_swatches
            ]
        )
        
        export_json_btn.click(fn=export_tokens_json, outputs=json_output)
        export_css_btn.click(fn=export_css, outputs=css_output)
        cleanup_btn.click(fn=check_expired_images, outputs=cleanup_output)
    
    return demo


def launch_tesserae_ui(share=False):
    """Launch the complete Tesserae interface."""
    demo = create_tesserae_interface()
    demo.launch(share=share, server_name="0.0.0.0", server_port=7860)