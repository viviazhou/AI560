import os
from pathlib import Path
from PIL import Image
import torch
from datetime import datetime, timedelta
import hashlib

class ImageUploadHandler:
    """
    Handles image uploads for Tesserae with privacy-first design.
    Implements 30-day auto-deletion as per ethics guidelines.
    """
    
    def __init__(self, upload_dir="./uploads", max_size_mb=10):
        """
        Initialize the upload handler.
        
        Args:
            upload_dir: Directory to store uploaded images
            max_size_mb: Maximum file size in megabytes
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.allowed_formats = {'.jpg', '.jpeg', '.png', '.webp'}
        
        # Metadata tracking for 30-day deletion
        self.metadata_file = self.upload_dir / "upload_metadata.json"
        self._init_metadata()
    
    def _init_metadata(self):
        """Initialize or load metadata tracking file."""
        import json
        if not self.metadata_file.exists():
            with open(self.metadata_file, 'w') as f:
                json.dump({}, f)
    
    def _generate_safe_filename(self, original_name):
        """Generate a unique, safe filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_suffix = hashlib.md5(original_name.encode()).hexdigest()[:8]
        ext = Path(original_name).suffix.lower()
        return f"moodboard_{timestamp}_{hash_suffix}{ext}"
    
    def validate_image(self, file_path):
        """
        Validate uploaded image meets requirements.
        
        Returns:
            tuple: (is_valid, error_message)
        """
        file_path = Path(file_path)
        
        # Check file exists
        if not file_path.exists():
            return False, "File does not exist"
        
        # Check file size
        if file_path.stat().st_size > self.max_size_bytes:
            return False, f"File exceeds {self.max_size_bytes / (1024*1024):.1f}MB limit"
        
        # Check file format
        if file_path.suffix.lower() not in self.allowed_formats:
            return False, f"Format must be one of: {', '.join(self.allowed_formats)}"
        
        # Try to open with PIL to verify it's a valid image
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True, "Valid image"
        except Exception as e:
            return False, f"Invalid or corrupted image: {str(e)}"
    
    def process_upload(self, source_path, user_id=None):
        """
        Process an uploaded image with privacy safeguards.
        
        Args:
            source_path: Path to the uploaded file
            user_id: Optional user identifier for multi-user systems
            
        Returns:
            dict: Upload result with file_id and metadata
        """
        import json
        import shutil
        
        # Validate the image
        is_valid, message = self.validate_image(source_path)
        if not is_valid:
            return {"success": False, "error": message}
        
        # Generate safe filename and copy to upload directory
        safe_name = self._generate_safe_filename(Path(source_path).name)
        dest_path = self.upload_dir / safe_name
        shutil.copy2(source_path, dest_path)
        
        # Remove EXIF data for privacy (strip metadata)
        self._strip_metadata(dest_path)
        
        # Get basic image info
        with Image.open(dest_path) as img:
            width, height = img.size
            format_type = img.format
        
        # Record metadata with deletion date
        upload_time = datetime.now()
        deletion_date = upload_time + timedelta(days=30)
        
        metadata = {
            "file_id": safe_name,
            "upload_time": upload_time.isoformat(),
            "deletion_date": deletion_date.isoformat(),
            "user_id": user_id,
            "width": width,
            "height": height,
            "format": format_type,
            "original_name": Path(source_path).name
        }
        
        # Save metadata
        with open(self.metadata_file, 'r') as f:
            all_metadata = json.load(f)
        
        all_metadata[safe_name] = metadata
        
        with open(self.metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        return {
            "success": True,
            "file_id": safe_name,
            "path": str(dest_path),
            "deletion_date": deletion_date.isoformat(),
            "dimensions": f"{width}x{height}",
            "format": format_type
        }
    
    def _strip_metadata(self, image_path):
        """
        Remove EXIF and other metadata from image for privacy.
        Keeps only essential image data.
        """
        try:
            img = Image.open(image_path)
            
            # Create new image without EXIF data
            data = list(img.getdata())
            image_without_exif = Image.new(img.mode, img.size)
            image_without_exif.putdata(data)
            
            # Save back to same path
            image_without_exif.save(image_path)
            
        except Exception as e:
            print(f"Warning: Could not strip metadata: {e}")
    
    def cleanup_expired(self):
        """
        Remove images past their 30-day retention period.
        Should be run periodically (daily cron job recommended).
        """
        import json
        
        with open(self.metadata_file, 'r') as f:
            all_metadata = json.load(f)
        
        now = datetime.now()
        deleted_files = []
        
        for file_id, metadata in list(all_metadata.items()):
            deletion_date = datetime.fromisoformat(metadata['deletion_date'])
            
            if now >= deletion_date:
                file_path = self.upload_dir / file_id
                if file_path.exists():
                    file_path.unlink()
                    deleted_files.append(file_id)
                
                del all_metadata[file_id]
        
        # Save updated metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        return {
            "deleted_count": len(deleted_files),
            "deleted_files": deleted_files
        }
    
    def get_image_for_processing(self, file_id):
        """
        Load image for AI processing.
        
        Returns:
            PIL.Image or None
        """
        file_path = self.upload_dir / file_id
        
        if not file_path.exists():
            return None
        
        try:
            return Image.open(file_path)
        except Exception as e:
            print(f"Error loading image: {e}")
            return None


# Example usage in notebook
def create_upload_interface():
    """Create a simple Jupyter widget for image upload."""
    import ipywidgets as widgets
    from IPython.display import display
    
    handler = ImageUploadHandler()
    
    uploader = widgets.FileUpload(
        accept='image/*',
        multiple=False,
        description='Upload Moodboard'
    )
    
    output = widgets.Output()
    
    def on_upload_change(change):
        with output:
            output.clear_output()
            
            # Get uploaded file
            uploaded_file = list(uploader.value.values())[0]
            
            # Save temporarily
            temp_path = Path("./temp_upload.jpg")
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file['content'])
            
            # Process upload
            result = handler.process_upload(temp_path)
            
            # Clean up temp file
            temp_path.unlink()
            
            if result['success']:
                print(f"✅ Upload successful!")
                print(f"File ID: {result['file_id']}")
                print(f"Dimensions: {result['dimensions']}")
                print(f"Auto-delete date: {result['deletion_date']}")
                
                # Display the image
                img = handler.get_image_for_processing(result['file_id'])
                if img:
                    display(img)
            else:
                print(f"❌ Upload failed: {result['error']}")
    
    uploader.observe(on_upload_change, names='value')
    
    display(uploader, output)