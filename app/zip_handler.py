"""
ZIP Archive Handler
Handles extraction and processing of ZIP files containing images and HTML documents
"""
import os
import zipfile
import tempfile
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


class ZipHandler:
    """Handles ZIP file extraction and validation"""
    
    def __init__(self):
        self.allowed_extensions = (
            settings.ALLOWED_IMAGE_EXTENSIONS + 
            settings.ALLOWED_HTML_EXTENSIONS
        )
    
    def is_valid_file(self, filename: str) -> bool:
        """Check if file has valid extension"""
        ext = os.path.splitext(filename)[1].lower()
        return ext in self.allowed_extensions
    
    def extract_and_validate(self, zip_path: str, extract_dir: str) -> Tuple[List[str], Dict[str, Any]]:
        """
        Extract ZIP file and return list of valid files
        
        Returns:
            Tuple of (list of extracted file paths, metadata dict)
        """
        extracted_files = []
        metadata = {
            "total_files": 0,
            "valid_files": 0,
            "skipped_files": 0,
            "total_size": 0,
            "errors": []
        }
        
        try:
            # Validate ZIP file
            if not zipfile.is_zipfile(zip_path):
                raise ValueError("Invalid ZIP file")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get file list
                file_list = zip_ref.namelist()
                metadata["total_files"] = len(file_list)
                
                # Check for zip bombs (files that expand too much)
                total_uncompressed_size = sum(info.file_size for info in zip_ref.infolist())
                
                if total_uncompressed_size > settings.MAX_ZIP_EXTRACT_SIZE:
                    raise ValueError(
                        f"ZIP file too large when extracted: {total_uncompressed_size / (1024*1024):.2f}MB "
                        f"(max: {settings.MAX_ZIP_EXTRACT_SIZE / (1024*1024):.2f}MB)"
                    )
                
                metadata["total_size"] = total_uncompressed_size
                
                # Extract valid files
                for file_info in zip_ref.infolist():
                    # Skip directories
                    if file_info.is_dir():
                        continue
                    
                    filename = file_info.filename
                    
                    # Skip hidden files and system files
                    if os.path.basename(filename).startswith('.') or '__MACOSX' in filename:
                        metadata["skipped_files"] += 1
                        continue
                    
                    # Check if file type is allowed
                    if not self.is_valid_file(filename):
                        logger.debug("Skipping unsupported file: %s", filename)
                        metadata["skipped_files"] += 1
                        continue
                    
                    # Check file count limit
                    if len(extracted_files) >= settings.MAX_FILES_IN_ZIP:
                        logger.warning("Reached maximum file limit (%d)", settings.MAX_FILES_IN_ZIP)
                        metadata["errors"].append(f"Maximum file limit reached ({settings.MAX_FILES_IN_ZIP})")
                        break
                    
                    try:
                        extract_dir_abs = os.path.abspath(extract_dir)
                        target_path = os.path.normpath(os.path.join(extract_dir, filename))
                        target_path_abs = os.path.abspath(target_path)
                        try:
                            if os.path.commonpath([target_path_abs, extract_dir_abs]) != extract_dir_abs or target_path_abs == extract_dir_abs:
                                metadata["skipped_files"] += 1
                                metadata["errors"].append(f"Rejected path escape: {filename}")
                                continue
                        except ValueError:
                            metadata["skipped_files"] += 1
                            metadata["errors"].append(f"Rejected path escape: {filename}")
                            continue
                        os.makedirs(os.path.dirname(target_path_abs), exist_ok=True)
                        with open(target_path_abs, "wb") as f:
                            f.write(zip_ref.read(file_info.filename))
                        extracted_files.append(target_path_abs)
                        metadata["valid_files"] += 1
                        logger.info("Extracted: %s", filename)
                    except (OSError, zipfile.BadZipFile) as e:
                        logger.error("Error extracting %s: %s", filename, e)
                        metadata["errors"].append(f"Failed to extract {filename}: {str(e)}")
                        continue
                
                logger.info(
                    "ZIP extraction complete: %d valid files, %d skipped",
                    metadata['valid_files'],
                    metadata['skipped_files']
                )
                
                return extracted_files, metadata
                
        except zipfile.BadZipFile as e:
            logger.error("Bad ZIP file: %s", e)
            raise ValueError("Corrupted ZIP file: %s" % str(e)) from e
        except (OSError, IOError) as e:
            logger.error("Error processing ZIP file: %s", e)
            raise
    
    def process_zip_file(self, zip_path: str, processor, output_dir: str) -> Dict[str, Any]:
        """
        Process all valid files in ZIP archive
        
        Args:
            zip_path: Path to ZIP file
            processor: GrammarCorrectionProcessor instance
            output_dir: Directory for output files
        
        Returns:
            Dictionary with processing results
        """
        # Create temporary extraction directory
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Extract files
                extracted_files, metadata = self.extract_and_validate(zip_path, temp_dir)
                
                if not extracted_files:
                    return {
                        "success": False,
                        "error": "No valid files found in ZIP archive",
                        "metadata": metadata,
                        "input_type": "zip"
                    }
                
                # Process each file
                results = []
                total_corrections = 0
                
                for file_path in extracted_files:
                    try:
                        logger.info("Processing: %s", os.path.basename(file_path))
                        
                        # Process individual file
                        result = processor.process_input(file_path, output_dir=output_dir)
                        
                        # Add filename to result
                        result["filename"] = os.path.basename(file_path)
                        results.append(result)
                        
                        # Count corrections
                        if result.get("success"):
                            total_corrections += result.get("corrections_count", 0)
                        
                    except (OSError, RuntimeError, ValueError) as e:
                        logger.error("Error processing %s: %s", file_path, e)
                        results.append({
                            "success": False,
                            "filename": os.path.basename(file_path),
                            "error": str(e)
                        })
                
                # Compile summary
                successful = sum(1 for r in results if r.get("success"))
                failed = len(results) - successful
                
                return {
                    "success": True,
                    "input_type": "zip",
                    "metadata": metadata,
                    "summary": {
                        "total_files_processed": len(results),
                        "successful": successful,
                        "failed": failed,
                        "total_corrections": total_corrections
                    },
                    "results": results
                }
                
            except (OSError, RuntimeError) as e:
                logger.error("Error processing ZIP file: %s", e)
                return {
                    "success": False,
                    "error": str(e),
                    "input_type": "zip"
                }


_zip_handler: ZipHandler = None


def get_zip_handler() -> ZipHandler:
    """Get or create singleton ZIP handler instance"""
    global _zip_handler
    if _zip_handler is None:
        _zip_handler = ZipHandler()
    return _zip_handler

