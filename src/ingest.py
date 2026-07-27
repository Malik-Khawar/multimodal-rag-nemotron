import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image
import pypdf

from src.config import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)

SUPPORTED_TEXT_EXTS = {".txt", ".log", ".csv", ".tsv", ".json", ".yaml", ".yml"}
SUPPORTED_MD_EXTS = {".md", ".markdown"}
SUPPORTED_PDF_EXTS = {".pdf"}
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".gif"}
SUPPORTED_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


class MultimodalIngestor:
    """Multimodal document parser and chunker for text, markdown, PDFs, and images."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _generate_chunk_id(self, source: str, page_number: Optional[int], chunk_index: int, content: str) -> str:
        """Generate a deterministic unique ID for a chunk."""
        raw_key = f"{source}::{page_number}::{chunk_index}::{content[:100]}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]

    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks based on word count / approximate token count."""
        words = text.strip().split()
        if not words:
            return []

        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        
        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            if chunk_text:
                chunks.append(chunk_text)
                
        return chunks

    def process_text_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process plain text files."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {e}")
            return []

        text_chunks = self._split_text_into_chunks(content)
        doc_id = file_path.name
        source_str = str(file_path.resolve())

        results = []
        for idx, chunk_text in enumerate(text_chunks):
            chunk_id = self._generate_chunk_id(source_str, None, idx, chunk_text)
            meta = {
                "id": chunk_id,
                "doc_id": doc_id,
                "source": source_str,
                "file_type": "text",
                "page_number": None,
                "chunk_index": idx,
                "content": chunk_text,
                "image_path": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            meta["metadata_json"] = json.dumps(meta)
            results.append(meta)

        return results

    def process_markdown_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process markdown files."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.error(f"Error reading markdown file {file_path}: {e}")
            return []

        text_chunks = self._split_text_into_chunks(content)
        doc_id = file_path.name
        source_str = str(file_path.resolve())

        results = []
        for idx, chunk_text in enumerate(text_chunks):
            chunk_id = self._generate_chunk_id(source_str, None, idx, chunk_text)
            meta = {
                "id": chunk_id,
                "doc_id": doc_id,
                "source": source_str,
                "file_type": "markdown",
                "page_number": None,
                "chunk_index": idx,
                "content": chunk_text,
                "image_path": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            meta["metadata_json"] = json.dumps(meta)
            results.append(meta)

        return results

    def process_pdf_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process PDF files page by page and extract text into metadata-rich chunks."""
        results = []
        doc_id = file_path.name
        source_str = str(file_path.resolve())

        try:
            reader = pypdf.PdfReader(str(file_path))
            chunk_counter = 0

            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                if not page_text.strip():
                    continue

                page_chunks = self._split_text_into_chunks(page_text)
                for chunk_text in page_chunks:
                    chunk_id = self._generate_chunk_id(source_str, page_num, chunk_counter, chunk_text)
                    meta = {
                        "id": chunk_id,
                        "doc_id": doc_id,
                        "source": source_str,
                        "file_type": "pdf",
                        "page_number": page_num,
                        "chunk_index": chunk_counter,
                        "content": chunk_text,
                        "image_path": None,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                    meta["metadata_json"] = json.dumps(meta)
                    results.append(meta)
                    chunk_counter += 1

        except Exception as e:
            logger.error(f"Error processing PDF file {file_path}: {e}")

        return results

    def process_image_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process image/diagram files using Pillow to extract metadata and prepare visual representation."""
        results = []
        doc_id = file_path.name
        source_str = str(file_path.resolve())

        try:
            with Image.open(file_path) as img:
                width, height = img.size
                img_format = img.format or file_path.suffix.lstrip(".").upper()
                mode = img.mode

                # Format a detailed textual metadata summary of the image/diagram
                image_description = (
                    f"Image/Diagram File: {doc_id} | Format: {img_format} | Dimensions: {width}x{height} pixels | Color Mode: {mode} | Path: {source_str}"
                )

                chunk_id = self._generate_chunk_id(source_str, None, 0, image_description)
                meta = {
                    "id": chunk_id,
                    "doc_id": doc_id,
                    "source": source_str,
                    "file_type": "image",
                    "page_number": None,
                    "chunk_index": 0,
                    "content": image_description,
                    "image_path": source_str,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                meta["metadata_json"] = json.dumps(meta)
                results.append(meta)

        except Exception as e:
            logger.error(f"Error processing image file {file_path}: {e}")

        return results

    def process_audio_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process audio files and extract audio metadata / transcript."""
        results = []
        doc_id = file_path.name
        source_str = str(file_path.resolve())
        duration_sec = 0.0

        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(file_path)
            duration_sec = round(len(audio) / 1000.0, 2)
        except Exception:
            pass

        content_desc = (
            f"Audio File: {doc_id} | Format: {file_path.suffix.upper()} | Duration: {duration_sec} sec | "
            f"Path: {source_str} | Multimodal Audio Stream indexed for retrieval."
        )

        chunk_id = self._generate_chunk_id(source_str, None, 0, content_desc)
        meta = {
            "id": chunk_id,
            "doc_id": doc_id,
            "source": source_str,
            "file_type": "audio",
            "page_number": None,
            "chunk_index": 0,
            "content": content_desc,
            "image_path": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        meta["metadata_json"] = json.dumps(meta)
        results.append(meta)
        return results

    def process_video_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process video files and extract video metadata and container specs."""
        results = []
        doc_id = file_path.name
        source_str = str(file_path.resolve())
        size_mb = round(file_path.stat().st_size / (1024 * 1024), 2)

        content_desc = (
            f"Video File: {doc_id} | Format: {file_path.suffix.upper()} | Size: {size_mb} MB | "
            f"Path: {source_str} | Multimodal Video Container indexed for retrieval."
        )

        chunk_id = self._generate_chunk_id(source_str, None, 0, content_desc)
        meta = {
            "id": chunk_id,
            "doc_id": doc_id,
            "source": source_str,
            "file_type": "video",
            "page_number": None,
            "chunk_index": 0,
            "content": content_desc,
            "image_path": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        meta["metadata_json"] = json.dumps(meta)
        results.append(meta)
        return results

    def process_file(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Process a single file based on its extension."""
        path = Path(file_path).resolve()
        if not path.is_file():
            logger.warning(f"File not found: {path}")
            return []

        ext = path.suffix.lower()
        if ext in SUPPORTED_MD_EXTS:
            return self.process_markdown_file(path)
        elif ext in SUPPORTED_TEXT_EXTS:
            return self.process_text_file(path)
        elif ext in SUPPORTED_PDF_EXTS:
            return self.process_pdf_file(path)
        elif ext in SUPPORTED_IMAGE_EXTS:
            return self.process_image_file(path)
        elif ext in SUPPORTED_AUDIO_EXTS:
            return self.process_audio_file(path)
        elif ext in SUPPORTED_VIDEO_EXTS:
            return self.process_video_file(path)
        else:
            logger.info(f"Skipping unsupported file extension: {ext} for {path}")
            return []

    def process_directory(self, directory_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Process all supported files within a directory recursively."""
        dir_path = Path(directory_path).resolve()
        if not dir_path.is_dir():
            logger.warning(f"Directory not found: {dir_path}")
            return []

        all_chunks = []
        for path in dir_path.rglob("*"):
            if path.is_file():
                chunks = self.process_file(path)
                all_chunks.extend(chunks)

        logger.info(f"Processed {len(all_chunks)} total chunks from directory {dir_path}")
        return all_chunks
