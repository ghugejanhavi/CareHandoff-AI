"""
Clinical Guidelines Loader
Loads ACC/AHA PDFs and AHRQ RED Toolkit documents from data/raw/guidelines/.

Supported formats: .pdf, .txt, .html, .htm
Each loaded document becomes a LangChain Document for ChromaDB ingestion.
"""

import os
import logging
from typing import List
from langchain_core.documents import Document

from config import GUIDELINES_DIR

logger = logging.getLogger(__name__)


class GuidelinesLoader:
    """
    Loads clinical guideline documents from the guidelines directory.
    Returns a flat list of LangChain Documents ready for chunking and embedding.
    """

    def __init__(self, guidelines_dir: str = GUIDELINES_DIR):
        self.guidelines_dir = guidelines_dir

    def load_all(self) -> List[Document]:
        """
        Walk the guidelines directory, load every supported file.
        Returns a list of Documents with source metadata attached.
        """
        if not os.path.exists(self.guidelines_dir):
            logger.warning("Guidelines directory not found: %s", self.guidelines_dir)
            return []

        docs: List[Document] = []
        for fname in os.listdir(self.guidelines_dir):
            fpath = os.path.join(self.guidelines_dir, fname)
            ext = os.path.splitext(fname)[1].lower()

            if ext == ".pdf":
                docs.extend(self._load_pdf(fpath, fname))
            elif ext in (".txt",):
                docs.extend(self._load_text(fpath, fname))
            elif ext in (".html", ".htm"):
                docs.extend(self._load_html(fpath, fname))
            else:
                logger.debug("Skipping unsupported file: %s", fname)

        logger.info("Loaded %d guideline documents from %s", len(docs), self.guidelines_dir)
        return docs

    # ── Format-specific loaders ────────────────────────────────────────────

    def _load_pdf(self, path: str, fname: str) -> List[Document]:
        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            docs = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": fname,
                            "page": i + 1,
                            "doc_type": "guideline",
                            "format": "pdf",
                        }
                    ))
            return docs
        except ImportError:
            logger.error("pypdf not installed. Run: pip install pypdf")
            return []
        except Exception as e:
            logger.error("Failed to load PDF %s: %s", fname, e)
            return []

    def _load_text(self, path: str, fname: str) -> List[Document]:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            if text.strip():
                return [Document(
                    page_content=text,
                    metadata={"source": fname, "doc_type": "guideline", "format": "txt"}
                )]
            return []
        except Exception as e:
            logger.error("Failed to load text file %s: %s", fname, e)
            return []

    def _load_html(self, path: str, fname: str) -> List[Document]:
        try:
            from bs4 import BeautifulSoup
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
            text = soup.get_text(separator="\n")
            if text.strip():
                return [Document(
                    page_content=text,
                    metadata={"source": fname, "doc_type": "guideline", "format": "html"}
                )]
            return []
        except ImportError:
            # Fall back to plain text read if BeautifulSoup not available
            return self._load_text(path, fname)
        except Exception as e:
            logger.error("Failed to load HTML file %s: %s", fname, e)
            return []

    def list_available_files(self) -> List[str]:
        """Return names of all guideline files present in the directory."""
        if not os.path.exists(self.guidelines_dir):
            return []
        return [
            f for f in os.listdir(self.guidelines_dir)
            if os.path.splitext(f)[1].lower() in (".pdf", ".txt", ".html", ".htm")
        ]

    def is_data_available(self) -> bool:
        return len(self.list_available_files()) > 0
