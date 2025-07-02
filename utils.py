import os
import hashlib
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import streamlit as st
import config  # Make sure config.py exists and defines CACHE_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_file_hash(file_path: str) -> str:
    """Generate hash for file to detect changes."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def save_cache(key: str, data: Any) -> None:
    """Save data to cache."""
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(config.CACHE_DIR, f"{key}.json")
    with open(cache_file, 'w') as f:
        json.dump(data, f)

def load_cache(key: str) -> Optional[Any]:
    """Load data from cache."""
    cache_file = os.path.join(config.CACHE_DIR, f"{key}.json")
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    return None

def display_sources(sources: List[str]) -> None:
    """Display sources in a formatted way."""
    if not sources:
        return
    
    st.markdown("**📚 Sources:**")
    for i, source in enumerate(sources, 1):
        if source.startswith("http"):
            st.markdown(f"{i}. 🌐 [{source}]({source})")
        else:
            st.markdown(f"{i}. 📄 {os.path.basename(source)}")

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text for display."""
    return text[:max_length] + "..." if len(text) > max_length else text