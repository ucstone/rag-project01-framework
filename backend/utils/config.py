from enum import Enum
from typing import Dict, Any

class VectorDBProvider(str, Enum):
    MILVUS = "milvus"
    CHROMA = "chroma"
    # More providers can be added later

# 可以在这里添加其他配置相关的内容
MILVUS_CONFIG = {
    "uri": "03-vector-store/langchain_milvus.db",
    "index_types": {
        "flat": "FLAT",
        "hnsw": "HNSW",
        "auto": "AUTOINDEX"
    },
    "index_params": {
        "flat": {},
        "hnsw": {
            "M": 8,
            "efConstruction": 64
        },
        "auto": {}
    }
}

CHROMA_CONFIG = {
    "persist_directory": "03-vector-store/chroma",
    "index_types": {
        "hnsw": "hnsw",
        "flat": "flat",
        "cosine": "cosine",
        "l2": "l2"
    }
} 