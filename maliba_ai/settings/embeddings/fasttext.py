from dataclasses import dataclass
from typing import List, Dict


class Settings: 
    model_repo: str = "MALIBA-AI/bambara-embeddings"
    model_file_name: str = "bam.bin"
    local_paths: List[str] = ["bam.bin", "./bam.bin", "../bam.bin"]

    
