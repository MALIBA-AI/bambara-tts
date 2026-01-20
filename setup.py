import os
from setuptools import find_packages, setup

this_directory = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="maliba_ai",
    version="2.0.0-beta",
    author="sudoping01",  
    author_email="sudoping01@gmail.com",  
    description="Bambara Text-to-Speech system using Maliba-AI models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MALIBA-AI/bambara-tts",  
    project_urls={
        "Bug Tracker": "https://github.com/MALIBA-AI/bambara-tts/issues",
        "Documentation": "https://github.com/MALIBA-AI/bambara-tts/wiki",
        "Source Code": "https://github.com/MALIBA-AI/bambara-tts",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "License :: Other/Proprietary License",  
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",

    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0", 
        "transformers>=4.40.0",
        "huggingface_hub>=0.20.0",
    
        "librosa==0.11.0",
        "soundfile==0.13.1",
        "soxr==0.5.0.post1",
        
        "einops==0.8.1",
        "einx",
        
        "omegaconf",
        "numpy>=1.24.0",
        "tqdm>=4.60.0",
        "requests>=2.25.0",
        "safetensors>=0.4.0",
        "packaging>=20.0",
         "unsloth==2025.9.7",  
         "bambara-text-normalizer"
    ],

    extras_require={
        "full": [
             "unsloth==2025.9.7",  
            "bitsandbytes>=0.41.0; platform_system!='Darwin'",
            "triton>=2.1.0,<3.0.0; platform_system=='Linux'",
            "scipy>=1.10.0",
            "scikit-learn>=1.2.0",
            "numba>=0.56.0",
            "llvmlite>=0.39.0",
            "audioread>=3.0.0",
            "decorator>=5.1.0",
            "joblib>=1.2.0",
            "lazy_loader>=0.3.0",
            "pooch>=1.6.0",
            "regex>=2023.0.0",
            "filelock>=3.9.0",
            "fsspec>=2023.5.0",
            "tokenizers>=0.15.0",
            "PyYAML>=6.0",
            "typing_extensions>=4.5.0",
            "platformdirs>=3.0.0",
            "certifi>=2022.0.0",
            "charset_normalizer>=3.0.0",
            "idna>=3.4.0",
            "urllib3>=1.26.0,<3.0.0",
            "sympy>=1.12.0",
            "networkx>=3.0.0",
            "jinja2>=3.1.0",
        ],
        
        "unsloth": [
             "unsloth==2025.9.7", 
            "bitsandbytes>=0.41.0; platform_system!='Darwin'",
            "triton>=2.1.0,<3.0.0; platform_system=='Linux'",
        ],
        
        "dev": [
            "ruff==0.11.4",
            "isort==6.0.1", 
            "pre-commit==4.2.0",
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    
    keywords=[
        "bambara",
        "tts", 
        "text-to-speech",
        "speech synthesis",
        "maliba-ai",
        "artificial intelligence",
        "machine learning",
        "natural language processing",
    ],
    include_package_data=True,
    zip_safe=False,
    license="CC BY-NC-SA 4.0",
    platforms=["any"],

)
