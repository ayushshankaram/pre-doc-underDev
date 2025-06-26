# 📦 Python Dependencies

## 🛠️ Installation Commands

### 1. Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv edtech_rag

# Activate virtual environment
# On macOS/Linux:
source edtech_rag/bin/activate

# On Windows:
edtech_rag\Scripts\activate
```

### 2. Install All Dependencies
```bash

# Install all requirements
pip install -r requirements.txt
```

### 3. Alternative: Install Individually
```bash
# Web Framework
pip install fastapi==0.104.1
pip install uvicorn[standard]==0.24.0
pip install python-multipart==0.0.6
pip install jinja2==3.1.2
pip install python-dotenv==1.0.0

# Document Processing
pip install PyMuPDF==1.23.8
pip install python-pptx==0.6.23
pip install Pillow==10.1.0
pip install pytesseract==0.3.10

# AI and ML
pip install google-generativeai==0.3.2
pip install sentence-transformers==2.2.2
pip install chromadb==0.4.18
pip install numpy==1.24.4

# Others
pip install fpdf2==2.7.6
pip install pandas==2.1.3
pip install pydantic==2.5.0
```

## 🔧 System Dependencies

### macOS:
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system dependencies
brew install tesseract
brew install poppler
```

### Ubuntu/Debian:
```bash
# Update package list
sudo apt update

# Install system dependencies
sudo apt install tesseract-ocr tesseract-ocr-eng
sudo apt install poppler-utils
sudo apt install python3-dev python3-pip
sudo apt install libjpeg-dev zlib1g-dev libpng-dev
```

### Windows:
```bash
# Using Chocolatey (recommended)
choco install tesseract

# Or download manually:
# Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
# Poppler: https://github.com/oschwartz10612/poppler-windows/releases/




```

##  Quick Setup 
```
# Check Python version
python3 --version


# Create virtual environment
python3 -m venv edtech_rag
source edtech_rag/bin/activate


# Install requirements
pip install -r requirements.txt
```

### Create .env file
```
# Google Gemini API Configuration

GOOGLE_API_KEY=your_gemini_api_key_here

# Server Configuration
HOST=0.0.0.0
PORT=8000

# File Processing Configuration
MAX_FILE_SIZE=50000000
UPLOAD_DIR=uploads
PROCESSED_DIR=processed
VECTOR_DB_DIR=vector_db


```



## 📋 Package Versions Explanation

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | 0.104.1 | Modern web framework for APIs |
| `uvicorn` | 0.24.0 | ASGI server for FastAPI |
| `PyMuPDF` | 1.23.8 | PDF processing and text extraction |
| `python-pptx` | 0.6.23 | PowerPoint file processing |
| `pytesseract` | 0.3.10 | OCR (Optical Character Recognition) |
| `google-generativeai` | 0.3.2 | Google Gemini AI API client |
| `sentence-transformers` | 2.2.2 | Text embeddings for semantic search |
| `chromadb` | 0.4.18 | Vector database for embeddings |
| `fpdf2` | 2.7.6 | PDF generation |
| `Pillow` | 10.1.0 | Image processing |

