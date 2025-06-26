import os
import shutil
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
from typing import Dict, List, Any, Optional
import re
# Import our enhanced components
from enhanced_rag_system import EnhancedRAGSystem
from enhanced_prerequisite_generator import EnhancedPrerequisiteGenerator

# Initialize FastAPI app
app = FastAPI(title="Enhanced EdTech RAG System", version="2.0.0")

# Setup directories
UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
VECTOR_DB_DIR = "vector_db"
TEMPLATES_DIR = "templates"
STATIC_DIR = "static"

for dir_path in [UPLOAD_DIR, PROCESSED_DIR, VECTOR_DB_DIR, TEMPLATES_DIR, STATIC_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Initialize templates and static files
templates = Jinja2Templates(directory=TEMPLATES_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/generated_docs", StaticFiles(directory="generated_docs"), name="generated_docs")
# Add this after the existing mounts
app.mount("/vector_db", StaticFiles(directory=VECTOR_DB_DIR), name="vector_db")

# Initialize enhanced components
# Note: You need to set your OpenAI API key as an environment variable
# export OPENAI_API_KEY="your-api-key-here"
rag_system = EnhancedRAGSystem(VECTOR_DB_DIR)
prereq_generator = EnhancedPrerequisiteGenerator(rag_system)

# Pydantic models
class UploadRequest(BaseModel):
    document_type: Optional[str] = "lecture"
    course_name: Optional[str] = ""

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    content_types: Optional[List[str]] = None

class PrerequisiteRequest(BaseModel):
    query: str
    target_level: Optional[str] = "undergraduate"
    include_images: Optional[bool] = True

class ChatRequest(BaseModel):
    query: str
    conversation_history: Optional[List[dict]] = None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Enhanced home page"""
    # Get system stats
    stats = rag_system.get_document_stats()
    
    return templates.TemplateResponse("enhanced_upload.html", {
        "request": request,
        "stats": stats
    })

@app.post("/upload/")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Enhanced document upload with comprehensive processing"""
    try:
        results = []
        
        for file in files:
            # Validate file type
            if not file.filename.lower().endswith(('.pdf', '.ppt', '.pptx')):
                raise HTTPException(400, f"Unsupported file type: {file.filename}")
            
            print(f"Processing file: {file.filename}")
            
            # Save uploaded file
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Process document with enhanced system
            try:
                processed_content = rag_system.process_and_add_document(file_path, file.filename)
                
                # Create detailed result
                result = {
                    "filename": file.filename,
                    "document_id": processed_content["document_id"],
                    "processing_details": {
                        "text_chunks": len(processed_content["text_chunks"]),
                        "images_extracted": len(processed_content["images"]),
                        "code_blocks": len(processed_content["code_blocks"]),
                        "formulas": len(processed_content["formulas"]),
                        "pages_processed": processed_content["metadata"]["total_pages"]
                    },
                    "content_analysis": {
                        "has_mathematical_content": len(processed_content["formulas"]) > 0,
                        "has_code_examples": len(processed_content["code_blocks"]) > 0,
                        "has_visual_content": len(processed_content["images"]) > 0,
                        "estimated_reading_time": len(processed_content["text_chunks"]) * 2  # 2 minutes per chunk
                    },
                    "status": "success"
                }
                
                print(f"✅ Successfully processed {file.filename}")
                print(f"   - Text chunks: {len(processed_content['text_chunks'])}")
                print(f"   - Images: {len(processed_content['images'])}")
                print(f"   - Code blocks: {len(processed_content['code_blocks'])}")
                print(f"   - Formulas: {len(processed_content['formulas'])}")
                
            except Exception as e:
                print(f"❌ Error processing {file.filename}: {e}")
                result = {
                    "filename": file.filename,
                    "status": "error",
                    "error": str(e)
                }
            
            results.append(result)
        
        return JSONResponse({
            "message": f"Processed {len(files)} documents",
            "results": results,
            "system_stats": rag_system.get_document_stats()
        })
    
    except Exception as e:
        raise HTTPException(500, f"Error processing documents: {str(e)}")

@app.post("/search/")
async def search_documents(request: SearchRequest):
    """Enhanced intelligent search across all content types"""
    try:
        print(f"Searching for: {request.query}")
        
        # Use intelligent search
        results = rag_system.intelligent_search(
            request.query, 
            request.top_k
        )
        
        if not results:
            return JSONResponse({
                "message": "No relevant content found",
                "suggestion": "Try uploading relevant documents or using different keywords",
                "query": request.query,
                "results": []
            })
        
        # Enhance results with additional metadata
        enhanced_results = []
        for result in results:
            enhanced_result = {
                "content": result["content"],
                "content_type": result["content_type"],
                "relevance_score": round(result["weighted_score"], 3),
                "source": {
                    "filename": result["metadata"]["filename"],
                    "page": result["metadata"]["page_number"],
                    "document_id": result["metadata"]["document_id"]
                }
            }
            
            # Add content-specific metadata
            if result["content_type"] == "image":
                enhanced_result["image_info"] = {
                    "image_path": result["metadata"].get("image_path", ""),
                    "image_type": result["metadata"].get("image_type", "unknown"),
                    "contains_math": result["metadata"].get("contains_math", False),
                    "contains_code": result["metadata"].get("contains_code", False)
                }
            elif result["content_type"] == "code":
                enhanced_result["code_info"] = {
                    "language": result["metadata"].get("language", "unknown"),
                    "confidence": result["metadata"].get("confidence", 0)
                }
            elif result["content_type"] == "formula":
                enhanced_result["formula_info"] = {
                    "formula_type": result["metadata"].get("formula_type", "unknown"),
                    "confidence": result["metadata"].get("confidence", 0)
                }
            
            enhanced_results.append(enhanced_result)
        
        return JSONResponse({
            "query": request.query,
            "results": enhanced_results,
            "total_found": len(enhanced_results),
            "search_summary": {
                "text_results": len([r for r in results if r["content_type"] == "text"]),
                "image_results": len([r for r in results if r["content_type"] == "image"]),
                "code_results": len([r for r in results if r["content_type"] == "code"]),
                "formula_results": len([r for r in results if r["content_type"] == "formula"])
            }
        })
    
    except Exception as e:
        raise HTTPException(500, f"Error searching documents: {str(e)}")

@app.post("/generate-prerequisites/")
async def generate_prerequisites(request: PrerequisiteRequest):
    """Generate comprehensive prerequisite document with AI"""
    try:
        print(f"Generating prerequisites for: {request.query}")
        print(f"Target level: {request.target_level}")
        
        # Generate prerequisites using enhanced system
        result = prereq_generator.generate_prerequisites(
            query=request.query,
            target_level=request.target_level,
            include_images=request.include_images
        )
        
        if "error" in result:
            return JSONResponse({
                "status": "error",
                "message": result["error"],
                "suggestions": result.get("suggestions", "Please upload relevant educational materials first")
            })
        
        # Return comprehensive result
        return JSONResponse({
            "status": "success",
            "message": "Prerequisite document generated successfully",
            "files": {
                "pdf_path": result["pdf_path"],
                "html_path": result["html_path"],
                "pdf_url": f"/generated_docs/{os.path.basename(result['pdf_path'])}",
                "html_url": f"/generated_docs/{os.path.basename(result['html_path'])}"
            },
            "content_summary": {
                "sections_count": len(result["content"]["sections"]),
                "source_documents": result["source_documents"],
                "content_stats": result["stats"]
            },
            "ai_analysis": {
                "prerequisite_topics": result["analysis"]["prerequisite_topics"],
                "difficulty_assessment": result["analysis"]["difficulty_level"],
                "learning_time_estimate": result["analysis"].get("estimated_time", "2-4 weeks")
            },
            "preview":_generate_preview(result["content"])
        })
    
    except Exception as e:
        print(f"Error generating prerequisites: {e}")
        raise HTTPException(500, f"Error generating prerequisites: {str(e)}")

def _generate_preview(content: Dict) -> str:
    """Generate a preview of the prerequisite document"""
    preview = f"# {content['title']}\n\n"
    
    for section in content['sections'][:3]:  # First 3 sections
        preview += f"## {section['title']}\n"
        
        if section['type'] == 'overview':
            preview += f"{section['content']['overview_text'][:200]}...\n\n"
        elif section['type'] == 'text' and isinstance(section['content'], list):
            for item in section['content'][:2]:  # First 2 items
                if isinstance(item, dict):
                    preview += f"- {item.get('title', str(item)[:100])}\n"
        
        preview += "\n"
    
    return preview

@app.post("/chat/")
async def chat_with_documents(request: ChatRequest):
    """AI-powered chat with uploaded documents"""
    try:
        print(f"Chat query: {request.query}")
        
        # Generate response using RAG system
        response = rag_system.generate_chat_response(
            request.query,
            request.conversation_history
        )
        
        return JSONResponse({
            "response": response,
            "query": request.query,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        raise HTTPException(500, f"Error processing chat: {str(e)}")

@app.get("/documents/")
async def list_documents():
    """List all processed documents with detailed statistics"""
    try:
        documents = rag_system.get_document_stats()
        
        return JSONResponse({
            "documents": documents["by_document"],
            "summary": {
                "total_documents": documents["total_documents"],
                "total_content": documents["content_counts"],
                "processing_info": documents["processing_info"]
            }
        })
    
    except Exception as e:
        raise HTTPException(500, f"Error listing documents: {str(e)}")

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a specific document and all its content"""
    try:
        # Find document filename first
        stats = rag_system.get_document_stats()
        doc_info = None
        
        for filename, info in stats["by_document"].items():
            if info["document_id"] == doc_id:
                doc_info = {"filename": filename, **info}
                break
        
        if not doc_info:
            raise HTTPException(404, f"Document {doc_id} not found")
        
        # Delete from vector database
        success = rag_system.delete_document(doc_id)
        
        if success:
            # Also delete original file if it exists
            file_path = os.path.join(UPLOAD_DIR, doc_info["filename"])
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return JSONResponse({
                "message": f"Document '{doc_info['filename']}' deleted successfully",
                "deleted_content": {
                    "text_chunks": doc_info["text_chunks"],
                    "images": doc_info["images"],
                    "code_blocks": doc_info["code_blocks"],
                    "formulas": doc_info["formulas"]
                }
            })
        else:
            raise HTTPException(500, "Failed to delete document from database")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error deleting document: {str(e)}")

@app.get("/analyze/{doc_id}")
async def analyze_document(doc_id: str):
    """Get detailed analysis of a specific document"""
    try:
        # Get document content
        content = rag_system.get_document_content(doc_id)
        
        if not any(content.values()):
            raise HTTPException(404, f"Document {doc_id} not found or has no content")
        
        # Analyze content
        analysis = {
            "document_id": doc_id,
            "content_breakdown": {
                "text_chunks": len(content["text"]),
                "images": len(content["images"]),
                "code_blocks": len(content["code"]),
                "formulas": len(content["formulas"])
            },
            "content_analysis": {
                "primary_topics": [],
                "programming_languages": [],
                "mathematical_concepts": [],
                "visual_content_types": []
            }
        }
        
        # Analyze text content for topics
        if content["text"]:
            text_sample = " ".join([item["content"][:200] for item in content["text"][:5]])
            # Simple keyword extraction (could be enhanced with NLP)
            common_words = {}
            for word in text_sample.lower().split():
                if len(word) > 4 and word.isalpha():
                    common_words[word] = common_words.get(word, 0) + 1
            
            analysis["content_analysis"]["primary_topics"] = [
                word for word, count in sorted(common_words.items(), key=lambda x: x[1], reverse=True)[:10]
            ]
        
        # Analyze programming languages
        if content["code"]:
            languages = {}
            for code_item in content["code"]:
                lang = code_item["metadata"].get("language", "unknown")
                languages[lang] = languages.get(lang, 0) + 1
            
            analysis["content_analysis"]["programming_languages"] = list(languages.keys())
        
        # Analyze mathematical content
        if content["formulas"]:
            math_types = {}
            for formula_item in content["formulas"]:
                formula_type = formula_item["metadata"].get("formula_type", "unknown")
                math_types[formula_type] = math_types.get(formula_type, 0) + 1
            
            analysis["content_analysis"]["mathematical_concepts"] = list(math_types.keys())
        
        # Analyze visual content
        if content["images"]:
            image_types = {}
            for image_item in content["images"]:
                img_type = image_item["metadata"].get("image_type", "unknown")
                image_types[img_type] = image_types.get(img_type, 0) + 1
            
            analysis["content_analysis"]["visual_content_types"] = list(image_types.keys())
        
        return JSONResponse(analysis)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error analyzing document: {str(e)}")

@app.get("/download/{file_type}/{filename}")
async def download_file(file_type: str, filename: str):
    """Download generated files"""
    try:
        if file_type == "prerequisite":
            file_path = os.path.join("generated_docs", filename)
        else:
            raise HTTPException(400, f"Invalid file type: {file_type}")
        
        if not os.path.exists(file_path):
            raise HTTPException(404, f"File not found: {filename}")
        
        return FileResponse(
            file_path,
            media_type="application/octet-stream",
            filename=filename
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error downloading file: {str(e)}")

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    try:
        # Test basic functionality
        stats = rag_system.get_document_stats()
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "enhanced_rag_system": "active",
                "prerequisite_generator": "active",
                "vector_database": "active",
                "openai_integration": "active" if os.getenv('OPENAI_API_KEY') else "missing_api_key"
            },
            "system_stats": {
                "total_documents": stats["total_documents"],
                "total_content_items": sum(stats["content_counts"].values()),
                "embedding_model": stats["processing_info"]["embedding_model"]
            },
            "storage": {
                "upload_dir": os.path.exists(UPLOAD_DIR),
                "vector_db_dir": os.path.exists(VECTOR_DB_DIR),
                "generated_docs_dir": os.path.exists("generated_docs")
            }
        }
        
        return JSONResponse(health_status)
    
    except Exception as e:
        return JSONResponse({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }, status_code=500)

@app.get("/api/stats")
async def get_system_stats():
    """Get detailed system statistics"""
    try:
        stats = rag_system.get_document_stats()
        
        # Add file system stats
        upload_files = len(os.listdir(UPLOAD_DIR)) if os.path.exists(UPLOAD_DIR) else 0
        generated_files = len(os.listdir("generated_docs")) if os.path.exists("generated_docs") else 0
        
        return JSONResponse({
            "database_stats": stats,
            "file_system": {
                "uploaded_files": upload_files,
                "generated_documents": generated_files,
                "disk_usage": {
                    "upload_dir":   _get_dir_size(UPLOAD_DIR),
                    "vector_db_dir":   _get_dir_size(VECTOR_DB_DIR),
                    "generated_docs_dir": _get_dir_size("generated_docs")
                }
            },
            "api_status": {
                "openai_configured": bool(os.getenv('OPENAI_API_KEY')),
                "tesseract_available":   _check_tesseract(),
                "embedding_model_loaded": True
            }
        })
    
    except Exception as e:
        raise HTTPException(500, f"Error getting system stats: {str(e)}")

def _get_dir_size(directory: str) -> str:
    """Get directory size in human readable format"""
    if not os.path.exists(directory):
        return "0 B"
    
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    
    # Convert to human readable
    for unit in ['B', 'KB', 'MB', 'GB']:
        if total_size < 1024:
            return f"{total_size:.1f} {unit}"
        total_size /= 1024
    return f"{total_size:.1f} TB"

def _check_tesseract() -> bool:
    """Check if Tesseract OCR is available"""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except:
        return False

# Add import for datetime
from datetime import datetime

if __name__ == "__main__":
    print("🚀 Starting Enhanced EdTech RAG System...")
    print(f"📁 Upload directory: {UPLOAD_DIR}")
    print(f"🗄️  Vector database: {VECTOR_DB_DIR}")
    print(f"🤖 OpenAI API key configured: {bool(os.getenv('OPENAI_API_KEY'))}")
    print(f"👁️  Tesseract OCR available: {_check_tesseract()}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )