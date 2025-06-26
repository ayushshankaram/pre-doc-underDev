import os
import json
from typing import Dict, List, Any, Optional
from fpdf import FPDF
from datetime import datetime
import google.generativeai as genai
from PIL import Image
import base64
import re
import io

class EnhancedPrerequisiteGenerator:
    def __init__(self, rag_system, openai_api_key: str = None):
        self.rag_system = rag_system
        self.output_dir = "generated_docs"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configure Gemini API
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model = genai.GenerativeModel('gemini-2.0-flash')
     
    def _clean_text_for_pdf(self, text: str) -> str:
        """Clean text for PDF generation by replacing Unicode characters"""
        if not text:
            return ""
        
        import unicodedata
        # Normalize text to composed form
        text = unicodedata.normalize('NFKC', text)
        
        # Replace specific Unicode characters
        replacements = {
            '\u2018': "'",   # Left single quote
            '\u2019': "'",   # Right single quote
            '\u201C': '"',   # Left double quote
            '\u201D': '"',   # Right double quote
            '\u2013': '-',   # En dash
            '\u2014': '-',   # Em dash
            '\u2026': '...', # Ellipsis
            '\u2022': '-',   # Bullet point
            '\u2605': '*',   # Star
            '\u2606': 'o',   # Empty star
            '\u2610': '[ ]', # Empty checkbox
            '\u2705': '[x]', # Check mark
            '\u274C': '[x]', # Cross mark
            '\u00B0': ' deg',  # Degree symbol
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove any remaining non-ASCII characters
        text = re.sub(r'[^\x00-\x7F]', ' ', text)
        return text
    
    def generate_prerequisites(self, query: str, target_level: str = "undergraduate", 
                             include_images: bool = True) -> Dict[str, Any]:
        """Generate comprehensive prerequisite document using AI and RAG"""
        
        print(f"Generating prerequisites for: {query}")
        print(f"Target level: {target_level}")
        
        # Step 1: Get AI-powered prerequisite analysis
        prerequisite_analysis = self.rag_system.generate_prerequisites_with_ai(
            query, target_level
        )
        
        if "error" in prerequisite_analysis:
            return prerequisite_analysis
        
        # Step 2: Generate comprehensive prerequisite document
        doc_content = self._create_comprehensive_document(
            query, prerequisite_analysis, include_images
        )
        
        # Step 3: Create PDF document
        pdf_path = self._create_pdf_document(doc_content, query, target_level)
        
        # Step 4: Create interactive HTML version
        html_path = self._create_html_document(doc_content, query, target_level)
        
        return {
            "status": "success",
            "pdf_path": pdf_path,
            "html_path": html_path,
            "content": doc_content,
            "analysis": prerequisite_analysis["analysis"],
            "source_documents": prerequisite_analysis["source_documents"],
            "stats": prerequisite_analysis["content_stats"]
        }
    
    def _create_comprehensive_document(self, query: str, analysis: Dict, include_images: bool) -> Dict[str, Any]:
        """Create comprehensive document structure with all content types"""
        
        doc_content = {
            "title": f"Prerequisites for {query}",
            "sections": []
        }
        
        # Section 1: Overview and Learning Objectives
        overview_section = {
            "title": "Overview and Learning Objectives",
            "type": "overview",
            "content": self._generate_overview(query, analysis)
        }
        doc_content["sections"].append(overview_section)
        
        # Section 2: Fundamental Concepts
        if analysis["structured_content"]["sections"].get("Fundamental Concepts"):
            fundamental_section = {
                "title": "Fundamental Concepts",
                "type": "text",
                "content": analysis["structured_content"]["sections"]["Fundamental Concepts"]["content"]
            }
            doc_content["sections"].append(fundamental_section)
        
        # Section 3: Mathematical Prerequisites
        if analysis["structured_content"]["sections"].get("Mathematical Prerequisites"):
            math_section = {
                "title": "Mathematical Prerequisites",
                "type": "mathematics",
                "content": analysis["structured_content"]["sections"]["Mathematical Prerequisites"]["content"]
            }
            doc_content["sections"].append(math_section)
        
        # Section 4: Programming Prerequisites (if applicable)
        if analysis["structured_content"]["sections"].get("Programming Examples"):
            programming_section = {
                "title": "Programming Prerequisites",
                "type": "programming",
                "content": analysis["structured_content"]["sections"]["Programming Examples"]["content"]
            }
            doc_content["sections"].append(programming_section)
        
        # Section 5: Visual Learning Materials
        if include_images and analysis["structured_content"]["sections"].get("Visual Learning Materials"):
            visual_section = {
                "title": "Visual Learning Materials",
                "type": "visual",
                "content": analysis["structured_content"]["sections"]["Visual Learning Materials"]["content"]
            }
            doc_content["sections"].append(visual_section)
        
        # Section 6: Recommended Learning Path
        learning_path_section = {
            "title": "Recommended Learning Path",
            "type": "learning_path",
            "content": self._generate_learning_path(analysis)
        }
        doc_content["sections"].append(learning_path_section)
        
        # Section 7: Self-Assessment Checklist
        checklist_section = {
            "title": "Self-Assessment Checklist",
            "type": "checklist",
            "content": self._generate_checklist(analysis)
        }
        doc_content["sections"].append(checklist_section)
        
        return doc_content
    
    def _generate_overview(self, query: str, analysis: Dict) -> Dict[str, Any]:
        """Generate overview section using AI"""
        
        try:
            prompt = f"""
            Create a comprehensive overview for a prerequisite document about "{query}".
            
            Based on the analysis:
            {analysis['analysis']['ai_analysis'][:1000]}
            
            Include:
            1. What this topic is about
            2. Why these prerequisites are important
            3. What students will be able to do after mastering the prerequisites
            4. Estimated time to complete prerequisites
            
            Make it engaging and educational.
            """
            
            response = self.model.generate_content(prompt)
            
            return {
                "overview_text": response.text,
                "learning_objectives": analysis["analysis"]["prerequisite_topics"][:5],
                "estimated_time": self._estimate_learning_time(analysis),
                "difficulty_assessment": analysis["analysis"]["difficulty_level"]
            }
        
        except Exception as e:
            print(f"Error generating overview: {e}")
            return {
                "overview_text": f"This document outlines the prerequisite knowledge needed for {query}.",
                "learning_objectives": analysis["analysis"]["prerequisite_topics"][:5],
                "estimated_time": "2-4 weeks",
                "difficulty_assessment": analysis["analysis"]["difficulty_level"]
            }
    
    def _generate_learning_path(self, analysis: Dict) -> List[Dict[str, Any]]:
        """Generate recommended learning path"""
        
        learning_sequence = analysis["analysis"]["learning_sequence"]
        prerequisite_topics = analysis["analysis"]["prerequisite_topics"]
        
        # Create learning path steps
        learning_path = []
        
        # Step 1: Foundation
        foundation_topics = [topic for topic in prerequisite_topics 
                           if any(word in topic.lower() for word in ['basic', 'fundamental', 'introduction'])]
        
        if foundation_topics:
            learning_path.append({
                "step": 1,
                "title": "Foundation Knowledge",
                "topics": foundation_topics,
                "description": "Master these fundamental concepts first",
                "estimated_time": "1-2 weeks"
            })
        
        # Step 2: Core Concepts
        core_topics = [topic for topic in prerequisite_topics 
                      if topic not in foundation_topics and 
                      not any(word in topic.lower() for word in ['advanced', 'complex'])]
        
        if core_topics:
            learning_path.append({
                "step": 2,
                "title": "Core Concepts",
                "topics": core_topics,
                "description": "Build upon foundation with these core concepts",
                "estimated_time": "2-3 weeks"
            })
        
        # Step 3: Advanced Topics
        advanced_topics = [topic for topic in prerequisite_topics 
                          if any(word in topic.lower() for word in ['advanced', 'complex', 'deep'])]
        
        if advanced_topics:
            learning_path.append({
                "step": 3,
                "title": "Advanced Prerequisites",
                "topics": advanced_topics,
                "description": "Complete preparation with advanced topics",
                "estimated_time": "1-2 weeks"
            })
        
        return learning_path
    
    def _generate_checklist(self, analysis: Dict) -> List[Dict[str, Any]]:
        """Generate self-assessment checklist"""
        
        checklist_items = []
        
        # From prerequisite topics
        for i, topic in enumerate(analysis["analysis"]["prerequisite_topics"][:10]):
            checklist_items.append({
                "id": f"check_{i}",
                "item": f"I understand {topic.lower()}",
                "category": "concept",
                "importance": "high" if i < 5 else "medium"
            })
        
        # Add skill-based items
        skill_items = [
            "I can explain key concepts in my own words",
            "I can solve basic problems in this area",
            "I understand the mathematical foundations",
            "I can read and interpret relevant diagrams/graphs",
            "I am familiar with common terminology"
        ]
        
        for i, skill in enumerate(skill_items):
            checklist_items.append({
                "id": f"skill_{i}",
                "item": skill,
                "category": "skill",
                "importance": "high"
            })
        
        return checklist_items
    
    def _estimate_learning_time(self, analysis: Dict) -> str:
        """Estimate learning time based on complexity"""
        
        difficulty = analysis["analysis"]["difficulty_level"]["assessed_level"]
        num_topics = len(analysis["analysis"]["prerequisite_topics"])
        
        if difficulty == "beginner":
            base_time = 1
        elif difficulty == "intermediate":
            base_time = 2
        elif difficulty == "advanced":
            base_time = 3
        else:
            base_time = 4
        
        # Adjust based on number of topics
        total_weeks = base_time + (num_topics // 5)
        
        if total_weeks <= 2:
            return "1-2 weeks"
        elif total_weeks <= 4:
            return "2-4 weeks"
        elif total_weeks <= 6:
            return "4-6 weeks"
        else:
            return "6+ weeks"
    
    def _create_pdf_document(self, doc_content: Dict, query: str, target_level: str) -> str:
        """Create PDF version of prerequisite document"""
        # Clean inputs before use
        clean_query = self._clean_text_for_pdf(query)
        clean_target_level = self._clean_text_for_pdf(target_level)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"prerequisites_{clean_query.replace(' ', '_')}_{timestamp}.pdf"
        pdf_path = os.path.join(self.output_dir, pdf_filename)
        
        pdf = FPDF()
        pdf.add_page()
        
        # Use standard font that supports basic characters
        pdf.set_font('Arial', 'B', 16)
        
        # Clean title before use
        clean_title = self._clean_text_for_pdf(doc_content["title"])
        pdf.cell(0, 20, clean_title, ln=True, align='C')
        
        # Clean all text elements
        pdf.set_font('Arial', '', 12)
        target_text = self._clean_text_for_pdf(f"Target Level: {target_level.title()}")
        pdf.cell(0, 10, target_text, ln=True, align='C')
        
        date_text = self._clean_text_for_pdf(f"Generated: {datetime.now().strftime('%B %d, %Y')}")
        pdf.cell(0, 10, date_text, ln=True, align='C')
        pdf.ln(20)
        
        # Process each section
        for section in doc_content["sections"]:
            self._add_section_to_pdf(pdf, section)
        
        pdf.output(pdf_path)
        return pdf_path
    
    def _add_section_to_pdf(self, pdf: FPDF, section: Dict):
        """Add a section to the PDF"""
        # Clean and add section title
        clean_title = self._clean_text_for_pdf(section["title"])
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 12, clean_title, ln=True)
        pdf.ln(5)
        
        if section["type"] == "overview":
            self._add_overview_to_pdf(pdf, section["content"])
        elif section["type"] == "text":
            self._add_text_content_to_pdf(pdf, section["content"])
        elif section["type"] == "mathematics":
            self._add_math_content_to_pdf(pdf, section["content"])
        elif section["type"] == "programming":
            self._add_code_content_to_pdf(pdf, section["content"])
        elif section["type"] == "visual":
            self._add_visual_content_to_pdf(pdf, section["content"])
        elif section["type"] == "learning_path":
            self._add_learning_path_to_pdf(pdf, section["content"])
        elif section["type"] == "checklist":
            self._add_checklist_to_pdf(pdf, section["content"])
        
        pdf.ln(10)
    
    def _add_overview_to_pdf(self, pdf: FPDF, content: Dict):
        """Add overview content to PDF"""
        pdf.set_font('Arial', '', 10)
        
        # Clean overview text
        overview_text = self._clean_text_for_pdf(content["overview_text"])
        overview_lines = self._wrap_text(overview_text, 90)
        for line in overview_lines:
            pdf.cell(0, 6, line, ln=True)
        
        pdf.ln(5)
        
        # Learning objectives
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, "Learning Objectives:", ln=True)
        pdf.set_font('Arial', '', 10)
        
        for obj in content["learning_objectives"]:
            clean_obj = self._clean_text_for_pdf(f"- {obj}")
            pdf.cell(0, 6, clean_obj, ln=True)
        
        pdf.ln(3)
        
        # Time estimate
        pdf.set_font('Arial', 'B', 10)
        estimated_time_text = self._clean_text_for_pdf(f"Estimated Time: {content['estimated_time']}")
        pdf.cell(0, 6, estimated_time_text, ln=True)
    
    def _add_text_content_to_pdf(self, pdf: FPDF, content: List[Dict]):
        """Add text content to PDF"""
        pdf.set_font('Arial', '', 10)
        
        for item in content:
            # Clean title
            clean_title = self._clean_text_for_pdf(item["title"])
            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 8, clean_title, ln=True)
            
            # Clean description
            clean_desc = self._clean_text_for_pdf(item["description"])
            pdf.set_font('Arial', '', 10)
            desc_lines = self._wrap_text(clean_desc, 90)
            for line in desc_lines[:3]:
                pdf.cell(0, 5, line, ln=True)
            
            # Source reference
            pdf.set_font('Arial', 'I', 9)
            source_text = self._clean_text_for_pdf(f"Source: {item['source']}, Page {item['page']}")
            pdf.cell(0, 5, source_text, ln=True)
            pdf.ln(3)
    
    def _add_math_content_to_pdf(self, pdf: FPDF, content: List[Dict]):
        """Add mathematical content to PDF"""
        pdf.set_font('Arial', '', 10)
        
        for item in content:
            # Formula
            clean_formula = self._clean_text_for_pdf(item["formula"])
            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 8, f"Formula: {clean_formula}", ln=True)
            
            # Explanation
            clean_explanation = self._clean_text_for_pdf(item["explanation"])
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 6, clean_explanation, ln=True)
            
            # Source
            pdf.set_font('Arial', 'I', 9)
            source_text = self._clean_text_for_pdf(f"From: {item['source']}, Page {item['page']}")
            pdf.cell(0, 5, source_text, ln=True)
            pdf.ln(3)
    
    def _add_code_content_to_pdf(self, pdf: FPDF, content: List[Dict]):
        """Add code content to PDF"""
        for item in content:
            # Language and source
            pdf.set_font('Arial', 'B', 11)
            language_text = self._clean_text_for_pdf(f"{item['language'].title()} Example:")
            pdf.cell(0, 8, language_text, ln=True)
            
            # Code (simplified for PDF)
            pdf.set_font('Courier', '', 9)
            code_lines = item["code"].split('\n')[:5]  # Limit lines
            for line in code_lines:
                clean_line = self._clean_text_for_pdf(line[:80])
                pdf.cell(0, 4, clean_line, ln=True)
            
            # Source
            pdf.set_font('Arial', 'I', 9)
            source_text = self._clean_text_for_pdf(f"From: {item['source']}, Page {item['page']}")
            pdf.cell(0, 5, source_text, ln=True)
            pdf.ln(3)
    
    def _add_visual_content_to_pdf(self, pdf: FPDF, content: List[Dict]):
        """Add visual content references to PDF"""
        pdf.set_font('Arial', '', 10)
        
        for item in content:
            # Image description
            pdf.set_font('Arial', 'B', 11)
            title_text = self._clean_text_for_pdf(f"{item['type'].title()} - Visual Material")
            pdf.cell(0, 8, title_text, ln=True)
            
            # Description
            pdf.set_font('Arial', '', 10)
            clean_desc = self._clean_text_for_pdf(item["description"])
            desc_lines = self._wrap_text(clean_desc, 90)
            for line in desc_lines[:2]:
                pdf.cell(0, 5, line, ln=True)
            
            # Note about image
            pdf.set_font('Arial', 'I', 9)
            note_text = self._clean_text_for_pdf(f"[Image available in digital version] - From: {item['source']}")
            pdf.cell(0, 5, note_text, ln=True)
            pdf.ln(3)
    
    def _add_learning_path_to_pdf(self, pdf: FPDF, content: List[Dict]):
        """Add learning path to PDF"""
        pdf.set_font('Arial', '', 10)
        
        for step in content:
            # Step title
            clean_step_title = self._clean_text_for_pdf(f"Step {step['step']}: {step['title']}")
            clean_description = self._clean_text_for_pdf(step["description"])
            clean_time = self._clean_text_for_pdf(f"Estimated time: {step['estimated_time']}")
            
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, clean_step_title, ln=True)
            
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 6, clean_description, ln=True)
            
            pdf.set_font('Arial', 'I', 10)
            pdf.cell(0, 5, clean_time, ln=True)
            
            pdf.set_font('Arial', '', 10)
            for topic in step["topics"][:3]:
                clean_topic = self._clean_text_for_pdf(f"- {topic}")
                pdf.cell(0, 5, clean_topic, ln=True)
            
            pdf.ln(3)
    
    def _add_checklist_to_pdf(self, pdf: FPDF, content: List[Dict]):
        """Add checklist to PDF"""
        pdf.set_font('Arial', '', 10)
        
        for item in content:
            clean_item = self._clean_text_for_pdf(item['item'])
            checkbox = "[ ]"
            importance = "*" if item["importance"] == "high" else "o"
            pdf.cell(0, 6, f"{checkbox} {clean_item} {importance}", ln=True)
    
    def _wrap_text(self, text: str, width: int) -> List[str]:
        """Wrap text to specified width"""
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + word) <= width:
                current_line += word + " "
            else:
                if current_line:
                    lines.append(current_line.strip())
                current_line = word + " "
        
        if current_line:
            lines.append(current_line.strip())
        
        return lines
    
    def _create_html_document(self, doc_content: Dict, query: str, target_level: str) -> str:
        """Create interactive HTML version of the document"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_filename = f"prerequisites_{query.replace(' ', '_')}_{timestamp}.html"
        html_path = os.path.join(self.output_dir, html_filename)
        
        html_content = self._generate_html_content(doc_content, query, target_level)
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_path
    
    def _generate_html_content(self, doc_content: Dict, query: str, target_level: str) -> str:
        """Generate HTML content for the prerequisite document"""
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{doc_content['title']}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            background: #f5f7fa;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 2.2rem;
        }}
        
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .section h2 {{
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        
        .concept {{
            margin-bottom: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            border-radius: 5px;
        }}
        
        .concept h3 {{
            color: #667eea;
            margin-bottom: 10px;
        }}
        
        .source-ref {{
            font-size: 0.9em;
            color: #666;
            font-style: italic;
            margin-top: 10px;
        }}
        
        .formula {{
            background: #fff3cd;
            padding: 15px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            margin: 10px 0;
            border-left: 4px solid #ffc107;
        }}
        
        .code-block {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            margin: 10px 0;
            border-left: 4px solid #28a745;
            overflow-x: auto;
        }}
        
        .learning-step {{
            background: #e8f4fd;
            padding: 20px;
            margin: 15px 0;
            border-radius: 8px;
            border-left: 5px solid #007bff;
        }}
        
        .learning-step h3 {{
            color: #007bff;
            margin-bottom: 10px;
        }}
        
        .checklist {{
            list-style: none;
            padding: 0;
        }}
        
        .checklist li {{
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }}
        
        .checklist input[type="checkbox"] {{
            margin-right: 10px;
            transform: scale(1.2);
        }}
        
        .image-gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .image-item {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .image-placeholder {{
            width: 100%;
            height: 200px;
            background: #f0f0f0;
            border: 2px dashed #ccc;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 10px;
            border-radius: 5px;
        }}
        
        .progress-tracker {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            z-index: 1000;
        }}
        
        .progress-bar {{
            width: 200px;
            height: 10px;
            background: #e0e0e0;
            border-radius: 5px;
            overflow: hidden;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(45deg, #667eea, #764ba2);
            width: 0%;
            transition: width 0.3s ease;
        }}
        
        @media (max-width: 768px) {{
            .progress-tracker {{
                position: relative;
                top: auto;
                right: auto;
                margin-bottom: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="progress-tracker">
        <div style="font-size: 0.9em; color: #666; margin-bottom: 5px;">Reading Progress</div>
        <div class="progress-bar">
            <div class="progress-fill" id="progressFill"></div>
        </div>
        <div style="font-size: 0.8em; color: #666; margin-top: 5px;" id="progressText">0%</div>
    </div>
    
    <div class="header">
        <h1>{doc_content['title']}</h1>
        <p>Target Level: {target_level.title()} | Generated: {datetime.now().strftime('%B %d, %Y')}</p>
    </div>
"""
        
        # Add sections
        for section in doc_content['sections']:
            html += self._generate_section_html(section)
        
        # Add JavaScript for interactivity
        html += """
    <script>
        // Progress tracking
        function updateProgress() {
            const scrolled = (window.scrollY / (document.documentElement.scrollHeight - window.innerHeight)) * 100;
            document.getElementById('progressFill').style.width = scrolled + '%';
            document.getElementById('progressText').textContent = Math.round(scrolled) + '%';
        }
        
        window.addEventListener('scroll', updateProgress);
        
        // Checklist functionality
        document.querySelectorAll('.checklist input[type="checkbox"]').forEach(checkbox => {
            checkbox.addEventListener('change', function() {
                const completed = document.querySelectorAll('.checklist input[type="checkbox"]:checked').length;
                const total = document.querySelectorAll('.checklist input[type="checkbox"]').length;
                
                if (completed === total) {
                    alert('Congratulations! You have completed all prerequisite checks. You are ready to proceed with the main topic.');
                }
            });
        });
        
        // Smooth scrolling for internal links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({
                    behavior: 'smooth'
                });
            });
        });
    </script>
</body>
</html>
"""
        
        return html
    
    def _generate_section_html(self, section: Dict) -> str:
        """Generate HTML for a specific section"""
        
        html = f'<div class="section"><h2>{section["title"]}</h2>'
        
        if section["type"] == "overview":
            html += self._generate_overview_html(section["content"])
        elif section["type"] == "text":
            html += self._generate_text_html(section["content"])
        elif section["type"] == "mathematics":
            html += self._generate_math_html(section["content"])
        elif section["type"] == "programming":
            html += self._generate_code_html(section["content"])
        elif section["type"] == "visual":
            html += self._generate_visual_html(section["content"])
        elif section["type"] == "learning_path":
            html += self._generate_learning_path_html(section["content"])
        elif section["type"] == "checklist":
            html += self._generate_checklist_html(section["content"])
        
        html += '</div>'
        return html
    
    def _generate_overview_html(self, content: Dict) -> str:
        """Generate HTML for overview section"""
        html = f'<p>{content["overview_text"]}</p>'
        
        html += '<h3>Learning Objectives</h3><ul>'
        for obj in content["learning_objectives"]:
            html += f'<li>{obj}</li>'
        html += '</ul>'
        
        html += f'<p><strong>Estimated Time:</strong> {content["estimated_time"]}</p>'
        
        return html
    
    def _generate_text_html(self, content: List[Dict]) -> str:
        """Generate HTML for text content"""
        html = ''
        
        for item in content:
            html += f'''
            <div class="concept">
                <h3>{item["title"]}</h3>
                <p>{item["description"]}</p>
                <div class="source-ref">Source: {item["source"]}, Page {item["page"]}</div>
            </div>
            '''
        
        return html
    
    def _generate_math_html(self, content: List[Dict]) -> str:
        """Generate HTML for mathematical content"""
        html = ''
        
        for item in content:
            html += f'''
            <div class="formula">
                <strong>Formula:</strong> {item["formula"]}<br>
                <em>{item["explanation"]}</em>
                <div class="source-ref">From: {item["source"]}, Page {item["page"]}</div>
            </div>
            '''
        
        return html
    
    def _generate_code_html(self, content: List[Dict]) -> str:
        """Generate HTML for code content"""
        html = ''
        
        for item in content:
            html += f'''
            <div class="concept">
                <h3>{item["language"].title()} Example</h3>
                <div class="code-block"><pre>{item["code"]}</pre></div>
                <p>{item["explanation"]}</p>
                <div class="source-ref">From: {item["source"]}, Page {item["page"]}</div>
            </div>
            '''
        
        return html
    
    def _generate_visual_html(self, content: List[Dict]) -> str:
        """Generate HTML for visual content"""
        html = '<div class="image-gallery">'
        for item in content:
            image_path = item["image_path"]
            web_image_path = f"/vector_db/extracted_images/{os.path.basename(image_path)}"
        
        # Check if image file exists
            image_exists = os.path.exists(image_path)
        
            html += f'''
            <div class="image-item">
                <div class="image-placeholder">
                    {'<img src="' + web_image_path + '" style="max-width: 100%; max-height: 100%;" alt="Visual content" onerror="this.style.display=''">' if image_exists else f'{item["type"].title()} - Visual Material'}
                 </div>
                 <h4>{item["type"].title()}</h4>
                 <p>{item["description"]}</p>
                 <div class="source-ref">From: {item["source"]}, Page {item["page"]}</div>
            </div> 
            '''
        html += '</div>'
        return html
      

    def _generate_learning_path_html(self, content: List[Dict]) -> str:
        """Generate HTML for learning path"""
        html = ''
        
        for step in content:
            html += f'''
            <div class="learning-step">
                <h3>Step {step["step"]}: {step["title"]}</h3>
                <p>{step["description"]}</p>
                <p><strong>Estimated time:</strong> {step["estimated_time"]}</p>
                <h4>Topics to master:</h4>
                <ul>
            '''
            
            for topic in step["topics"]:
                html += f'<li>{topic}</li>'
            
            html += '</ul></div>'
        
        return html
    
    def _generate_checklist_html(self, content: List[Dict]) -> str:
        """Generate HTML for checklist"""
        html = '<ul class="checklist">'
        
        for item in content:
            importance_star = "⭐" if item["importance"] == "high" else "☆"
            html += f'''
            <li>
                <input type="checkbox" id="{item['id']}">
                <label for="{item['id']}">{item['item']} {importance_star}</label>
            </li>
            '''
        
        html += '</ul>'
        return html
    
    def chat_with_prerequisites(self, query: str, prerequisite_context: Dict = None) -> str:
        """Chat about prerequisites using the generated content"""
        
        if prerequisite_context:
            # Use prerequisite context for more targeted responses
            context = f"Based on the prerequisite analysis for {prerequisite_context.get('title', 'the topic')}:\n"
            
            # Add relevant sections to context
            for section in prerequisite_context.get('sections', []):
                if section['type'] in ['text', 'mathematics', 'programming']:
                    context += f"\n{section['title']}:\n"
                    if isinstance(section['content'], list):
                        for item in section['content'][:2]:  # Limit context
                            if isinstance(item, dict):
                                context += f"- {item.get('title', item.get('formula', str(item)[:100]))}\n"
        else:
            # Fall back to general RAG search
            return self.rag_system.generate_chat_response(query)
        
        try:
            prompt = f"""
            Answer the question about prerequisites: "{query}"
            
            Context:
            {context}
            
            Provide a helpful response that:
            1. Directly answers the question
            2. References the prerequisite content when relevant
            3. Suggests next steps or related topics
            4. Maintains an educational tone
            """
            
            full_prompt = f"You are an AI tutor helping students understand prerequisite knowledge for their courses.\n\n{prompt}"
            response = self.model.generate_content(full_prompt)
            return response.text
        
        except Exception as e:
            print(f"Chat error: {e}")
            return "I'm having trouble processing your question right now. Please try rephrasing or ask about specific prerequisite topics."