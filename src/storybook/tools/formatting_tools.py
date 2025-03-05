from typing import Dict, Any, List
from langchain_core.tools import BaseTool, tool
from datetime import datetime
import uuid

from db_config import get_collection, COLLECTIONS

@tool
def format_validator(project_id: str, format_type: str = "manuscript") -> Dict[str, Any]:
    """Check if document meets industry format standards.
    
    Args:
        project_id: The project ID.
        format_type: Type of format to check (manuscript, query, synopsis).
        
    Returns:
        Format validation results.
    """
    format_collection = get_collection("format_validation")
    
    # Define format standards based on type
    standards = {}
    
    if format_type == "manuscript":
        standards = {
            "font": "Times New Roman or Courier New, 12-point",
            "margins": "1 inch (2.54 cm) on all sides",
            "line_spacing": "Double-spaced",
            "paragraph_indent": "0.5 inch (1.27 cm) for first line of each paragraph",
            "header": "Author last name / Title / Page number in upper right",
            "chapter_heading": "Centered, about 1/3 down first page of each chapter",
            "scene_breaks": "Indicated by # or *** centered on a line",
            "italics": "Use actual italics rather than underlines"
        }
    elif format_type == "query":
        standards = {
            "length": "250-350 words",
            "structure": "3-4 paragraphs (hook, synopsis, bio, closing)",
            "addressing": "Personalized to specific agent",
            "content": "Include word count, genre, title, comparable titles",
            "format": "Business letter format with contact information"
        }
    elif format_type == "synopsis":
        standards = {
            "length": "1-2 pages for short synopsis, 3-5 pages for long synopsis",
            "tense": "Present tense",
            "perspective": "Third-person, regardless of manuscript POV",
            "content": "Cover all major plot points and character arcs",
            "spoilers": "Include all major reveals and the ending"
        }
    else:
        return {
            "error": f"Unknown format type: {format_type}",
            "status": "error"
        }
    
    # In a real implementation, this would analyze the actual document
    # Here we're generating placeholder validation results
    
    # Generate validation results
    results = {}
    issues = []
    overall_status = "pass"
    
    for standard, requirement in standards.items():
        # Generate random compliance
        is_compliant = (hash(standard + project_id) % 4 != 0)  # 75% chance of compliance
        
        results[standard] = {
            "requirement": requirement,
            "status": "pass" if is_compliant else "fail",
            "details": "Meets standard" if is_compliant else "Does not meet standard"
        }
        
        if not is_compliant:
            issues.append({
                "standard": standard,
                "requirement": requirement,
                "solution": f"Update {standard} to match required format: {requirement}"
            })
            overall_status = "fail"
    
    # Store validation results in MongoDB
    validation_record = {
        "validation_id": str(uuid.uuid4()),
        "project_id": project_id,
        "format_type": format_type,
        "standards": standards,
        "results": results,
        "issues": issues,
        "overall_status": overall_status,
        "created_at": datetime.utcnow().isoformat()
    }
    
    format_collection.insert_one(validation_record)
    
    return {
        "format_type": format_type,
        "results": results,
        "issues": issues,
        "overall_status": overall_status,
        "validation_id": validation_record["validation_id"]
    }

@tool
def style_guide_formatter(project_id: str, style_type: str = "chicago") -> Dict[str, Any]:
    """Apply standard style guide rules to manuscript.
    
    Args:
        project_id: The project ID.
        style_type: Style guide to apply (chicago, ap, etc.).
        
    Returns:
        Style guide requirements and application status.
    """
    style_collection = get_collection("style_formatting")
    
    # Define style guide rules
    style_guides = {
        "chicago": {
            "citations": "Chicago Manual of Style (CMOS) format for citations",
            "numbers": "Spell out numbers one through one hundred",
            "dates": "Month day, year format (e.g., January 1, 2023)",
            "titles": "Italicize book titles, use quotation marks for articles",
            "punctuation": "Use serial comma (Oxford comma)",
            "dialogue": "Use double quotation marks for dialogue"
        },
        "ap": {
            "citations": "Associated Press (AP) style for citations",
            "numbers": "Spell out numbers one through nine",
            "dates": "AP date format (e.g., Jan. 1, 2023)",
            "titles": "Use quotation marks for book titles and articles",
            "punctuation": "No serial comma unless needed for clarity",
            "dialogue": "Use double quotation marks for dialogue"
        },
        "mla": {
            "citations": "MLA format for citations",
            "numbers": "Follow general guidelines for number formatting",
            "dates": "Day Month Year format (e.g., 1 January 2023)",
            "titles": "Italicize book titles, use quotation marks for articles",
            "punctuation": "Use serial comma (Oxford comma)",
            "dialogue": "Use double quotation marks for dialogue"
        }
    }
    
    # Check if style type exists
    if style_type not in style_guides:
        return {
            "error": f"Unknown style type: {style_type}. Available types: {', '.join(style_guides.keys())}",
            "status": "error"
        }
    
    style_rules = style_guides[style_type]
    
    # In a real implementation, this would analyze the manuscript and apply rules
    # Here we're generating placeholder application results
    
    # Generate application results
    application_results = {}
    issues = []
    
    for rule, requirement in style_rules.items():
        # Generate random compliance
        is_applied = (hash(rule + project_id) % 5 != 0)  # 80% chance of compliance
        
        application_results[rule] = {
            "requirement": requirement,
            "status": "applied" if is_applied else "not_applied",
            "instances": hash(rule + project_id) % 20 + 1  # Random number of instances (1-20)
        }
        
        if not is_applied:
            issues.append({
                "rule": rule,
                "requirement": requirement,
                "recommendation": f"Apply {style_type} style for {rule}: {requirement}"
            })
    
    # Generate examples
    examples = {
        "chicago": {
            "citations": 'According to Smith, "the evidence suggests otherwise" (Smith 2020, 45).',
            "numbers": "The ninety-five participants completed the task in thirty minutes.",
            "dates": "The event occurred on April 15, 2023.",
            "titles": "She read _The Great Gatsby_ and the article \"Modern Literature Trends.\""
        },
        "ap": {
            "citations": "According to Smith, \"the evidence suggests otherwise\" (Smith, 2020).",
            "numbers": "The 95 participants completed the task in 30 minutes.",
            "dates": "The event occurred on April 15, 2023.",
            "titles": "She read \"The Great Gatsby\" and the article \"Modern Literature Trends.\""
        },
        "mla": {
            "citations": "According to Smith, \"the evidence suggests otherwise\" (Smith 45).",
            "numbers": "The ninety-five participants completed the task in thirty minutes.",
            "dates": "The event occurred on 15 April 2023.",
            "titles": "She read _The Great Gatsby_ and the article \"Modern Literature Trends.\""
        }
    }
    
    # Store style guide application in MongoDB
    style_record = {
        "record_id": str(uuid.uuid4()),
        "project_id": project_id,
        "style_type": style_type,
        "style_rules": style_rules,
        "application_results": application_results,
        "issues": issues,
        "examples": examples.get(style_type, {}),
        "created_at": datetime.utcnow().isoformat()
    }
    
    style_collection.insert_one(style_record)
    
    return {
        "style_type": style_type,
        "style_rules": style_rules,
        "application_results": application_results,
        "issues": issues,
        "examples": examples.get(style_type, {}),
        "record_id": style_record["record_id"]
    }

@tool
def ebook_formatter(project_id: str, format_type: str = "epub") -> Dict[str, Any]:
    """Format manuscript for e-book publishing.
    
    Args:
        project_id: The project ID.
        format_type: E-book format (epub, mobi, pdf).
        
    Returns:
        E-book formatting requirements and status.
    """
    ebook_collection = get_collection("ebook_formatting")
    
    # Define format requirements
    format_requirements = {
        "epub": {
            "file_format": "EPUB 3.0 or later",
            "stylesheet": "CSS for formatting",
            "toc": "HTML table of contents",
            "metadata": "Title, author, publisher, publication date, etc.",
            "cover": "JPG or PNG, 1600x2560 pixels recommended",
            "images": "JPG or PNG, optimized for size"
        },
        "mobi": {
            "file_format": "MOBI (Kindle format)",
            "stylesheet": "Limited CSS support",
            "toc": "NCX table of contents",
            "metadata": "Title, author, publisher, publication date, etc.",
            "cover": "JPG or PNG, 1600x2560 pixels recommended",
            "images": "JPG or PNG, optimized for size"
        },
        "pdf": {
            "file_format": "PDF/A for archival quality",
            "fonts": "Embedded fonts",
            "size": "Standard print sizes (5.5\"x8.5\" or 6\"x9\")",
            "margins": "0.5-0.75\" margins",
            "metadata": "Title, author, publisher, publication date, etc.",
            "bookmarks": "PDF bookmarks for chapters"
        }
    }
    
    # Check if format type exists
    if format_type not in format_requirements:
        return {
            "error": f"Unknown format type: {format_type}. Available types: {', '.join(format_requirements.keys())}",
            "status": "error"
        }
    
    requirements = format_requirements[format_type]
    
    # In a real implementation, this would format the manuscript
    # Here we're generating placeholder formatting results
    
    # Generate formatting results
    formatting_results = {}
    issues = []
    
    for element, requirement in requirements.items():
        # Generate random status
        is_formatted = (hash(element + project_id) % 5 != 0)  # 80% chance of formatted
        
        formatting_results[element] = {
            "requirement": requirement,
            "status": "formatted" if is_formatted else "not_formatted",
            "details": "Properly formatted" if is_formatted else "Needs formatting"
        }
        
        if not is_formatted:
            issues.append({
                "element": element,
                "requirement": requirement,
                "solution": f"Format {element} according to {format_type} requirements: {requirement}"
            })
    
    # Generate best practices
    best_practices = {
        "epub": [
            "Use semantic HTML tags for proper structure",
            "Avoid fixed layouts for better device compatibility",
            "Test on multiple e-readers for compatibility",
            "Include proper chapter breaks and navigation",
            "Use responsive design principles"
        ],
        "mobi": [
            "Be aware of Amazon's specific formatting requirements",
            "Use KindleGen or Kindle Previewer for testing",
            "Simplify CSS for better Kindle compatibility",
            "Ensure proper paragraph spacing",
            "Test on multiple Kindle devices"
        ],
        "pdf": [
            "Ensure all fonts are embedded",
            "Use vector graphics when possible",
            "Include proper page numbers and headers/footers",
            "Set up printer marks if intended for printing",
            "Add document properties and metadata"
        ]
    }
    
    # Store formatting results in MongoDB
    formatting_record = {
        "formatting_id": str(uuid.uuid4()),
        "project_id": project_id,
        "format_type": format_type,
        "requirements": requirements,
        "formatting_results": formatting_results,
        "issues": issues,
        "best_practices": best_practices.get(format_type, []),
        "created_at": datetime.utcnow().isoformat()
    }
    
    ebook_collection.insert_one(formatting_record)
    
    return {
        "format_type": format_type,
        "requirements": requirements,
        "formatting_results": formatting_results,
        "issues": issues,
        "best_practices": best_practices.get(format_type, []),
        "formatting_id": formatting_record["formatting_id"]
    }

@tool
def submission_package_formatter(project_id: str, submission_type: str = "agent") -> Dict[str, Any]:
    """Prepare professional submission materials.
    
    Args:
        project_id: The project ID.
        submission_type: Type of submission (agent, publisher, contest).
        
    Returns:
        Submission package requirements and preparation status.
    """
    submission_collection = get_collection("submission_formatting")
    
    # Define submission package requirements
    submission_requirements = {
        "agent": {
            "query_letter": "Professional query letter (250-350 words)",
            "synopsis": "1-2 page synopsis of the entire work",
            "sample_pages": "First 10-50 pages or 3 chapters (varies by agent)",
            "bio": "Brief author bio highlighting relevant experience",
            "metadata": "Include word count, genre, title, comp titles"
        },
        "publisher": {
            "cover_letter": "Professional cover letter introducing the work",
            "synopsis": "Detailed synopsis (3-5 pages)",
            "sample": "Sample chapters or full manuscript (per guidelines)",
            "marketing_plan": "Potential audience and promotion ideas",
            "author_bio": "Detailed author bio and platform information"
        },
        "contest": {
            "entry_form": "Completed entry form for the contest",
            "formatting": "Specific formatting per contest guidelines",
            "anonymity": "No identifying information on manuscript (for blind judging)",
            "word_count": "Adherence to contest word count limits",
            "entry_fee": "Payment of required entry fee"
        }
    }
    
    # Check if submission type exists
    if submission_type not in submission_requirements:
        return {
            "error": f"Unknown submission type: {submission_type}. Available types: {', '.join(submission_requirements.keys())}",
            "status": "error"
        }
    
    requirements = submission_requirements[submission_type]
    
    # In a real implementation, this would prepare the submission package
    # Here we're generating placeholder preparation results
    
    # Generate preparation results
    preparation_results = {}
    missing_elements = []
    
    for element, requirement in requirements.items():
        # Generate random status
        is_prepared = (hash(element + project_id) % 5 != 0)  # 80% chance of prepared
        
        preparation_results[element] = {
            "requirement": requirement,
            "status": "prepared" if is_prepared else "not_prepared",
            "details": "Ready for submission" if is_prepared else "Needs preparation"
        }
        
        if not is_prepared:
            missing_elements.append({
                "element": element,
                "requirement": requirement,
                "priority": "high" if element in ["query_letter", "cover_letter", "entry_form"] else "medium"
            })
    
    # Generate checklist
    checklist = {
        "agent": [
            "Research agent to ensure they represent your genre",
            "Personalize query letter to specific agent",
            "Follow agent's specific submission guidelines",
            "Proofread all materials carefully",
            "Prepare tracking system for submissions"
        ],
        "publisher": [
            "Research publisher to ensure they publish your genre",
            "Follow publisher's specific submission guidelines",
            "Include SASE if submitting by mail",
            "Reference relevant titles from publisher's catalog",
            "Proofread all materials carefully"
        ],
        "contest": [
            "Verify eligibility requirements",
            "Follow all contest rules exactly",
            "Submit before deadline",
            "Keep a copy of all submission materials",
            "Pay entry fee if required"
        ]
    }
    
    # Store preparation results in MongoDB
    submission_record = {
        "submission_id": str(uuid.uuid4()),
        "project_id": project_id,
        "submission_type": submission_type,
        "requirements": requirements,
        "preparation_results": preparation_results,
        "missing_elements": missing_elements,
        "checklist": checklist.get(submission_type, []),
        "created_at": datetime.utcnow().isoformat()
    }
    
    submission_collection.insert_one(submission_record)
    
    return {
        "submission_type": submission_type,
        "requirements": requirements,
        "preparation_results": preparation_results,
        "missing_elements": missing_elements,
        "checklist": checklist.get(submission_type, []),
        "submission_id": submission_record["submission_id"]
    }

# Register tools
from storybook.tools import tool_registry
for tool_func in [
    format_validator,
    style_guide_formatter,
    ebook_formatter,
    submission_package_formatter
]:
    tool_registry.register_tool(tool_func, "formatting")