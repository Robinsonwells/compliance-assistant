# Enhanced Legal Document Preprocessor
# This code will clean XML structure while preserving precise citation information

from typing import Dict, List, Tuple, Optional, Any
import PyPDF2
import docx
import io
import re
import xml.etree.ElementTree as ET
import logging
import html

def robust_xml_parse(text: str) -> List[Dict]:
    """Parse XML legal documents with robust section extraction - DEPRECATED"""
    # This function is now deprecated in favor of general text processing
    # Keeping for backward compatibility but will return empty list
    return []

def strip_xml_tags(text: str) -> str:
    """Strip all XML tags and convert to clean plain text"""
    try:
        # First, handle HTML entities
        text = html.unescape(text)
        
        # Remove XML declaration and DTD
        text = re.sub(r'<\?xml[^>]*\?>', '', text)
        text = re.sub(r'<!DOCTYPE[^>]*>', '', text)
        
        # Remove all XML/HTML tags but preserve their text content
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Clean up multiple spaces and normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    except Exception as e:
        print(f"Error stripping XML tags: {e}")
        return text

def detect_document_format(text: str) -> str:
    """Detect if document is XML, HTML, or plain text"""
    text_sample = text.strip()[:1000].lower()
    
    if text_sample.startswith('<?xml') or '<cfrdoc' in text_sample or '<code type=' in text_sample:
        return 'xml'
    elif '<html' in text_sample or '<body' in text_sample or '<div' in text_sample:
        return 'html'
    else:
        return 'plain_text'

def extract_general_legal_sections(text: str) -> List[Dict]:
    """Extract legal sections using general patterns that work across document types"""
    legal_blocks = []
    
    try:
        # Multiple comprehensive patterns for different legal document structures
        patterns = [
            # Federal regulations: "§ 100.1" or "Section 100.1"
            r'(?:^|\n)\s*(?:§|Section)\s+(\d+(?:\.\d+)*)\s*([^\n]*?)\n(.*?)(?=(?:^|\n)\s*(?:§|Section)\s+\d+|\Z)',
            
            # Parts and Subparts: "Part 100" or "Subpart A"
            r'(?:^|\n)\s*((?:Part|Subpart)\s+[A-Z0-9]+)\s*[-–—]?\s*([^\n]*?)\n(.*?)(?=(?:^|\n)\s*(?:Part|Subpart)\s+[A-Z0-9]+|\Z)',
            
            # Numbered sections: "100.1" at start of line
            r'(?:^|\n)\s*(\d+(?:\.\d+)+)\s+([^\n]*?)\n(.*?)(?=(?:^|\n)\s*\d+(?:\.\d+)+\s+|\Z)',
            
            # Chapter/Title patterns: "Chapter I" or "Title 29"
            r'(?:^|\n)\s*((?:Chapter|Title)\s+[IVXLCDM0-9]+)\s*[-–—]?\s*([^\n]*?)\n(.*?)(?=(?:^|\n)\s*(?:Chapter|Title)\s+[IVXLCDM0-9]+|\Z)',
            
            # Lettered subsections: "(a)" or "a."
            r'(?:^|\n)\s*(?:\(([a-z])\)|(a-z)\.)\s+([^\n]*?)\n(.*?)(?=(?:^|\n)\s*(?:\([a-z]\)|[a-z]\.)\s+|\Z)',
            
            # Numbered subsections: "(1)" or "1."
            r'(?:^|\n)\s*(?:\((\d+)\)|(\d+)\.)\s+([^\n]*?)\n(.*?)(?=(?:^|\n)\s*(?:\(\d+\)|\d+\.)\s+|\Z)',
            
            # Roman numeral subsections: "(i)" or "i."
            r'(?:^|\n)\s*(?:\(([ivxlcdm]+)\)|([ivxlcdm]+)\.)\s+([^\n]*?)\n(.*?)(?=(?:^|\n)\s*(?:\([ivxlcdm]+\)|[ivxlcdm]+\.)\s+|\Z)',
            
            # Generic section headers (words followed by colon or dash)
            r'(?:^|\n)\s*([A-Z][A-Za-z\s]+)[:–—]\s*([^\n]*?)\n(.*?)(?=(?:^|\n)\s*[A-Z][A-Za-z\s]+[:–—]|\Z)',
        ]
        
        for pattern in patterns:
            matches = list(re.finditer(pattern, text, re.DOTALL | re.MULTILINE | re.IGNORECASE))
            if matches:
                print(f"Found {len(matches)} matches using pattern for legal sections")
                for match in matches:
                    groups = match.groups()
                    # Handle different group structures from different patterns
                    if len(groups) >= 3:
                        # Extract number (could be in different group positions)
                        number = next((g for g in groups[:2] if g and g.strip()), 'Unknown')
                        # Title is usually the last non-content group
                        title = groups[-2] if len(groups) > 2 else ''
                        # Content is always the last group
                        content = groups[-1]
                        
                        if content and content.strip():
                            legal_blocks.append({
                                'number': number.strip(),
                                'title': title.strip() if title else '',
                                'content': content.strip(),
                                'version': ''
                            })
                
                if legal_blocks:
                    break  # Use the first pattern that finds matches
        
        if legal_blocks:
            print(f"Successfully extracted {len(legal_blocks)} sections using general patterns")
            return legal_blocks
            
    except Exception as e:
        print(f"Error in general legal section extraction: {e}")
    
    return []

def detect_jurisdiction(text: str) -> str:
    """Detect jurisdiction from legal text"""
    text_lower = text.lower()
    
    # Check for specific state indicators
    if any(fed_term in text_lower for fed_term in ['federal', 'cfr', 'code of federal regulations', 'usc', 'flsa', 'fmla', 'department of labor']):
        return 'Federal'
    elif any(ny_term in text_lower for ny_term in ['new york', 'ny admin', 'nycrr', 'ny labor law']):
        return 'NY'
    elif any(nj_term in text_lower for nj_term in ['new jersey', 'nj admin', 'njac', 'nj labor']):
        return 'NJ'
    elif any(ct_term in text_lower for ct_term in ['connecticut', 'ct admin', 'conn. gen. stat', 'ct labor']):
        return 'CT'
    else:
        return 'Unknown'

def detect_law_type(text: str) -> str:
    """Detect type of legal document"""
    text_lower = text.lower()
    
    if any(term in text_lower for term in ['statute', 'general statutes', 'labor law']):
        return 'statute'
    elif any(term in text_lower for term in ['regulation', 'admin code', 'administrative code', 'cfr']):
        return 'regulation'
    elif any(term in text_lower for term in ['guidance', 'interpretation', 'advisory', 'bulletin']):
        return 'guidance'
    elif any(term in text_lower for term in ['case law', 'court decision', 'ruling']):
        return 'case_law'
    else:
        return 'regulation'  # Default assumption

def detect_industry_specific(text: str) -> str:
    """Detect industry-specific legal requirements"""
    text_lower = text.lower()
    
    if any(term in text_lower for term in ['transportation', 'trucking', 'logistics', 'dot', 'hours of service']):
        return 'transportation'
    elif any(term in text_lower for term in ['healthcare', 'medical', 'hospital', 'nursing']):
        return 'healthcare'
    elif any(term in text_lower for term in ['construction', 'building', 'prevailing wage', 'davis-bacon']):
        return 'construction'
    elif any(term in text_lower for term in ['government contractor', 'federal contractor', 'service contract act']):
        return 'government_contractor'
    elif any(term in text_lower for term in ['restaurant', 'food service', 'hospitality', 'tipped employee']):
        return 'hospitality'
    else:
        return 'general'

def classify_federal_state(text: str) -> str:
    """Classify whether law is federal or state level"""
    text_lower = text.lower()
    
    federal_indicators = ['usc', 'cfr', 'federal', 'flsa', 'fmla', 'department of labor', 'interstate commerce']
    state_indicators = ['admin code', 'general statutes', 'state labor', 'commissioner']
    
    federal_count = sum(1 for indicator in federal_indicators if indicator in text_lower)
    state_count = sum(1 for indicator in state_indicators if indicator in text_lower)
    
    if federal_count > state_count:
        return 'federal'
    elif state_count > federal_count:
        return 'state'
    else:
        return 'mixed'

def assess_content_complexity(text: str) -> str:
    """Assess complexity level of legal content"""
    text_lower = text.lower()
    
    # High complexity indicators
    high_complexity = ['multi-state', 'interstate', 'choice of law', 'conflict of laws', 'federal preemption']
    medium_complexity = ['exception', 'provided that', 'notwithstanding', 'subject to']
    
    if any(indicator in text_lower for indicator in high_complexity):
        return 'high'
    elif any(indicator in text_lower for indicator in medium_complexity):
        return 'medium'
    elif len(text.split()) > 200:  # Long text tends to be more complex
        return 'medium'
    else:
        return 'low'

def extract_pdf_text(file_obj) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_obj.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def extract_docx_text(file_obj) -> str:
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(io.BytesIO(file_obj.read()))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        return f"Error reading DOCX: {str(e)}"

class LegalSemanticChunker:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def legal_aware_chunking(self, text: str, max_chunk_size: int = 1200) -> List[Dict[str, Any]]:
        """Main entry point - uses intelligent semantic chunking for any legal document"""
        # Always use intelligent chunking now
        return self.intelligent_legal_chunking(text, max_chunk_size)

    def intelligent_legal_chunking(self, text: str, max_chunk_size: int = 1200) -> List[Dict[str, Any]]:
        """AI-powered semantic chunking that works with any legal document format"""
        
        # Step 1: Extract document metadata
        doc_metadata = self._extract_document_metadata(text)
        
        # Step 2: Extract pure legal content, filtering XML but preserving structure
        legal_blocks = self._extract_legal_content(text)
        
        # Step 3: Split by semantic meaning
        semantic_chunks = self._semantic_legal_splitting(legal_blocks, doc_metadata, max_chunk_size)
        
        # Step 4: Generate proper citations and metadata
        final_chunks = self._enrich_chunks_with_citations(semantic_chunks, doc_metadata)
        
        return final_chunks

    def _extract_document_metadata(self, text: str) -> Dict[str, str]:
        """Extract metadata from any legal document format"""
        metadata = {}
        
        # Detect document format first
        doc_format = detect_document_format(text)
        metadata['document_format'] = doc_format
        
        # Federal document detection (CFR, USC, etc.)
        federal_patterns = [
            r'Code\s+of\s+Federal\s+Regulations',
            r'CFR',
            r'Title\s+(\d+)',
            r'Federal\s+Register',
            r'U\.S\.C\.',
            r'United\s+States\s+Code'
        ]
        
        is_federal = any(re.search(pattern, text, re.IGNORECASE) for pattern in federal_patterns)
        
        if is_federal:
            metadata['jurisdiction'] = 'Federal'
            # Extract title number for federal documents
            if match := re.search(r'Title\s+(\d+)', text, re.IGNORECASE):
                metadata['title_number'] = match.group(1)
            elif match := re.search(r'TITLENUM>Title\s+(\d+)', text):
                metadata['title_number'] = match.group(1)
        else:
            # State document detection
            state_patterns = [
                r'(New Jersey|Connecticut|New York|NJ|CT|NY)\s+(Administrative|Admin)\s+Code',
                r'State\s+of\s+(New Jersey|Connecticut|New York)',
                r'(NJ|CT|NY)\s+Admin',
                r'N\.J\.A\.C\.',
                r'NYCRR',
                r'Conn\.\s+Agencies\s+Regs'
            ]
            for pattern in state_patterns:
                if match := re.search(pattern, text, re.IGNORECASE):
                    state_map = {'New Jersey': 'NJ', 'Connecticut': 'CT', 'New York': 'NY'}
                    found_state = match.group(1)
                    metadata['jurisdiction'] = state_map.get(found_state, found_state)
                    break
        
        # Year detection
        year_patterns = [
            r'Revised\s+as\s+of\s+\w+\s+\d+,\s+(20\d{2})',
            r'as\s+of\s+\w+\s+\d+,\s+(20\d{2})',
            r'sessionyear="([^"]+)"',
            r'(20\d{2})'
        ]
        for pattern in year_patterns:
            if match := re.search(pattern, text, re.IGNORECASE):
                metadata['year'] = match.group(1)
                break
        
        # Title detection
        if not metadata.get('title_number') and not is_federal:
            title_patterns = [
                r'Title\s+(\d+)',
                r'TITLE\s+(\d+)',
                r'title0*(\d+)',
                r'<code type="Title"><number>([^<]+)</number>'
            ]
            for pattern in title_patterns:
                if match := re.search(pattern, text, re.IGNORECASE):
                    metadata['title_number'] = match.group(1)
                    break
        
        # Document name/subject detection
        name_patterns = [
            r'<SUBJECT>([^<]+)</SUBJECT>',
            r'SUBJECT>([^<]+)',
            r'<name>([^<]+)</name>',
            r'Subject:\s*([^\n]+)'
        ]
        for pattern in name_patterns:
            if match := re.search(pattern, text, re.IGNORECASE):
                metadata['document_name'] = match.group(1).strip()
                break
        
        # Document status detection
        metadata['document_status'] = self._detect_document_status(text)
        
        # Set defaults
        metadata.setdefault('jurisdiction', 'Unknown')
        metadata.setdefault('year', '2025')
        metadata.setdefault('title_number', 'Unknown')
        
        return metadata

    def _detect_document_status(self, text: str) -> str:
        """Detect document status from legal text"""
        text_lower = text.lower()
        
        # Check for explicit status indicators
        if any(indicator in text_lower for indicator in ['repealed', 'revoked', 'rescinded']):
            return 'repealed'
        elif any(indicator in text_lower for indicator in ['proposed', 'draft']):
            return 'proposed'
        elif any(indicator in text_lower for indicator in ['withdrawn', 'retracted']):
            return 'proposed_withdrawn'
        elif any(indicator in text_lower for indicator in ['reserved', 'placeholder']):
            return 'reserved'
        elif any(indicator in text_lower for indicator in ['effective', 'current', 'in force']):
            return 'current'
        else:
            # Default to current if no specific status indicators found
            return 'current'
    def _extract_legal_content(self, text: str) -> List[Dict]:
        """Extract pure legal content, filtering XML but preserving structure"""
        
        # Detect document format
        doc_format = detect_document_format(text)
        
        # If it's XML or HTML, strip tags to get clean text
        if doc_format in ['xml', 'html']:
            print(f"Detected {doc_format} document, stripping tags for clean text processing")
            clean_text = strip_xml_tags(text)
        else:
            print("Processing plain text document")
            clean_text = text
        
        # Try to extract structured legal sections
        legal_blocks = extract_general_legal_sections(clean_text)
        
        # If no structured sections found, fall back to paragraph-based chunking
        if not legal_blocks:
            print("No structured sections found, using paragraph-based chunking")
            paragraphs = [p.strip() for p in clean_text.split('\n\n') if p.strip()]
            for i, para in enumerate(paragraphs):
                if self._has_legal_significance(para):
                    legal_blocks.append({
                        'number': str(i+1),
                        'title': f'Paragraph {i+1}',
                        'content': self._clean_legal_text(para),
                        'version': ''
                    })
        else:
            # Clean the content of extracted sections
            for block in legal_blocks:
                block['content'] = self._clean_legal_text(block['content'])
        
        return legal_blocks

    def _clean_legal_text(self, text: str) -> str:
        """Clean legal text while preserving meaning and structure"""
        if not text:
            return ""
        
        # Remove all XML tags but preserve their text content
        text = re.sub(r'<[^>]+>', '', text)
        
        # Clean up entities
        text = text.replace('&sect;', '§')
        text = text.replace('&amp;', '&')
        text = text.replace('&quot;', '"')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        
        # Normalize whitespace but preserve paragraph breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove processing artifacts and excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s+', '\n', text)
        
        return text.strip()

    def _semantic_legal_splitting(self, legal_blocks: List[Dict], doc_metadata: Dict, max_chunk_size: int) -> List[Dict]:
        """Split legal content by semantic meaning, not arbitrary size"""
        
        chunks = []
        
        for block in legal_blocks:
            content = block['content']
            
            if not content or not self._has_legal_significance(content):
                continue
            
            # Split by natural legal boundaries
            subsections = self._detect_legal_subsections(content)
            
            if not subsections:
                subsections = [content]
            
            for i, subsection in enumerate(subsections):
                # Only create chunk if it has meaningful legal content
                if self._has_legal_significance(subsection):
                    # Check if subsection exceeds max_chunk_size
                    subsection_text = subsection.strip()
                    if len(subsection_text) <= max_chunk_size:
                        # Subsection fits within limit, add as single chunk
                        chunks.append({
                            'text': subsection_text,
                            'section_number': block['number'],
                            'section_title': block['title'],
                            'subsection_index': str(i),
                            'semantic_type': self._classify_legal_content(subsection_text),
                            'version': block.get('version', '')
                        })
                    else:
                        # Subsection is too large, split into smaller sub-chunks
                        sub_chunks = self._split_oversized_subsection(subsection_text, max_chunk_size)
                        for j, sub_chunk in enumerate(sub_chunks):
                            chunks.append({
                                'text': sub_chunk,
                                'section_number': block['number'],
                                'section_title': block['title'],
                                'subsection_index': f"{i}.{j}",
                                'semantic_type': self._classify_legal_content(sub_chunk),
                                'version': block.get('version', '')
                            })
        
        return chunks

    def _detect_legal_subsections(self, text: str) -> List[str]:
        """Detect natural legal subsection boundaries"""
        
        # Look for common legal subsection markers
        patterns = [
            r'(?=\n\s*\([a-z]\))',  # (a), (b), (c) - with lookahead
            r'(?=\n\s*\(\d+\))',    # (1), (2), (3) - with lookahead
            r'(?=\n\s*\([ivx]+\))', # (i), (ii), (iii) - with lookahead
            r'(?=\n\s*\d+\.\s)',    # 1. 2. 3. - with lookahead
        ]
        
        # Try to split by subsection markers first
        for pattern in patterns:
            if re.search(pattern, text):
                splits = re.split(pattern, text)
                # Filter out empty splits and return non-empty subsections
                result = [split.strip() for split in splits if split.strip()]
                if len(result) > 1:
                    return result
        

    def _detect_legal_subsections(self, text: str) -> List[str]:
        """Detect natural legal subsection boundaries"""
        
        # Look for common legal subsection markers
        patterns = [
            r'(?=\n\s*\([a-z]\))',  # (a), (b), (c) - with lookahead
            r'(?=\n\s*\(\d+\))',    # (1), (2), (3) - with lookahead
            r'(?=\n\s*\([ivx]+\))', # (i), (ii), (iii) - with lookahead
            r'(?=\n\s*\d+\.\s)',    # 1. 2. 3. - with lookahead
        ]
        
        # Try to split by subsection markers first
        for pattern in patterns:
            if re.search(pattern, text):
                splits = re.split(pattern, text)
                # Filter out empty splits and return non-empty subsections
                result = [split.strip() for split in splits if split.strip()]
                if len(result) > 1:
                    return result
        
        # If no subsection markers found, return the original text
        return [text]

    def _split_oversized_subsection(self, text: str, max_chunk_size: int) -> List[str]:
        """Split an oversized subsection into smaller chunks while preserving semantic meaning"""
        if len(text) <= max_chunk_size:
            return [text]
        
        sub_chunks = []
        
        # Try splitting by sentences first (preserves most semantic meaning)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""
        
        for sentence in sentences:
            # Check if adding this sentence would exceed the limit
            test_chunk = current_chunk + (" " if current_chunk else "") + sentence
            
            if len(test_chunk) <= max_chunk_size:
                current_chunk = test_chunk
            else:
                # Save current chunk if it has content
                if current_chunk:
                    sub_chunks.append(current_chunk.strip())
                
                # Handle case where single sentence is too large
                if len(sentence) > max_chunk_size:
                    # Split by clauses (semicolons, commas)
                    clauses = re.split(r'[;,]\s*', sentence)
                    clause_chunk = ""
                    
                    for clause in clauses:
                        test_clause_chunk = clause_chunk + ("; " if clause_chunk else "") + clause
                        
                        if len(test_clause_chunk) <= max_chunk_size:
                            clause_chunk = test_clause_chunk
                        else:
                            if clause_chunk:
                                sub_chunks.append(clause_chunk.strip())
                            
                            # If single clause is still too large, split by words
                            if len(clause) > max_chunk_size:
                                words = clause.split()
                                word_chunk = ""
                                
                                for word in words:
                                    test_word_chunk = word_chunk + (" " if word_chunk else "") + word
                                    
                                    if len(test_word_chunk) <= max_chunk_size:
                                        word_chunk = test_word_chunk
                                    else:
                                        if word_chunk:
                                            sub_chunks.append(word_chunk.strip())
                                        word_chunk = word
                                
                                if word_chunk:
                                    clause_chunk = word_chunk
                                else:
                                    clause_chunk = ""
                            else:
                                clause_chunk = clause
                    
                    if clause_chunk:
                        current_chunk = clause_chunk
                    else:
                        current_chunk = ""
                else:
                    current_chunk = sentence
        
        # Add any remaining chunk
        if current_chunk:
            sub_chunks.append(current_chunk.strip())
        
        # Filter out empty chunks
        return [chunk for chunk in sub_chunks if chunk.strip()]

    def _has_legal_significance(self, text: str) -> bool:
        """Determine if text chunk has legal significance worth indexing"""
        
        if not text or len(text.strip()) < 30:  # Too short to be meaningful
            return False
        
        # Look for legal keywords
        legal_indicators = [
            'shall', 'must', 'may', 'required', 'prohibited', 'violation',
            'penalty', 'fine', 'regulation', 'statute', 'provision',
            'definition', 'means', 'includes', 'procedure', 'application',
            'commissioner', 'department', 'section', 'subsection', 'rule'
        ]
        
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in legal_indicators if keyword in text_lower)
        
        # Require some legal keywords for significance
        return keyword_count >= 1

    def _classify_legal_content(self, text: str) -> str:
        """Classify the type of legal content for better retrieval"""
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['definition', 'means', 'includes']):
            return 'definition'
        elif any(word in text_lower for word in ['procedure', 'process', 'application']):
            return 'procedure'  
        elif any(word in text_lower for word in ['penalty', 'fine', 'violation']):
            return 'penalty'
        elif any(word in text_lower for word in ['requirement', 'must', 'shall']):
            return 'requirement'
        elif any(word in text_lower for word in ['repealed', 'reserved']):
            return 'repealed'
        else:
            return 'substantive'

    def _enrich_chunks_with_citations(self, chunks: List[Dict], doc_metadata: Dict) -> List[Dict[str, Any]]:
        """Generate proper citations and metadata for chunks"""
        
        final_chunks = []
        chunk_id = 1
        
        for chunk in chunks:
            citation = self._create_citation(doc_metadata, chunk['section_number'])
            
            # Create metadata dictionary separately
            chunk_metadata = {
                'chunk_id': f"chunk_{chunk_id}",
                'citation': citation,
                'section_number': chunk['section_number'],
                'section_title': chunk['section_title'],
                'subsection_index': chunk['subsection_index'],
                'semantic_type': chunk['semantic_type'],
                'version': chunk.get('version', ''),
                'jurisdiction': doc_metadata.get('jurisdiction', detect_jurisdiction(chunk['text'])),
                'law_type': detect_law_type(chunk['text']),
                'industry_specific': detect_industry_specific(chunk['text']),
                'federal_vs_state': classify_federal_state(chunk['text']),
                'complexity_level': assess_content_complexity(chunk['text']),
                'document_format': doc_metadata.get('document_format', 'unknown'),
                'document_name': doc_metadata.get('document_name', ''),
                'year': doc_metadata.get('year', '2025'),
                'title_number': doc_metadata.get('title_number', 'Unknown')
            }
            
            # Create final chunk data separately
            final_chunk_data = {
                'text': chunk['text'],
                'metadata': chunk_metadata
            }
            
            # Append to final chunks
            final_chunks.append(final_chunk_data)
            chunk_id += 1
        
        return final_chunks

    def _create_citation(self, doc_metadata: Dict[str, str], section_number: str) -> str:
        """Create precise legal citation"""
        jurisdiction = doc_metadata.get('jurisdiction', 'Unknown')
        year = doc_metadata.get('year', '2025')
        title_number = doc_metadata.get('title_number', 'Unknown')
        
        if jurisdiction == 'Federal':
            if title_number != 'Unknown':
                return f"{title_number} CFR § {section_number}"
            else:
                return f"CFR § {section_number} ({year})"
        elif jurisdiction == 'NY':
            return f"12 NYCRR § {section_number}"
        elif jurisdiction == 'NJ':
            return f"N.J.A.C. § {section_number}"
        elif jurisdiction == 'CT':
            return f"Conn. Agencies Regs. § {section_number}"
        elif jurisdiction != 'Unknown':
            return f"{jurisdiction} Admin. Code § {section_number} ({year})"
        else:
            return f"Admin. Code § {section_number} ({year})"

    def _basic_text_chunking(self, text: str, max_chunk_size: int) -> List[Dict[str, Any]]:
        """Basic chunking for non-legal content (fallback)"""
        chunks = []
        sentences = text.split('. ')
        current_chunk = ""
        chunk_id = 1

        for sentence in sentences:
            if len(current_chunk + sentence) <= max_chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunk_data = {
                        'text': current_chunk.strip(),
                        'section_number': str(chunk_id),
                        'section_title': f'Chunk {chunk_id}',
                        'subsection_index': '0',
                        'semantic_type': 'basic',
                        'version': ''
                    }
                    chunks.append(chunk_data)
                    chunk_id += 1
                current_chunk = sentence + ". "

        if current_chunk:
            chunk_data = {
                'text': current_chunk.strip(),
                'section_number': str(chunk_id),
                'section_title': f'Chunk {chunk_id}',
                'subsection_index': '0',
                'semantic_type': 'basic',
                'version': ''
            }
            chunks.append(chunk_data)

        return chunks

# Legacy functions maintained for compatibility
class LegalDocumentProcessor:
    """Legacy processor - now handled by intelligent chunking"""
    def __init__(self):
        pass

def enhanced_legal_chunking(text: str, max_chunk_size: int = 1200) -> List[Dict[str, Any]]:
    """Legacy function - redirects to intelligent chunking"""
    chunker = LegalSemanticChunker("")
    return chunker.intelligent_legal_chunking(text, max_chunk_size)

def calculate_semantic_density(text: str) -> float:
    """Calculate semantic density focusing on legal content"""
    if len(text.strip()) < 10:
        return 0.0
    keywords = [
        'shall', 'must', 'required', 'prohibited', 'penalty', 'fine', 'liable',
        'violation', 'compliance', 'regulation', 'statute', 'provision', 'section',
        'subsection', 'commissioner', 'taxpayer', 'tax', 'certificate', 'notice',
        'procedure', 'requirement', 'definition', 'means', 'includes', 'application'
    ]
    words = text.lower().split()
    count = sum(1 for w in words if any(k in w for k in keywords))
    base = count / len(words) if words else 0.0
    bonus = min(0.2, len(re.findall(r'[.!?]+', text)) / 10)
    penalty = 0.3 if len(text) < 50 else 0.1 if len(text) > 2000 else 0
    return max(0.0, min(1.0, base + bonus - penalty))
