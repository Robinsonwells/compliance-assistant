# Enhanced Legal Document Preprocessor
# This code will clean XML structure while preserving precise citation information

from typing import Dict, List, Tuple, Optional, Any
import PyPDF2
import docx
import io
import re
import xml.etree.ElementTree as ET
import logging

def robust_xml_parse(text: str) -> List[Dict]:
    """Parse XML legal documents with robust section extraction"""
    legal_blocks = []
    
    try:
        # Clean up the XML text first
        xml_text = text.strip()
        
        # If it doesn't start with XML declaration, add a root wrapper
        if not xml_text.startswith('<?xml') and not xml_text.startswith('<root'):
            xml_text = f'<root>{xml_text}</root>'
        
        # Parse the XML
        root = ET.fromstring(xml_text)
        
        # Find all code elements with type="Section"
        sections = root.findall('.//code[@type="Section"]')
        
        for section in sections:
            # Extract section number
            number_elem = section.find('number')
            section_number = number_elem.text.strip() if number_elem is not None and number_elem.text else ''
            
            # Extract section name/title
            name_elem = section.find('name')
            section_title = name_elem.text.strip() if name_elem is not None and name_elem.text else ''
            
            # Extract version if available
            version_elem = section.find('version')
            version = version_elem.text.strip() if version_elem is not None and version_elem.text else ''
            
            # Extract content
            content_elem = section.find('content')
            if content_elem is not None:
                # Get all text content from the content element, including nested elements
                content_text = ET.tostring(content_elem, encoding='unicode', method='text')
                content_text = content_text.strip()
                
                if content_text and section_number:
                    legal_blocks.append({
                        'number': section_number,
                        'title': section_title,
                        'content': content_text,
                        'version': version
                    })
        
        # If we found sections, return them
        if legal_blocks:
            return legal_blocks
            
    except ET.ParseError as e:
        print(f"XML parsing failed: {e}")
    except Exception as e:
        print(f"Unexpected error in XML parsing: {e}")
    
    # Return empty list if XML parsing failed
    return []

def detect_jurisdiction(text: str) -> str:
    """Detect jurisdiction from legal text"""
    text_lower = text.lower()
    
    # Check for specific state indicators
    if any(ny_term in text_lower for ny_term in ['new york', 'ny admin', 'nycrr', 'ny labor law']):
        return 'NY'
    elif any(nj_term in text_lower for nj_term in ['new jersey', 'nj admin', 'njac', 'nj labor']):
        return 'NJ'
    elif any(ct_term in text_lower for ct_term in ['connecticut', 'ct admin', 'conn. gen. stat', 'ct labor']):
        return 'CT'
    elif any(fed_term in text_lower for fed_term in ['federal', 'usc', 'cfr', 'flsa', 'fmla', 'department of labor']):
        return 'Federal'
    else:
        return 'Multi-State'

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
        
        # XML-based metadata
        if match := re.search(r'statecd="([^"]+)"', text):
            metadata['state'] = match.group(1)
        if match := re.search(r'sessionyear="([^"]+)"', text):
            metadata['year'] = match.group(1)
        if match := re.search(r'<code type="Title"><number>([^<]+)</number><name>([^<]+)</name>', text):
            metadata['title_number'] = match.group(1).strip()
            metadata['title_name'] = match.group(2).strip()
        
        # Plain text metadata patterns
        if not metadata.get('state'):
            # Look for state names or codes in various formats
            state_patterns = [
                r'(New Jersey|Connecticut|New York|NJ|CT|NY)\s+(Administrative|Admin)\s+Code',
                r'State\s+of\s+(New Jersey|Connecticut|New York)',
                r'(NJ|CT|NY)\s+Admin'
            ]
            for pattern in state_patterns:
                if match := re.search(pattern, text, re.IGNORECASE):
                    state_map = {'New Jersey': 'NJ', 'Connecticut': 'CT', 'New York': 'NY'}
                    metadata['state'] = state_map.get(match.group(1), match.group(1))
                    break
        
        # Year detection
        if not metadata.get('year'):
            year_match = re.search(r'(20\d{2})', text)
            if year_match:
                metadata['year'] = year_match.group(1)
        
        # Title detection
        if not metadata.get('title_number'):
            title_patterns = [
                r'Title\s+(\d+)',
                r'TITLE\s+(\d+)',
                r'title0*(\d+)'
            ]
            for pattern in title_patterns:
                if match := re.search(pattern, text, re.IGNORECASE):
                    metadata['title_number'] = match.group(1)
                    break
        
        # Set defaults
        metadata.setdefault('state', 'Unknown')
        metadata.setdefault('year', '2025')
        metadata.setdefault('title_number', '12')
        
        return metadata

    def _extract_legal_content(self, text: str) -> List[Dict]:
        """Extract pure legal content, filtering XML but preserving structure"""
        
        # First, try robust XML parsing for structured legal documents
        xml_blocks = robust_xml_parse(text)
        if xml_blocks:
            print(f"Successfully parsed {len(xml_blocks)} sections using XML parser")
            return xml_blocks
        
        print("XML parsing failed or no sections found, falling back to regex patterns")
        
        # Remove XML declaration and DTD
        content = re.sub(r'<\?xml[^>]*\?>', '', text)
        content = re.sub(r'<!DOCTYPE[^>]*>', '', content)
        
        legal_blocks = []
        
        # Multiple patterns to handle different legal document formats
        patterns = [
            # XML Section pattern (CT, NJ, etc.)
            r'<code type="Section"><number>([^<]+)</number>(?:<version>([^<]+)</version>)?<name>([^<]+)</name><content>(.*?)</content>\s*</code>',
            # Alternative XML pattern
            r'<number>([^<]+)</number>.*?<name>([^<]+)</name>.*?<content>(.*?)</content>',
            # Plain text section patterns
            r'Section\s+(\d+[.\-\w]*)[:\.]?\s*([^\n]+)\n(.*?)(?=Section\s+\d+|\Z)',
            r'§\s*(\d+[.\-\w]*)[:\.]?\s*([^\n]+)\n(.*?)(?=§\s*\d+|\Z)',
            # Numbered regulations
            r'^(\d+[.\-\w]*)\.\s*([^\n]+)\n(.*?)(?=^\d+[.\-\w]*\.|\Z)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.DOTALL | re.MULTILINE | re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                legal_blocks.append({
                    'number': groups[0].strip() if groups[0] else '',
                    'title': groups[-2].strip() if len(groups) > 2 else groups[1].strip() if len(groups) > 1 else '',
                     'content': self._clean_legal_text(groups[-1]),
                     'version': groups[1].strip() if len(groups) > 3 and groups[1] else ''
                })
        
        # If no structured content found, try to extract meaningful text blocks
        if not legal_blocks:
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            for i, para in enumerate(paragraphs):
                if self._has_legal_significance(para):
                    legal_blocks.append({
                        'number': str(i+1),
                        'title': f'Paragraph {i+1}',
                        'content': self._clean_legal_text(para),
                        'version': ''
                    })
        
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
                'jurisdiction': detect_jurisdiction(chunk['text']),
                'law_type': detect_law_type(chunk['text']),
                'industry_specific': detect_industry_specific(chunk['text']),
                'federal_vs_state': classify_federal_state(chunk['text']),
                'complexity_level': assess_content_complexity(chunk['text'])
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
        state = doc_metadata.get('state', 'Unknown')
        year = doc_metadata.get('year', '2025')
        
        if state != 'Unknown':
            return f"{state} Admin. Code § {section_number} ({year})"
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
