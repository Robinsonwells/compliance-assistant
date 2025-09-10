# Enhanced Legal Document Preprocessor
# This code will clean XML structure while preserving precise citation information

from typing import Dict, List, Tuple, Optional, Any
import PyPDF2
import docx
import io
import re
import xml.etree.ElementTree as ET
import logging

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
        semantic_chunks = self._semantic_legal_splitting(legal_blocks, doc_metadata)
        
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
                    'content': self._clean_legal_text(groups[-1]) if groups[-1] else '',
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

    def _semantic_legal_splitting(self, legal_blocks: List[Dict], doc_metadata: Dict) -> List[Dict]:
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
                    chunks.append({
                        'text': subsection.strip(),
                        'section_number': block['number'],
                        'section_title': block['title'],
                        'subsection_index': i,
                        'semantic_type': self._classify_legal_content(subsection),
                        'version': block.get('version', '')
                    })
        
        return chunks

    def _detect_legal_subsections(self, text: str) -> List[str]:
        """Detect natural legal subsection boundaries"""
        
        # Look for common legal subsection markers
        patterns = [
            r'\n\s*\([a-z]\)',  # (a), (b), (c)
            r'\n\s*\(\d+\)',    # (1), (2), (3) 
            r'\n\s*\([ivx]+\)', # (i), (ii), (iii)
            r'\n\s*[A-Z][^.]*\.',  # New sentence starting paragraph
        ]
        
        # Try to split by subsection markers first
        for pattern in patterns:
            if re.search(pattern, text):
                splits = re.split(pattern, text)
                if len(splits) > 1:
                    # Reconstruct with markers
                    result = []
                    markers = re.findall(pattern, text)
                    if splits[0].strip():
                        result.append(splits[0].strip())
                    for marker, split in zip(markers, splits[1:]):
                        if split.strip():
                            result.append(marker.strip() + ' ' + split.strip())
                    return result
        
        # Fall back to paragraph-based splitting
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        return paragraphs if len(paragraphs) > 1 else [text]

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
            
            final_chunks.append({
                'text': chunk['text'],
                'metadata': {
                    'chunk_id': chunk_id,
                    'section': f"{chunk['section_number']} - {chunk['section_title']}",
                    'citation': citation,
                    'section_type': chunk['semantic_type'],
                    'semantic_density': calculate_semantic_density(chunk['text']),
                    'is_complete_section': chunk['subsection_index'] == 0,
                    'version': chunk.get('version', '')
                }
            })
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
                    chunks.append({
                        'text': current_chunk.strip(),
                        'metadata': {
                            'chunk_id': chunk_id,
                            'section': f'Chunk {chunk_id}',
                            'semantic_density': calculate_semantic_density(current_chunk),
                            'section_type': 'basic'
                        }
                    })
                    chunk_id += 1
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': {
                    'chunk_id': chunk_id,
                    'section': f'Chunk {chunk_id}',
                    'semantic_density': calculate_semantic_density(current_chunk),
                    'section_type': 'basic'
                }
            })

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
