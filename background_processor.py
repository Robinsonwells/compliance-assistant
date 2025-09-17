import threading
import queue
import time
import json
import os
import uuid
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import streamlit as st
from io import BytesIO
import PyPDF2
import docx
from advanced_chunking import LegalSemanticChunker, extract_pdf_text, extract_docx_text
from qdrant_client.models import PointStruct

@dataclass
class ProcessingJob:
    job_id: str
    files: List[Dict[str, Any]]  # List of file metadata
    status: str  # 'queued', 'processing', 'completed', 'failed'
    current_file: str
    progress: int
    total_files: int
    start_time: datetime
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    processed_chunks: int = 0
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'job_id': self.job_id,
            'files': self.files,
            'status': self.status,
            'current_file': self.current_file,
            'progress': self.progress,
            'total_files': self.total_files,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'error_message': self.error_message,
            'processed_chunks': self.processed_chunks
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create ProcessingJob from dictionary"""
        return cls(
            job_id=data['job_id'],
            files=data['files'],
            status=data['status'],
            current_file=data['current_file'],
            progress=data['progress'],
            total_files=data['total_files'],
            start_time=datetime.fromisoformat(data['start_time']) if data['start_time'] else datetime.now(),
            end_time=datetime.fromisoformat(data['end_time']) if data['end_time'] else None,
            error_message=data.get('error_message'),
            processed_chunks=data.get('processed_chunks', 0)
        )

class BackgroundProcessor:
    def __init__(self):
        self.jobs: Dict[str, ProcessingJob] = {}
        self.job_queue = queue.Queue()
        self.worker_thread = None
        self.is_running = False
        self.jobs_file = "processing_jobs.json"
        self._load_all_jobs()
    
    def start_worker(self):
        """Start the background worker thread"""
        if not self.is_running:
            self.is_running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            print("Background processor worker started")
    
    def submit_batch_job(self, files_data: List[Dict], chunker, qdrant_client, embedding_model, user_manager, session_id):
        """Submit a batch job for processing"""
        job_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        job = ProcessingJob(
            job_id=job_id,
            files=[{'name': f['name'], 'type': f['type'], 'size': f['size']} for f in files_data],
            status='queued',
            current_file='',
            progress=0,
            total_files=len(files_data),
            start_time=datetime.now()
        )
        
        self.jobs[job_id] = job
        self._save_job_state(job)
        
        # Put job in queue with all necessary data
        self.job_queue.put({
            'job_id': job_id,
            'files_data': files_data,
            'chunker': chunker,
            'qdrant_client': qdrant_client,
            'embedding_model': embedding_model,
            'user_manager': user_manager,
            'session_id': session_id
        })
        
        print(f"Submitted batch job {job_id} with {len(files_data)} files")
        return job_id
    
    def _worker_loop(self):
        """Main worker loop that processes jobs"""
        print("Worker loop started")
        while self.is_running:
            try:
                if not self.job_queue.empty():
                    job_data = self.job_queue.get(timeout=1)
                    self._process_batch_job(job_data)
                else:
                    time.sleep(1)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker loop error: {e}")
                time.sleep(5)  # Wait before retrying
    
    def _process_batch_job(self, job_data: Dict):
        """Process a batch job"""
        job_id = job_data['job_id']
        files_data = job_data['files_data']
        chunker = job_data['chunker']
        qdrant_client = job_data['qdrant_client']
        embedding_model = job_data['embedding_model']
        user_manager = job_data['user_manager']
        session_id = job_data['session_id']
        
        job = self.jobs[job_id]
        job.status = 'processing'
        self._save_job_state(job)
        
        print(f"Starting processing of job {job_id}")
        
        try:
            total_chunks = 0
            
            for i, file_data in enumerate(files_data):
                job.current_file = file_data['name']
                job.progress = i
                self._save_job_state(job)
                
                # Extend admin session every file
                user_manager.extend_session_activity(session_id)
                
                print(f"Processing file {i+1}/{len(files_data)}: {file_data['name']}")
                
                # Process single file
                success, message, chunks_processed = self._process_single_file_logic(
                    file_data, chunker, qdrant_client, embedding_model
                )
                
                if not success:
                    job.status = 'failed'
                    job.error_message = f"Failed on {file_data['name']}: {message}"
                    job.end_time = datetime.now()
                    self._save_job_state(job)
                    print(f"Job {job_id} failed: {job.error_message}")
                    return
                
                total_chunks += chunks_processed
                job.processed_chunks = total_chunks
                job.progress = i + 1
                self._save_job_state(job)
                
                print(f"Completed file {file_data['name']}: {chunks_processed} chunks")
            
            job.status = 'completed'
            job.end_time = datetime.now()
            self._save_job_state(job)
            
            print(f"Job {job_id} completed successfully: {total_chunks} total chunks processed")
            
        except Exception as e:
            job.status = 'failed'
            job.error_message = f"Unexpected error: {str(e)}"
            job.end_time = datetime.now()
            self._save_job_state(job)
            print(f"Job {job_id} failed with exception: {e}")
    
    def _process_single_file_logic(self, file_data: Dict, chunker, qdrant_client, embedding_model):
        """Process a single file - core logic without Streamlit UI"""
        try:
            # Extract text based on file type
            file_content = file_data['content']
            file_name = file_data['name']
            file_type = file_data['type']
            
            if file_type == "application/pdf":
                # Create a BytesIO object from the content
                pdf_file = BytesIO(file_content)
                text = self._extract_pdf_text_from_bytes(pdf_file)
            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                # Create a BytesIO object from the content
                docx_file = BytesIO(file_content)
                text = self._extract_docx_text_from_bytes(docx_file)
            elif file_type == "text/plain":
                text = file_content.decode("utf-8")
            else:
                return False, f"Unsupported file type: {file_type}", 0
            
            if not text or text.startswith("Error"):
                return False, f"Failed to extract text: {text}", 0
            
            # Create chunks
            chunks = chunker.legal_aware_chunking(text, max_chunk_size=1200)
            if not chunks:
                return False, "No chunks were created - check file format", 0
            
            # Generate embeddings
            chunk_texts = [ch['text'] for ch in chunks]
            embeddings = embedding_model.encode(chunk_texts, show_progress_bar=False)
            
            # Prepare points for Qdrant
            points = []
            now = datetime.now().isoformat()
            
            for i, (ch, embedding) in enumerate(zip(chunks, embeddings)):
                vector = embedding.tolist()
                
                # Prepare payload
                payload = {
                    'text': ch['text'],
                    **ch['metadata'],
                    'source_file': file_name,
                    'upload_date': now,
                    'processed_by': 'admin_batch'
                }
                
                # Create point
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=payload
                )
                points.append(point)
            
            # Upload points to Qdrant in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                qdrant_client.upsert(
                    collection_name="legal_regulations",
                    points=batch
                )
            
            return True, f"Successfully processed {file_name}", len(chunks)
            
        except Exception as e:
            return False, f"Error processing file: {str(e)}", 0
    
    def _extract_pdf_text_from_bytes(self, pdf_bytes: BytesIO) -> str:
        """Extract text from PDF bytes"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_bytes)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
    
    def _extract_docx_text_from_bytes(self, docx_bytes: BytesIO) -> str:
        """Extract text from DOCX bytes"""
        try:
            doc = docx.Document(docx_bytes)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            return f"Error reading DOCX: {str(e)}"
    
    def get_job_status(self, job_id: str) -> Optional[ProcessingJob]:
        """Get status of a specific job"""
        return self.jobs.get(job_id)
    
    def get_all_jobs(self) -> List[ProcessingJob]:
        """Get all jobs, sorted by start time (newest first)"""
        return sorted(self.jobs.values(), key=lambda x: x.start_time, reverse=True)
    
    def get_active_jobs(self) -> List[ProcessingJob]:
        """Get only active (queued or processing) jobs"""
        return [job for job in self.jobs.values() if job.status in ['queued', 'processing']]
    
    def _save_job_state(self, job: ProcessingJob):
        """Save job state to persistent storage"""
        try:
            # Load existing jobs
            if os.path.exists(self.jobs_file):
                with open(self.jobs_file, 'r') as f:
                    all_jobs = json.load(f)
            else:
                all_jobs = {}
            
            # Update with current job
            all_jobs[job.job_id] = job.to_dict()
            
            # Save back to file
            with open(self.jobs_file, 'w') as f:
                json.dump(all_jobs, f, indent=2)
                
        except Exception as e:
            print(f"Error saving job state: {e}")
    
    def _load_all_jobs(self):
        """Load all jobs from persistent storage"""
        try:
            if os.path.exists(self.jobs_file):
                with open(self.jobs_file, 'r') as f:
                    all_jobs_data = json.load(f)
                
                for job_id, job_data in all_jobs_data.items():
                    self.jobs[job_id] = ProcessingJob.from_dict(job_data)
                
                print(f"Loaded {len(self.jobs)} jobs from storage")
            else:
                print("No existing jobs file found")
                
        except Exception as e:
            print(f"Error loading jobs from storage: {e}")
    
    def cleanup_old_jobs(self, days_old: int = 7):
        """Remove jobs older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            jobs_to_remove = []
            
            for job_id, job in self.jobs.items():
                if job.start_time < cutoff_date and job.status in ['completed', 'failed']:
                    jobs_to_remove.append(job_id)
            
            for job_id in jobs_to_remove:
                del self.jobs[job_id]
            
            if jobs_to_remove:
                # Save updated jobs
                self._save_all_jobs()
                print(f"Cleaned up {len(jobs_to_remove)} old jobs")
                
        except Exception as e:
            print(f"Error cleaning up old jobs: {e}")
    
    def _save_all_jobs(self):
        """Save all current jobs to storage"""
        try:
            all_jobs_data = {job_id: job.to_dict() for job_id, job in self.jobs.items()}
            with open(self.jobs_file, 'w') as f:
                json.dump(all_jobs_data, f, indent=2)
        except Exception as e:
            print(f"Error saving all jobs: {e}")
    
    def stop_worker(self):
        """Stop the background worker"""
        self.is_running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        print("Background processor stopped")

# Global processor instance
@st.cache_resource
def get_background_processor():
    """Get or create the global background processor instance"""
    processor = BackgroundProcessor()
    processor.start_worker()
    return processor