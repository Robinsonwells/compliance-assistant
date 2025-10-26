"""
Integrated Document Processor with Comprehensive UI
Combines streaming processing, session management, and UI components
"""

import uuid
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any
from qdrant_client.models import PointStruct
import streamlit as st

from streaming_processor import StreamingDocumentProcessor, MemoryMonitor
from session_manager import SessionManager
from ui_components import ProgressTracker, ProcessingPhase


class IntegratedDocumentProcessor:
    """
    Unified document processor that handles:
    - Streaming processing for large files
    - Session management and checkpointing
    - Comprehensive UI progress tracking
    - Memory monitoring and management
    """

    def __init__(
        self,
        chunker,
        embedding_model,
        qdrant_client,
        chunk_batch_size: int = 500,
        embedding_batch_size: int = 500,
        upload_batch_size: int = 100
    ):
        self.streaming_processor = StreamingDocumentProcessor(
            chunker,
            embedding_model,
            chunk_batch_size,
            embedding_batch_size,
            upload_batch_size
        )
        self.session_manager = SessionManager()
        self.qdrant_client = qdrant_client
        self.memory_monitor = MemoryMonitor()

    def process_file_with_ui(
        self,
        uploaded_file,
        ui_container
    ) -> tuple[bool, str, Optional[str]]:
        """
        Process uploaded file with comprehensive UI feedback
        Returns: (success, message, content_hash)
        """

        # Calculate file hash
        file_content = uploaded_file.read()
        content_hash = hashlib.md5(file_content).hexdigest()
        uploaded_file.seek(0)

        file_info = {
            'name': uploaded_file.name,
            'size': len(file_content),
            'type': uploaded_file.type,
            'hash': content_hash,
            'status': 'Processing'
        }

        # Initialize progress tracker
        progress_tracker = ProgressTracker()
        activity_log = []

        # Create UI containers
        with ui_container:
            # Main progress card
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                     padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;">
                    <h2 style="margin: 0;">📄 Processing: {uploaded_file.name}</h2>
                    <p style="margin: 5px 0 0 0; opacity: 0.9;">
                        {self._format_size(file_info['size'])} • {uploaded_file.type}
                    </p>
                </div>
            """, unsafe_allow_html=True)

            # Create containers for different UI sections
            timeline_container = st.container()
            progress_container = st.container()
            memory_container = st.container()
            activity_container = st.container()

        # Session management
        session_id = None
        try:
            # Check for duplicate
            activity_log.append({
                'timestamp': datetime.now(),
                'message': 'Checking for duplicate files...',
                'level': 'info'
            })

            if self.session_manager.check_duplicate_by_hash(content_hash):
                with ui_container:
                    st.warning(f"⏭️ File already exists in database: {uploaded_file.name}")
                return True, f"Skipped - already in database: {uploaded_file.name}", content_hash

            # Check in Qdrant as well
            try:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                existing_points = self.qdrant_client.scroll(
                    collection_name="legal_regulations",
                    scroll_filter=Filter(
                        must=[FieldCondition(key="content_hash", match=MatchValue(value=content_hash))]
                    ),
                    limit=1,
                    with_payload=False,
                    with_vectors=False
                )
                if existing_points[0]:
                    with ui_container:
                        st.warning(f"⏭️ File already exists in vector database: {uploaded_file.name}")
                    return True, f"Skipped - already in database: {uploaded_file.name}", content_hash
            except Exception:
                pass

            # Create session
            session_id = self.session_manager.create_session(
                file_name=uploaded_file.name,
                file_hash=content_hash,
                file_size=len(file_content),
                metadata={'file_type': uploaded_file.type}
            )

            activity_log.append({
                'timestamp': datetime.now(),
                'message': f'Created processing session: {session_id}',
                'level': 'success'
            })

            # Initialize phase data
            phase_data = {
                'current': 0,
                'total': 1,
                'message': 'Initializing...'
            }

            # Process document in streaming batches
            total_chunks_uploaded = 0
            processing_start_time = datetime.now()

            for batch_data in self.streaming_processor.process_document_streaming(
                uploaded_file,
                uploaded_file.name,
                uploaded_file.type,
                content_hash,
                progress_callback=lambda data: self._update_ui(
                    data, timeline_container, progress_container,
                    memory_container, activity_container,
                    progress_tracker, activity_log, file_info
                ),
                checkpoint_callback=lambda data: self._create_checkpoint(
                    session_id, data
                )
            ):
                # Update session phase
                self.session_manager.update_session_phase(
                    session_id,
                    ProcessingPhase.EMBEDDING,
                    total_chunks_uploaded
                )

                # Generate embeddings for this batch
                batch_chunks = batch_data['chunks']
                activity_log.append({
                    'timestamp': datetime.now(),
                    'message': f"Generating embeddings for batch {batch_data['batch_num']}...",
                    'level': 'info'
                })

                embeddings = self.streaming_processor.generate_embeddings_batch(
                    batch_chunks,
                    progress_callback=lambda data: self._update_ui(
                        data, timeline_container, progress_container,
                        memory_container, activity_container,
                        progress_tracker, activity_log, file_info
                    )
                )

                # Upload to Qdrant
                self.session_manager.update_session_phase(
                    session_id,
                    ProcessingPhase.UPLOADING,
                    total_chunks_uploaded
                )

                uploaded_count = self._upload_batch_to_qdrant(
                    batch_chunks,
                    embeddings,
                    uploaded_file.name,
                    content_hash,
                    batch_data,
                    progress_tracker,
                    activity_log,
                    timeline_container,
                    progress_container,
                    memory_container,
                    activity_container,
                    file_info
                )

                total_chunks_uploaded += uploaded_count

                activity_log.append({
                    'timestamp': datetime.now(),
                    'message': f"Batch {batch_data['batch_num']} complete: {uploaded_count} chunks uploaded",
                    'level': 'success'
                })

                # Cleanup memory
                self.memory_monitor.cleanup()

            # Complete session
            processing_time = (datetime.now() - processing_start_time).total_seconds()
            self.session_manager.complete_session(
                session_id,
                total_chunks_uploaded,
                success=True
            )

            # Show completion summary
            with ui_container:
                st.markdown("---")
                st.success(f"✅ Successfully processed {total_chunks_uploaded:,} chunks from {uploaded_file.name}")

                stats = {
                    'total_chunks': total_chunks_uploaded,
                    'processing_time': processing_time,
                    'success_rate': 100.0,
                    'avg_speed': total_chunks_uploaded / processing_time if processing_time > 0 else 0
                }
                progress_tracker.render_summary_statistics(stats)

            return True, f"Processed {total_chunks_uploaded} chunks", content_hash

        except Exception as e:
            # Mark session as failed
            if session_id:
                self.session_manager.complete_session(
                    session_id,
                    0,
                    success=False,
                    error_message=str(e)
                )

            activity_log.append({
                'timestamp': datetime.now(),
                'message': f"Error: {str(e)}",
                'level': 'error'
            })

            with ui_container:
                st.error(f"❌ Error processing file: {str(e)}")

            return False, f"Error: {str(e)}", content_hash

    def _update_ui(
        self,
        data: Dict[str, Any],
        timeline_container,
        progress_container,
        memory_container,
        activity_container,
        progress_tracker: ProgressTracker,
        activity_log: list,
        file_info: Dict
    ):
        """Update all UI components with current progress"""

        phase = data.get('phase', ProcessingPhase.INITIALIZING)

        # Update timeline
        with timeline_container:
            progress_tracker.render_phase_timeline(phase, data)

        # Update detailed progress
        with progress_container:
            progress_tracker.render_detailed_progress(phase, data, file_info)

        # Update memory monitor
        with memory_container:
            memory_data = self.memory_monitor.get_memory_usage()
            progress_tracker.render_memory_monitor(memory_data)

        # Update activity log
        if 'message' in data:
            activity_log.append({
                'timestamp': datetime.now(),
                'message': data['message'],
                'level': 'info'
            })

        with activity_container:
            progress_tracker.render_activity_log(activity_log)

    def _create_checkpoint(self, session_id: str, data: Dict[str, Any]):
        """Create checkpoint in database"""
        memory_usage = self.memory_monitor.get_memory_usage()

        self.session_manager.create_checkpoint(
            session_id=session_id,
            phase=data.get('phase', 'unknown'),
            chunks_processed=data.get('chunks_processed', 0),
            current_batch=data.get('batch_num', 0),
            checkpoint_data=data,
            memory_usage_mb=int(memory_usage['used_mb'])
        )

    def _upload_batch_to_qdrant(
        self,
        chunks: list,
        embeddings: list,
        file_name: str,
        content_hash: str,
        batch_data: Dict,
        progress_tracker: ProgressTracker,
        activity_log: list,
        timeline_container,
        progress_container,
        memory_container,
        activity_container,
        file_info: Dict
    ) -> int:
        """Upload batch of chunks to Qdrant"""

        points = []
        now = datetime.now().isoformat()

        # Prepare points
        for i, (ch, embedding) in enumerate(zip(chunks, embeddings)):
            vector = embedding.tolist()

            payload = {
                'text': ch['text'],
                **ch['metadata'],
                'source_file': file_name,
                'content_hash': content_hash,
                'upload_date': now,
                'processed_by': 'admin_streaming'
            }

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload
            )
            points.append(point)

        # Upload in sub-batches
        upload_batch_size = 100
        uploaded = 0
        total_batches = (len(points) + upload_batch_size - 1) // upload_batch_size

        for i in range(0, len(points), upload_batch_size):
            batch = points[i:i+upload_batch_size]
            current_batch = (i // upload_batch_size) + 1

            self.qdrant_client.upsert(
                collection_name="legal_regulations",
                points=batch
            )

            uploaded += len(batch)

            # Update UI
            upload_data = {
                'phase': ProcessingPhase.UPLOADING,
                'uploaded': uploaded,
                'total': len(points),
                'current_batch': current_batch,
                'total_batches': total_batches,
                'failed': 0,
                'retries': 0,
                'message': f'Uploaded {uploaded}/{len(points)} chunks'
            }

            self._update_ui(
                upload_data,
                timeline_container,
                progress_container,
                memory_container,
                activity_container,
                progress_tracker,
                activity_log,
                file_info
            )

        return uploaded

    def _format_size(self, size_bytes: int) -> str:
        """Format file size"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
