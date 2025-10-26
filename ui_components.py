"""
Comprehensive UI Components for Document Processing
Provides multi-level progress tracking and detailed status displays
"""

import streamlit as st
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta


class ProcessingPhase:
    """Constants for processing phases"""
    INITIALIZING = "initializing"
    TEXT_EXTRACTION = "text_extraction"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"


class ProgressTracker:
    """Comprehensive progress tracking for document processing"""

    PHASE_INFO = {
        ProcessingPhase.INITIALIZING: {
            'icon': '🔍',
            'title': 'Initializing',
            'description': 'Analyzing document and preparing for processing'
        },
        ProcessingPhase.TEXT_EXTRACTION: {
            'icon': '📄',
            'title': 'Text Extraction',
            'description': 'Extracting text content from document'
        },
        ProcessingPhase.CHUNKING: {
            'icon': '✂️',
            'title': 'Semantic Chunking',
            'description': 'Creating intelligent semantic chunks'
        },
        ProcessingPhase.EMBEDDING: {
            'icon': '🧠',
            'title': 'Generating Embeddings',
            'description': 'Creating vector embeddings for search'
        },
        ProcessingPhase.UPLOADING: {
            'icon': '📤',
            'title': 'Uploading to Database',
            'description': 'Storing chunks in vector database'
        },
        ProcessingPhase.COMPLETED: {
            'icon': '✅',
            'title': 'Completed',
            'description': 'Processing finished successfully'
        },
        ProcessingPhase.FAILED: {
            'icon': '❌',
            'title': 'Failed',
            'description': 'Processing encountered an error'
        }
    }

    def __init__(self):
        self.start_time = datetime.now()
        self.phase_times = {}
        self.activity_log = []

    def render_phase_timeline(self, current_phase: str, phase_data: Dict[str, Any]):
        """Render a vertical timeline showing all processing phases"""

        st.markdown("### 📊 Processing Pipeline")

        phases = [
            ProcessingPhase.INITIALIZING,
            ProcessingPhase.TEXT_EXTRACTION,
            ProcessingPhase.CHUNKING,
            ProcessingPhase.EMBEDDING,
            ProcessingPhase.UPLOADING,
            ProcessingPhase.COMPLETED
        ]

        for phase in phases:
            info = self.PHASE_INFO[phase]
            is_current = (phase == current_phase)
            is_completed = phases.index(phase) < phases.index(current_phase) if current_phase in phases else False

            # Determine status icon
            if is_completed:
                status_icon = '✅'
                status_color = '#4CAF50'
            elif is_current:
                status_icon = '⏳'
                status_color = '#2196F3'
            else:
                status_icon = '⭕'
                status_color = '#9E9E9E'

            # Render phase
            st.markdown(f"""
                <div style="display: flex; align-items: center; margin: 10px 0; padding: 10px;
                     background: {'#E3F2FD' if is_current else 'transparent'};
                     border-radius: 8px; border-left: 4px solid {status_color};">
                    <div style="font-size: 24px; margin-right: 15px;">{status_icon}</div>
                    <div style="flex-grow: 1;">
                        <div style="font-weight: bold; font-size: 16px;">{info['icon']} {info['title']}</div>
                        <div style="font-size: 12px; color: #666;">{info['description']}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    def render_detailed_progress(
        self,
        phase: str,
        phase_data: Dict[str, Any],
        file_info: Dict[str, Any]
    ):
        """Render detailed progress information for current phase"""

        st.markdown("---")
        st.markdown("### 📈 Current Phase Details")

        # File information card
        with st.expander(f"📁 File Information: {file_info.get('name', 'Unknown')}", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Size", self._format_file_size(file_info.get('size', 0)))
            with col2:
                st.metric("File Type", file_info.get('type', 'Unknown'))
            with col3:
                st.metric("Status", file_info.get('status', 'Processing'))

        # Phase-specific progress
        if phase == ProcessingPhase.TEXT_EXTRACTION:
            self._render_extraction_progress(phase_data)
        elif phase == ProcessingPhase.CHUNKING:
            self._render_chunking_progress(phase_data)
        elif phase == ProcessingPhase.EMBEDDING:
            self._render_embedding_progress(phase_data)
        elif phase == ProcessingPhase.UPLOADING:
            self._render_upload_progress(phase_data)

    def _render_extraction_progress(self, data: Dict[str, Any]):
        """Render text extraction progress"""
        st.markdown("#### 📄 Text Extraction Progress")

        current = data.get('current', 0)
        total = data.get('total', 1)
        progress = current / total if total > 0 else 0

        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(progress)
        with col2:
            st.metric("Progress", f"{int(progress * 100)}%")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Pages/Sections", f"{current} / {total}")
        with col2:
            st.metric("Characters", f"{data.get('total_characters', 0):,}")
        with col3:
            st.metric("Time Elapsed", self._format_elapsed_time())

        st.caption(data.get('message', 'Extracting text...'))

    def _render_chunking_progress(self, data: Dict[str, Any]):
        """Render chunking progress"""
        st.markdown("#### ✂️ Semantic Chunking Progress")

        total_chunks = data.get('total_chunks', 0)
        batch_num = data.get('batch_num', 0)
        total_batches = data.get('total_batches', 0)

        if total_batches > 0:
            progress = batch_num / total_batches
            col1, col2 = st.columns([3, 1])
            with col1:
                st.progress(progress)
            with col2:
                st.metric("Progress", f"{int(progress * 100)}%")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Chunks", f"{total_chunks:,}")
        with col2:
            st.metric("Current Batch", f"{batch_num} / {total_batches}")
        with col3:
            st.metric("Chunk Size", "1,200 chars")

        st.caption(data.get('message', 'Creating semantic chunks...'))

    def _render_embedding_progress(self, data: Dict[str, Any]):
        """Render embedding generation progress"""
        st.markdown("#### 🧠 Embedding Generation Progress")

        current = data.get('current', 0)
        total = data.get('total', 1)
        progress = current / total if total > 0 else 0

        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(progress)
        with col2:
            st.metric("Progress", f"{int(progress * 100)}%")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Embeddings", f"{current:,} / {total:,}")
        with col2:
            batch = data.get('sub_batch', 0)
            total_batches = data.get('total_sub_batches', 0)
            st.metric("Batch", f"{batch} / {total_batches}")
        with col3:
            chunks_per_sec = data.get('speed', 0)
            st.metric("Speed", f"{chunks_per_sec:.1f}/sec" if chunks_per_sec > 0 else "Calculating...")
        with col4:
            remaining = self._estimate_remaining_time(current, total, data.get('speed', 0))
            st.metric("ETA", remaining)

        st.caption(data.get('message', 'Generating embeddings...'))

    def _render_upload_progress(self, data: Dict[str, Any]):
        """Render database upload progress"""
        st.markdown("#### 📤 Database Upload Progress")

        uploaded = data.get('uploaded', 0)
        total = data.get('total', 1)
        progress = uploaded / total if total > 0 else 0

        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(progress)
        with col2:
            st.metric("Progress", f"{int(progress * 100)}%")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Uploaded", f"{uploaded:,} / {total:,}")
        with col2:
            batch = data.get('current_batch', 0)
            total_batches = data.get('total_batches', 0)
            st.metric("Batch", f"{batch} / {total_batches}")
        with col3:
            failed = data.get('failed', 0)
            st.metric("Failed", failed, delta_color="inverse")
        with col4:
            retries = data.get('retries', 0)
            st.metric("Retries", retries)

        st.caption(data.get('message', 'Uploading to vector database...'))

    def render_memory_monitor(self, memory_data: Dict[str, float]):
        """Render memory usage monitoring"""
        st.markdown("---")
        st.markdown("### 💾 Memory Monitor")

        used_mb = memory_data.get('used_mb', 0)
        percent = memory_data.get('percent', 0)
        available_mb = memory_data.get('available_mb', 0)

        # Determine color based on usage
        if percent < 60:
            color = 'green'
            status = '🟢 Normal'
        elif percent < 80:
            color = 'orange'
            status = '🟡 Elevated'
        else:
            color = 'red'
            status = '🔴 High'

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Status", status)
        with col2:
            st.metric("Used Memory", f"{used_mb:.1f} MB")
        with col3:
            st.metric("Usage %", f"{percent:.1f}%")
        with col4:
            st.metric("Available", f"{available_mb:.1f} MB")

        # Memory usage bar
        st.progress(percent / 100)

        if percent > 80:
            st.warning("⚠️ High memory usage detected. Processing may slow down.")

    def render_activity_log(self, activities: List[Dict[str, Any]], max_items: int = 20):
        """Render live activity log"""
        st.markdown("---")
        st.markdown("### 📋 Activity Log")

        with st.expander("View Recent Activities", expanded=False):
            if not activities:
                st.info("No activities recorded yet.")
                return

            # Show last N activities
            for activity in activities[-max_items:]:
                timestamp = activity.get('timestamp', datetime.now())
                message = activity.get('message', '')
                level = activity.get('level', 'info')

                # Color based on level
                if level == 'error':
                    icon = '❌'
                    color = '#f44336'
                elif level == 'warning':
                    icon = '⚠️'
                    color = '#ff9800'
                elif level == 'success':
                    icon = '✅'
                    color = '#4caf50'
                else:
                    icon = 'ℹ️'
                    color = '#2196f3'

                st.markdown(f"""
                    <div style="padding: 5px; margin: 5px 0; border-left: 3px solid {color};">
                        <span style="color: #666; font-size: 11px;">{timestamp.strftime('%H:%M:%S')}</span>
                        <span style="margin-left: 10px;">{icon} {message}</span>
                    </div>
                """, unsafe_allow_html=True)

    def render_summary_statistics(self, stats: Dict[str, Any]):
        """Render processing summary statistics"""
        st.markdown("---")
        st.markdown("### 📊 Processing Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Chunks",
                f"{stats.get('total_chunks', 0):,}",
                delta=stats.get('chunks_delta', None)
            )

        with col2:
            st.metric(
                "Processing Time",
                self._format_duration(stats.get('processing_time', 0))
            )

        with col3:
            st.metric(
                "Success Rate",
                f"{stats.get('success_rate', 100):.1f}%"
            )

        with col4:
            st.metric(
                "Avg Speed",
                f"{stats.get('avg_speed', 0):.1f} chunks/sec"
            )

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

    def _format_elapsed_time(self) -> str:
        """Format elapsed time since start"""
        elapsed = datetime.now() - self.start_time
        return self._format_duration(elapsed.total_seconds())

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"

    def _estimate_remaining_time(self, current: int, total: int, speed: float) -> str:
        """Estimate remaining time based on current progress and speed"""
        if speed <= 0 or current >= total:
            return "Calculating..."

        remaining_items = total - current
        remaining_seconds = remaining_items / speed

        return self._format_duration(remaining_seconds)
