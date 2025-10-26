# Deployment Guide - Large Document Processing System

## Pre-Deployment Checklist

### 1. Environment Variables

Ensure these are set in your `.env` file or deployment environment:

```bash
# Qdrant Configuration
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key

# Supabase Configuration (NEW - REQUIRED)
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key

# Admin Configuration
ADMIN_PASSWORD=your_admin_password
```

### 2. Dependencies

All required dependencies are in `requirements.txt`. The new addition is:

```
psutil>=5.9.0
```

This is required for memory monitoring.

### 3. Database Setup

The Supabase migration has already been applied. Verify tables exist:

```sql
SELECT * FROM processing_sessions LIMIT 1;
SELECT * FROM processing_checkpoints LIMIT 1;
```

If tables don't exist, the migration will need to be re-applied (already done in this deployment).

## Deployment Steps

### Option 1: Streamlit Cloud (Recommended)

1. **Push code to GitHub repository**
   ```bash
   git add .
   git commit -m "Add large document processing system with streaming"
   git push
   ```

2. **Update Streamlit Cloud deployment**
   - Changes will auto-deploy if connected to GitHub
   - Verify environment variables are set in Streamlit Cloud dashboard

3. **Verify Supabase connection**
   - Check that `SUPABASE_URL` and `SUPABASE_ANON_KEY` are configured
   - Test database connectivity from admin panel

### Option 2: Self-Hosted

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables**
   ```bash
   export SUPABASE_URL="your_url"
   export SUPABASE_ANON_KEY="your_key"
   # ... other variables
   ```

3. **Run application**
   ```bash
   streamlit run app.py
   ```

## Post-Deployment Verification

### 1. Test Small File (< 10MB)

1. Log into Admin Panel
2. Upload a small PDF or text file
3. Verify all 5 phases complete successfully
4. Check Supabase for session record

### 2. Test Medium File (50-100MB)

1. Upload a medium-sized document
2. Monitor memory usage in the UI
3. Verify batch processing works correctly
4. Check checkpoint creation in Supabase

### 3. Test Large File (200MB+)

1. Upload your largest test file
2. Monitor comprehensive UI feedback
3. Verify memory stays stable throughout processing
4. Confirm all chunks upload successfully

### 4. Test Your 380MB File

1. Upload the 380MB file with 121,000 chunks
2. Expected processing time: 30-45 minutes
3. Monitor:
   - Phase transitions
   - Batch progress
   - Memory usage
   - Checkpoint creation
4. Verify completion and all chunks in Qdrant

## Monitoring During Deployment

### Watch for:

1. **Memory Usage**
   - Should stay below 80% during processing
   - Green/Yellow status is normal
   - Red status means reduce batch sizes

2. **Processing Speed**
   - Text extraction: 50-100 pages/sec
   - Chunking: 1000-2000 chunks/sec
   - Embeddings: 50-200 chunks/sec
   - Upload: 500-1000 chunks/sec

3. **Error Rates**
   - Session completion rate should be > 95%
   - Failed uploads should be < 1%
   - Automatic retries should resolve most issues

### Database Queries for Monitoring

```python
# Check recent sessions
SELECT file_name, status, chunks_uploaded,
       end_time - start_time as processing_time
FROM processing_sessions
ORDER BY start_time DESC
LIMIT 10;

# Check for failed sessions
SELECT file_name, error_message, start_time
FROM processing_sessions
WHERE status = 'failed'
ORDER BY start_time DESC;

# Check memory usage trends
SELECT checkpoint_phase,
       AVG(memory_usage_mb) as avg_memory,
       MAX(memory_usage_mb) as max_memory
FROM processing_checkpoints
GROUP BY checkpoint_phase;

# Check processing speed
SELECT
  file_name,
  chunks_uploaded,
  EXTRACT(EPOCH FROM (end_time - start_time)) as seconds,
  chunks_uploaded / EXTRACT(EPOCH FROM (end_time - start_time)) as chunks_per_sec
FROM processing_sessions
WHERE status = 'completed'
ORDER BY start_time DESC
LIMIT 20;
```

## Configuration Tuning

### If Processing is Too Slow

Increase batch sizes in `pages/1_Admin_Panel.py`:

```python
integrated_processor = IntegratedDocumentProcessor(
    chunker=chunker,
    embedding_model=embedding_model,
    qdrant_client=coll,
    chunk_batch_size=1000,      # ↑ Increased from 500
    embedding_batch_size=1000,  # ↑ Increased from 500
    upload_batch_size=200       # ↑ Increased from 100
)
```

### If Memory Usage is Too High

Reduce batch sizes:

```python
integrated_processor = IntegratedDocumentProcessor(
    chunker=chunker,
    embedding_model=embedding_model,
    qdrant_client=coll,
    chunk_batch_size=250,       # ↓ Reduced from 500
    embedding_batch_size=250,   # ↓ Reduced from 500
    upload_batch_size=50        # ↓ Reduced from 100
)
```

### If Upload Limit is Too Low

Increase in `.streamlit/config.toml`:

```toml
[server]
maxUploadSize = 800  # Increase to 800MB
```

## Rollback Plan

If issues occur with the new system, the legacy processing function is still available:

1. In `pages/1_Admin_Panel.py`, find line ~787
2. Replace this block:
   ```python
   success, message, content_hash = integrated_processor.process_file_with_ui(
       uploaded_file,
       file_container
   )
   ```

3. With legacy function:
   ```python
   success, message, content_hash = process_uploaded_file_legacy(
       uploaded_file, chunker, coll, embedding_model,
       progress_bar=st.progress(0), status_text=st.empty()
   )
   ```

4. Note: Legacy function will still fail on very large files (100,000+ chunks)

## Support & Troubleshooting

### Common Issues

**Issue: Session creation fails**
- Check Supabase credentials in environment
- Verify RLS policies are enabled
- Check network connectivity to Supabase

**Issue: Memory keeps climbing**
- Reduce batch sizes
- Check for memory leaks in custom code
- Restart application to clear memory

**Issue: Embeddings generation is very slow**
- Check if GPU is available (much faster)
- Consider reducing embedding_batch_size
- Verify sentence-transformers model is loaded correctly

**Issue: Qdrant upload fails**
- Check Qdrant API key and URL
- Verify collection exists
- Check network latency to Qdrant

### Debug Mode

Enable verbose logging by adding to top of `integrated_processor.py`:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Check

Create a simple health check endpoint to verify all systems:

```python
def health_check():
    checks = {
        'qdrant': test_qdrant_connection(),
        'supabase': test_supabase_connection(),
        'openai': test_openai_api(),
        'memory': check_available_memory()
    }
    return all(checks.values())
```

## Success Criteria

Deployment is successful when:

1. ✅ Small files (< 10MB) process in under 1 minute
2. ✅ Medium files (50MB) process in 5-10 minutes
3. ✅ Large files (200MB+) process without memory errors
4. ✅ Your 380MB file processes completely in < 60 minutes
5. ✅ Memory usage stays stable throughout processing
6. ✅ UI shows clear progress at all phases
7. ✅ Sessions are tracked in Supabase
8. ✅ Checkpoints are created regularly
9. ✅ All chunks appear in Qdrant after completion
10. ✅ No critical errors in logs

## Next Steps After Deployment

1. **Monitor first week** of production usage
2. **Collect metrics** on processing times and memory usage
3. **Fine-tune batch sizes** based on actual performance
4. **Review failed sessions** and improve error handling
5. **Consider implementing** resume functionality using checkpoints
6. **Add admin dashboard** for session monitoring
7. **Set up alerts** for failed processing sessions

## Contact & Support

For issues specific to this implementation:
1. Check LARGE_DOCUMENT_PROCESSING.md for architecture details
2. Review session records in Supabase
3. Check activity logs in UI during processing
4. Enable debug logging for detailed traces
