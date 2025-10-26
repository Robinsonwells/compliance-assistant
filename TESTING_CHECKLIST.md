# Testing Checklist - Large Document Processing System

## Pre-Deployment Testing

### ✅ Environment Setup
- [ ] Supabase credentials added to environment
  - [ ] SUPABASE_URL set
  - [ ] SUPABASE_ANON_KEY set
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Database tables created (migration applied)
- [ ] Qdrant connection still working
- [ ] OpenAI API key still valid

### ✅ Code Compilation
- [ ] No syntax errors in new modules
- [ ] Admin panel loads without errors
- [ ] No import errors in Python console

## Testing Phase 1: Small File (1-10 MB)

### Test File Details
- File name: ___________________
- File size: ___________ MB
- Expected chunks: ~___________
- Expected time: < 1 minute

### Testing Steps
1. [ ] Log into Admin Panel
2. [ ] Navigate to Knowledge Base tab
3. [ ] Upload small file
4. [ ] Click "Process All Documents"

### Verify UI Components
- [ ] Progress card appears with file info
- [ ] Phase timeline shows all 5 phases
- [ ] Phase 1: Document Analysis completes
- [ ] Phase 2: Text Extraction shows progress
- [ ] Phase 3: Semantic Chunking shows chunk count
- [ ] Phase 4: Embedding Generation shows speed
- [ ] Phase 5: Database Upload shows batch progress
- [ ] Memory monitor displays (green status expected)
- [ ] Activity log shows timestamped entries
- [ ] Summary statistics appear at end
- [ ] Success message displays

### Verify Database
```sql
-- Check session was created
SELECT * FROM processing_sessions
WHERE file_name = 'YOUR_FILE_NAME'
ORDER BY start_time DESC LIMIT 1;

-- Expected: status = 'completed', chunks_uploaded > 0
```

- [ ] Session created in processing_sessions
- [ ] Status shows 'completed'
- [ ] chunks_uploaded matches expected count
- [ ] Checkpoints created in processing_checkpoints
- [ ] end_time is set

### Verify Vector Database
- [ ] Chunks appear in "Browse Files & Chunks"
- [ ] Correct number of chunks listed
- [ ] Can browse and view chunk content
- [ ] Chunks have correct metadata

**Result:** ☐ PASS / ☐ FAIL
**Notes:** ___________________________________

---

## Testing Phase 2: Medium File (50-100 MB)

### Test File Details
- File name: ___________________
- File size: ___________ MB
- Expected chunks: ~___________
- Expected time: 5-10 minutes

### Testing Steps
1. [ ] Upload medium-sized file
2. [ ] Click "Process All Documents"
3. [ ] Observe processing (don't navigate away)

### Verify Batch Processing
- [ ] Chunking shows batch numbers (Batch 1 of X)
- [ ] Embedding generation shows sub-batches
- [ ] Upload shows batch progress (100 chunks per batch)
- [ ] Memory usage monitored throughout
- [ ] Memory stays in green/yellow zone (< 80%)
- [ ] Progress updates are smooth and regular
- [ ] No freezing or hanging
- [ ] Activity log updates in real-time

### Verify Performance
- [ ] Text extraction: ~50-100 pages/sec
- [ ] Chunking: ~1000-2000 chunks/sec
- [ ] Embeddings: ~50-200 chunks/sec
- [ ] Upload: ~500-1000 chunks/sec
- [ ] Total time: < 10 minutes

### Verify Memory Management
- [ ] Memory usage increases during embedding generation
- [ ] Memory cleanup happens between batches
- [ ] No sustained memory growth
- [ ] Memory returns to baseline after completion

### Verify Database
- [ ] Multiple checkpoints created (one per batch)
- [ ] checkpoint_phase shows progression
- [ ] memory_usage_mb recorded for each checkpoint
- [ ] Final session shows completed status

**Result:** ☐ PASS / ☐ FAIL
**Notes:** ___________________________________

---

## Testing Phase 3: Large File (200+ MB)

### Test File Details
- File name: ___________________
- File size: ___________ MB
- Expected chunks: ~___________
- Expected time: 15-30 minutes

### Testing Steps
1. [ ] Upload large file (200+ MB)
2. [ ] Click "Process All Documents"
3. [ ] Monitor throughout entire process

### Stress Test Verification
- [ ] System handles file without crashing
- [ ] Memory never reaches red zone (> 80%)
- [ ] All phases complete successfully
- [ ] No error messages appear
- [ ] Processing continues uninterrupted
- [ ] All batches process correctly
- [ ] Upload completes successfully

### Performance Under Load
- [ ] UI remains responsive throughout
- [ ] Progress updates continue regularly
- [ ] Memory monitor shows stable usage
- [ ] No browser tab freezing
- [ ] Can view activity log while processing

### Checkpoint Verification
```sql
-- Check checkpoint frequency
SELECT checkpoint_phase, COUNT(*) as checkpoint_count
FROM processing_checkpoints
WHERE session_id = 'YOUR_SESSION_ID'
GROUP BY checkpoint_phase;

-- Expected: Multiple checkpoints per phase
```

- [ ] Checkpoints created regularly (every ~500 chunks)
- [ ] Each phase has multiple checkpoints
- [ ] Memory usage tracked in checkpoints
- [ ] No gaps in checkpoint sequence

**Result:** ☐ PASS / ☐ FAIL
**Notes:** ___________________________________

---

## Testing Phase 4: YOUR 380MB FILE

### Test File Details
- File name: ___________________
- File size: 380 MB (confirmed)
- Expected chunks: 121,000
- Expected time: 30-45 minutes

### Pre-Test Preparation
- [ ] Clear browser cache
- [ ] Ensure stable internet connection
- [ ] Close unnecessary browser tabs
- [ ] Verify system has available memory (> 4GB free)

### Testing Steps
1. [ ] Upload the 380MB file
2. [ ] Click "Process All Documents"
3. [ ] DO NOT navigate away or close browser
4. [ ] Monitor continuously for 30-45 minutes

### Critical Success Factors
- [ ] File uploads successfully (< 400MB limit)
- [ ] Processing starts immediately
- [ ] Phase 1: Document Analysis completes
- [ ] Phase 2: Text Extraction succeeds
  - [ ] All pages/sections extracted
  - [ ] Character count displays
- [ ] Phase 3: Semantic Chunking succeeds
  - [ ] Shows ~121,000 chunks
  - [ ] Batch processing evident
- [ ] Phase 4: Embedding Generation succeeds
  - [ ] Progress through all batches
  - [ ] Speed metric displays
  - [ ] Memory stays stable
- [ ] Phase 5: Database Upload succeeds
  - [ ] All 121,000 chunks uploaded
  - [ ] No failed batches
  - [ ] Completion confirms success

### Monitor Throughout
- [ ] Memory stays below 80% (yellow/green zone)
- [ ] No error messages appear
- [ ] Progress is continuous (not stuck)
- [ ] Activity log updates regularly
- [ ] Batch numbers increment properly
- [ ] Time estimates seem reasonable

### Verify Completion
- [ ] Success message appears
- [ ] Summary shows 121,000 chunks processed
- [ ] Processing time recorded (should be 30-45 min)
- [ ] Success rate shows 100%

### Verify in Database
```sql
-- Your specific session
SELECT *
FROM processing_sessions
WHERE file_name = 'YOUR_380MB_FILE_NAME'
ORDER BY start_time DESC LIMIT 1;

-- Should show:
-- status = 'completed'
-- chunks_uploaded = 121000 (approximately)
-- end_time set
-- no error_message
```

- [ ] Session shows 'completed'
- [ ] chunks_uploaded ≈ 121,000
- [ ] Processing time reasonable
- [ ] No error_message

### Verify Checkpoints
```sql
SELECT COUNT(*) as total_checkpoints
FROM processing_checkpoints
WHERE session_id = 'YOUR_SESSION_ID';

-- Expected: 200-300 checkpoints (one per batch)
```

- [ ] 200-300 checkpoints created
- [ ] Covers all processing phases
- [ ] Memory usage recorded for each

### Verify in Qdrant
- [ ] File appears in "Browse Files & Chunks"
- [ ] Shows ~121,000 chunks
- [ ] Can browse first 100 chunks
- [ ] Chunks have correct metadata
- [ ] content_hash matches file

### Post-Processing Verification
- [ ] Can run Data Quality Audit
- [ ] Audit shows chunks with 'current' status
- [ ] Can search/query chunks via main app
- [ ] Chunks return relevant results

**Result:** ☐ PASS / ☐ FAIL
**If FAIL, error details:** ___________________________________

---

## Edge Case Testing

### Duplicate File Upload
- [ ] Upload same file twice
- [ ] System detects duplicate via hash
- [ ] Shows "Skipped - already in database"
- [ ] No duplicate chunks created

### Multiple Files in Batch
- [ ] Upload 3 files simultaneously
- [ ] Each processes with separate UI container
- [ ] All files complete successfully
- [ ] Progress tracked independently
- [ ] Final summary shows all 3 files

### Browser Tab Behavior
- [ ] Start processing large file
- [ ] Switch to another tab
- [ ] Return after 5 minutes
- [ ] Processing continued in background
- [ ] UI shows current progress

### Network Interruption Simulation
- [ ] Note: This is theoretical (checkpoint foundation)
- [ ] If upload fails midway, session recorded
- [ ] Can view failed session in database
- [ ] error_message contains useful info

---

## Performance Benchmarks

### File Size vs. Processing Time

| File Size | Chunks | Actual Time | Expected | Status |
|-----------|--------|-------------|----------|--------|
| Small (1-10 MB) | _____ | _____ min | < 1 min | ☐ Pass / ☐ Fail |
| Medium (50 MB) | _____ | _____ min | 5-10 min | ☐ Pass / ☐ Fail |
| Large (200 MB) | _____ | _____ min | 15-30 min | ☐ Pass / ☐ Fail |
| Your File (380 MB) | 121,000 | _____ min | 30-45 min | ☐ Pass / ☐ Fail |

### Memory Usage Tracking

| Phase | Peak Memory | Status | Notes |
|-------|-------------|--------|-------|
| Text Extraction | _____ MB | ☐ Green / ☐ Yellow / ☐ Red | _________ |
| Semantic Chunking | _____ MB | ☐ Green / ☐ Yellow / ☐ Red | _________ |
| Embedding Generation | _____ MB | ☐ Green / ☐ Yellow / ☐ Red | _________ |
| Database Upload | _____ MB | ☐ Green / ☐ Yellow / ☐ Red | _________ |

**Overall Memory Performance:** ☐ PASS / ☐ FAIL

---

## Rollback Testing (If Needed)

### If Issues Found
1. [ ] Implement rollback (see DEPLOYMENT_GUIDE.md)
2. [ ] Test with small file using legacy function
3. [ ] Verify legacy function still works
4. [ ] Document issues encountered

---

## Sign-Off

### All Critical Tests Passed
- [ ] Small file processing works
- [ ] Medium file processing works
- [ ] Large file processing works
- [ ] **380MB file processes successfully**
- [ ] Memory management works
- [ ] UI displays correctly
- [ ] Database tracking works
- [ ] No critical errors

### Overall System Status
☐ **PRODUCTION READY** - All tests passed, deploy with confidence
☐ **NEEDS TUNING** - Works but needs batch size adjustment
☐ **NEEDS FIXES** - Critical issues found, do not deploy

### Tester Information
- **Name:** ___________________________________
- **Date:** ___________________________________
- **Environment:** ☐ Development / ☐ Staging / ☐ Production
- **Browser:** ___________________________________
- **Notes:** ___________________________________

### Approval
☐ **APPROVED FOR PRODUCTION**

**Signature:** ___________________________________

---

## Quick Reference

### If Test Fails

1. **Check Logs**
   - Browser console for JavaScript errors
   - Activity log in UI for processing errors
   - Database error_message field

2. **Check Memory**
   - Was memory in red zone?
   - Did browser tab crash?
   - Reduce batch sizes if needed

3. **Check Database**
   - Was session created?
   - Are checkpoints being created?
   - Any error_message recorded?

4. **Check Configuration**
   - Correct Supabase credentials?
   - Qdrant accessible?
   - Batch sizes appropriate?

### Support Contacts
- **Technical Issues:** ___________________________________
- **Database Issues:** ___________________________________
- **Deployment Issues:** ___________________________________
