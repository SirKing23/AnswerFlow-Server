# RAG Pipeline System Design
### Static Site + Supabase + Python Server on Render

---

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Pattern 2 — Why We Chose It](#pattern-2)
4. [Supabase Schema](#supabase-schema)
5. [Postgres Trigger](#postgres-trigger)
6. [Python Server — File Structure](#python-server-file-structure)
7. [File Routing — Parser Decision Tree](#file-routing)
8. [Chunking Strategy](#chunking-strategy)
9. [Embedding Pipeline](#embedding-pipeline)
10. [Deployment — Render](#deployment-render)
11. [Troubleshooting Log](#troubleshooting-log)
12. [Cost Breakdown](#cost-breakdown)
13. [Future Considerations](#future-considerations)

---

## Overview

Full design of a RAG (Retrieval-Augmented Generation) pipeline built on top of an existing static website with Supabase as the backend.

**What it does:** When a user uploads a file, the system automatically parses, chunks, and embeds the content into pgvector. Those embeddings can then be used to answer questions about the user's documents using an LLM.

**Stack:**
- Frontend: Static site on Render
- Database + Storage: Supabase (Postgres + pgvector + Storage buckets)
- Processing Server: Python/FastAPI on Render
- Embeddings: OpenAI `text-embedding-3-small`
- Document Parsing: Unstructured.io API (PDFs), mammoth (DOCX), openpyxl (Excel), built-in (TXT/CSV/MD)

---

## Architecture

```
User uploads file (frontend)
        │
        ▼
user_files table INSERT (Supabase)
        │
        │  Postgres trigger fires automatically
        ▼
POST /webhook/file-uploaded  →  Python Server (Render)
        │
        │  Background task (returns 200 immediately)
        ▼
┌───────────────────────────────────────────┐
│  Determine parser by mime_type            │
│                                           │
│  .txt .md .csv  → built-in (free)         │
│  .docx          → mammoth (free)          │
│  .xlsx          → openpyxl (free)         │
│  .pdf .doc .ppt → Unstructured.io API     │
└───────────────────────────────────────────┘
        │
        ▼
chunk_text()  →  embed_chunks()  →  document_embeddings (Supabase)
        │
        ▼
processing_status updated → Supabase Realtime → Frontend UI updates
```

---

## Pattern 2

Three upload patterns were considered. Pattern 2 was chosen.

### Why Not Pattern 1 (Frontend sends to both)
- User's browser uploads the file twice — once to Supabase, once to Render
- Double bandwidth for the user, bad experience on slow connections

### Why Not Pattern 3 (Frontend sends to Render only)
- Render server holds the HTTP connection open for the entire upload duration
- Slow user connections tie up server memory
- Supabase CDN is better at handling large file uploads than a single Render instance

### Pattern 2 Flow
```
Browser → Supabase Storage    (CDN-optimized, fast)
Supabase trigger → Render     (server-to-server, after upload completes)
Render → Supabase pgvector    (vectors written back after processing)
```

**Key benefit:** Render is only involved after the file has fully landed in Supabase. The user's connection speed doesn't affect server memory.

---

## Supabase Schema

### Alter `user_files` (existing table)

```sql
alter table user_files
  add column if not exists storage_bucket      text default 'user-files',
  add column if not exists processing_status   text default 'pending',
  -- pending | queued | parsing | chunking | embedding | storing | done | failed | skipped
  add column if not exists is_embeddable       boolean default true,
  add column if not exists content_hash        text,
  add column if not exists processed_at        timestamptz,
  add column if not exists processing_job_id   uuid;

create index if not exists idx_user_files_status
  on user_files(user_id, processing_status);

create index if not exists idx_user_files_hash
  on user_files(content_hash);
```

### Create `file_processing_jobs` (new table)

```sql
create table file_processing_jobs (
  id              uuid primary key default gen_random_uuid(),
  user_file_id    uuid references user_files(id) on delete cascade,
  user_id         uuid references auth.users(id),
  storage_path    text not null,
  file_name       text,
  file_type       text,
  status          text default 'queued',
  error_message   text,
  chunk_count     int,
  created_at      timestamptz default now(),
  updated_at      timestamptz default now()
);

create index if not exists idx_jobs_user_file on file_processing_jobs(user_file_id);
create index if not exists idx_jobs_status    on file_processing_jobs(status);
create index if not exists idx_jobs_user      on file_processing_jobs(user_id);

alter table file_processing_jobs enable row level security;

create policy "users_own_jobs" on file_processing_jobs
  for all using (auth.uid() = user_id);
```

### Add FK back to `user_files`

```sql
alter table user_files
  add constraint fk_processing_job
  foreign key (processing_job_id)
  references file_processing_jobs(id)
  on delete set null;
```

### Alter `document_embeddings` (existing table)

```sql
alter table document_embeddings
  add column if not exists job_id uuid references file_processing_jobs(id) on delete set null;

create index if not exists idx_embeddings_job
  on document_embeddings(job_id);

-- HNSW index for fast vector similarity search
create index if not exists idx_embeddings_vector
  on document_embeddings
  using hnsw (embedding vector_cosine_ops)
  with (m = 16, ef_construction = 64);
```

### Vector Search SQL Function

```sql
create or replace function match_chunks(
  query_embedding vector(1536),
  match_user_id uuid,
  match_count int default 5,
  similarity_threshold float default 0.75
)
returns table (
  id uuid, filename text, chunk_index int,
  content text, metadata jsonb, similarity float
)
language sql stable as $$
  select id, filename, chunk_index, content, metadata,
    1 - (embedding <=> query_embedding) as similarity
  from document_embeddings
  where user_id = match_user_id
    and 1 - (embedding <=> query_embedding) > similarity_threshold
  order by embedding <=> query_embedding
  limit match_count;
$$;
```

---

## Postgres Trigger

The trigger fires on every INSERT into `user_files` and calls the Python server via HTTP.

> **Note:** `ALTER DATABASE` and `ALTER ROLE` for custom settings are restricted on Supabase. Values are hardcoded directly in the function instead.

```sql
create extension if not exists http with schema extensions;

create or replace function notify_python_server()
returns trigger as $$
declare
  is_processable boolean;
  payload        jsonb;
begin
  is_processable := NEW.mime_type in (
    'application/pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/vnd.ms-excel',
    'text/plain',
    'text/csv',
    'text/markdown'
  );

  if not is_processable then
    update user_files
      set processing_status = 'skipped',
          is_embeddable     = false,
          updated_at        = now()
      where id = NEW.id;
    return NEW;
  end if;

  update user_files
    set processing_status = 'queued',
        updated_at        = now()
    where id = NEW.id;

  payload := jsonb_build_object(
    'user_file_id',   NEW.id,
    'user_id',        NEW.user_id,
    'storage_path',   NEW.storage_path,
    'file_name',      NEW.file_name,
    'mime_type',      NEW.mime_type,
    'file_size',      NEW.file_size,
    'storage_bucket', 'user-files'
  );

  perform extensions.http_post(
    url     := 'https://your-server.onrender.com/webhook/file-uploaded',
    body    := payload::text,
    headers := jsonb_build_object(
      'Content-Type',     'application/json',
      'X-Webhook-Secret', 'your-webhook-secret-here'
    )
  );

  return NEW;
end;
$$ language plpgsql security definer;

create trigger on_user_file_inserted
  after insert on user_files
  for each row
  execute function notify_python_server();
```

> **Important:** Deploy the Python server first, get the Render URL, then create the trigger. Wrong order causes failed webhook calls.

---

## Python Server File Structure

```
render-server/
├── main.py                        ← FastAPI app entry point + health check
├── config.py                      ← All env vars loaded here
├── requirements.txt
├── render.yaml                    ← Render deployment config
├── middleware/
│   ├── __init__.py
│   └── auth.py                    ← Webhook secret verification
├── routers/
│   ├── __init__.py
│   └── ingest.py                  ← POST /webhook/file-uploaded
├── services/
│   ├── __init__.py
│   ├── pipeline.py                ← Orchestrates full processing flow
│   ├── chunker.py                 ← Sentence-aware chunking with overlap
│   ├── embedder.py                ← OpenAI batched embedding + retry
│   └── parser/
│       ├── __init__.py
│       ├── router.py              ← Decides which parser to use
│       ├── text_parser.py         ← TXT, MD, CSV
│       ├── docx_parser.py         ← mammoth
│       ├── excel_parser.py        ← openpyxl read_only mode
│       └── unstructured_parser.py ← Unstructured.io API
└── utils/
    ├── __init__.py
    └── supabase_client.py         ← All Supabase DB operations
```

### Start Command (Render)
```
uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `SUPABASE_URL` | ✅ | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | ✅ | Service role key — NOT anon key |
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
| `UNSTRUCTURED_API_KEY` | ✅ | Unstructured.io API key |
| `WEBHOOK_SECRET` | ✅ | Must match value hardcoded in trigger |
| `STORAGE_BUCKET` | ❌ | Default: `user-files` |
| `CHUNK_SIZE_TOKENS` | ❌ | Default: `512` |
| `CHUNK_OVERLAP_TOKENS` | ❌ | Default: `64` |
| `EMBEDDING_BATCH_SIZE` | ❌ | Default: `100` |
| `MAX_FILE_SIZE_MB` | ❌ | Default: `25` |

---

## File Routing

```
mime_type received
        │
        ├── text/plain              → parse_txt()       (built-in)
        ├── text/markdown           → parse_markdown()  (built-in, strips syntax)
        ├── text/csv                → parse_csv()       (built-in, row→sentence format)
        ├── text/tsv                → parse_csv()       (built-in)
        │
        ├── application/vnd...docx  → parse_docx()      (mammoth)
        ├── application/vnd...xlsx  → parse_xlsx()      (openpyxl read_only=True)
        ├── application/vnd.ms-excel→ parse_xlsx()      (openpyxl read_only=True)
        │
        ├── application/pdf         → parse_with_unstructured()  (hi_res → fast fallback)
        ├── application/msword      → parse_with_unstructured()
        └── application/vnd...pptx  → parse_with_unstructured()
```

### Why Excel Uses `read_only=True`
A 10MB `.xlsx` file can expand to 100MB+ in memory if loaded normally. `read_only=True` streams rows instead of loading the full workbook object.

### Why CSV Is Converted to Sentences
```
# Bad embedding:
"John,30,Manila,Engineering"

# Good embedding:
"Name: John | Age: 30 | City: Manila | Department: Engineering"
```

### Unstructured.io Strategy
- Always tries `hi_res` first — uses vision transformer models, best quality
- Falls back to `fast` if `hi_res` fails or times out
- Timeout: `connect=30s`, `read=600s`, `write=60s`

---

## Chunking Strategy

Token estimation: `len(text) // 4` (4 chars ≈ 1 token).

> `tiktoken` removed — no pre-built wheels for Python 3.14 on Render, requires Rust to compile.

```
CHUNK_SIZE_TOKENS    = 512
CHUNK_OVERLAP_TOKENS = 64
```

### How It Works
1. Split text into sentences (paragraph breaks + punctuation boundary detection)
2. Accumulate sentences until token limit hit
3. Save chunk, backtrack 64 tokens for overlap
4. Track current heading/section, store in chunk metadata

### Overlap Prevents Context Loss
```
Chunk 1: "...quarterly results showed 12% increase. Revenue from APAC..."
                                                   ↑ overlap starts
Chunk 2: "Revenue from APAC grew 18% driven by enterprise sales..."
```

---

## Embedding Pipeline

- Model: `text-embedding-3-small`
- Batching: 100 chunks per OpenAI API call
- Retry: 3 attempts with exponential backoff via `tenacity`
- Vector dimensions: 1536

### Processing Status Flow
```
pending → queued → parsing → chunking → embedding → storing → done
                                                             → failed
                                                             → skipped
```

Status written to both `user_files.processing_status` and `file_processing_jobs.status` at every step.

### Ownership Chain
```
user_files.user_id  (source of truth, set by Supabase Auth)
       │  trigger sends it in payload
       ▼
Python server stamps user_id on every record it creates
       │
       ▼
file_processing_jobs.user_id
document_embeddings.user_id
       │
       ▼
RLS policies enforce ownership on every query
```

---

## Deployment — Render

### Order of Operations
```
1. Push code to GitHub
2. Create Web Service in Render → connect repo
3. Set environment variables in Render dashboard
4. Deploy → get your Render URL
5. Update trigger function in Supabase with Render URL
6. Run trigger SQL in Supabase SQL editor
7. Test by uploading a file from frontend
```

### Render Plan Recommendation

| Plan | RAM | Cost | Suitable For |
|---|---|---|---|
| Free | 512MB | $0 | Development only — spins down after 15min |
| Starter | 512MB | $7/mo | Small apps, always-on |
| Standard | 2GB | $25/mo | Multiple concurrent users |

Use **Starter ($7/mo) minimum** for production. Free tier cold starts (30-60s) make webhook calls feel broken.

### Health Check
```
GET /health → {"status": "ok"}
```

---

## Troubleshooting Log

### `hashlib2==1.0.0` — Package Does Not Exist
`hashlib` is Python stdlib. No install needed. Removed from `requirements.txt`.

---

### `tiktoken` — Failed Building Wheel
Render runs Python 3.14. No pre-built tiktoken wheels for Python 3.14, requires Rust + read-only filesystem.

**Fix:** Replaced with `len(text) // 4`.

---

### `ModuleNotFoundError: No module named 'routers'`
Two fixes required:
1. Add empty `__init__.py` to every subfolder
2. Add `sys.path.insert()` to top of every file that imports across folders

```python
# One level deep (routers/, services/, middleware/, utils/)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Two levels deep (services/parser/)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
```

---

### Foreign Key Constraint Violation on `file_processing_jobs`

**Error:**
```
insert or update on table "file_processing_jobs" violates foreign key constraint
Key (user_file_id)=(xxx) is not present in table "user_files"
```

**Cause:** Race condition. Trigger fires before `user_files` INSERT transaction commits. Python server responds fast enough that it tries to reference a row that doesn't exist yet.

**Fix:** `await asyncio.sleep(2)` at start of `process_file_pipeline()`.

---

### `httpx.ReadError` on PDF Upload

**Cause:** `httpx` default timeout drops connection while waiting for Unstructured.io `hi_res` model. Complex PDFs take 3-5+ minutes.

**Fix:** Explicit per-operation timeouts:
```python
timeout = httpx.Timeout(
    connect=30.0,
    read=600.0,    # 10 minutes for hi_res
    write=60.0,
    pool=30.0
)
```

---

### `ALTER DATABASE permission denied`

Supabase restricts `ALTER DATABASE` and `ALTER ROLE` on hosted plans.

**Fix:** Hardcode webhook URL and secret directly in the trigger function body instead of using `current_setting()`.

---

## Cost Breakdown

### Per File Upload

| File Type | Parser Cost | Embedding Cost | Total |
|---|---|---|---|
| TXT / MD / CSV | $0.00 | ~$0.001–$0.01 | ~$0.01 |
| DOCX | $0.00 | ~$0.002–$0.02 | ~$0.02 |
| XLSX | $0.00 | ~$0.001–$0.01 | ~$0.01 |
| PDF (10 pages) | ~10 Unstructured pages | ~$0.01 | ~$0.01 |
| PDF (20 pages) | ~20 Unstructured pages | ~$0.02 | ~$0.02 |

### Monthly Infrastructure

| Service | Plan | Cost |
|---|---|---|
| Render Python server | Starter | $7/mo |
| Supabase | Free / Pro | $0–$25/mo |
| OpenAI | Pay-as-you-go | ~$5–20/mo |
| Unstructured.io | Free (15,000 pages) | $0 until exceeded |

### Extending the Free Tier
Route only PDF/DOC/PPTX through Unstructured. Self-parse TXT, MD, CSV, DOCX, XLSX for free. For apps where most uploads are DOCX/TXT the free tier can last 6-12 months.

---

## Future Considerations

### RAG Query Endpoint
A `/api/query` endpoint needs to be added:
1. Embed the user's question with `text-embedding-3-small`
2. Call `match_chunks()` SQL function for vector similarity search
3. Assemble context from top-K chunks
4. Call OpenAI chat completions with context
5. Return answer + source citations

### Re-processing Files
Delete old embeddings before re-processing:
```sql
delete from document_embeddings where job_id = 'old-job-id';
```

### Scaling Beyond Render Starter
When concurrent uploads cause memory issues, upgrade to Standard ($25/mo, 2GB RAM) or move to a queue-based system using Supabase `pg_cron`.

### Eliminating OpenAI Embedding Costs
`sentence-transformers` can run locally on Render. Tradeoff: higher RAM, slightly lower retrieval quality.
