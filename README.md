# RAG Processing Server

Python/FastAPI server that handles the file processing pipeline:
**Supabase Storage → Parse → Chunk → Embed → Supabase pgvector**

## How It Fits In

```
user_files INSERT (Supabase)
       │
       │  Postgres trigger fires
       ▼
POST /webhook/file-uploaded  ← this server
       │
       ├── .txt .md .csv     → self-parse (free)
       ├── .docx             → mammoth
       ├── .xlsx             → openpyxl (read_only)
       └── .pdf .doc .pptx   → Unstructured.io API
       │
       ▼
chunk_text() → embed_chunks() → document_embeddings (Supabase)
```

## Local Development

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Copy env file and fill in values
cp .env.example .env

# 3. Run
uvicorn main:app --reload --port 8000

# 4. Test health
curl http://localhost:8000/health
```

## Testing the Webhook Locally

Use ngrok to expose your local server to Supabase:

```bash
ngrok http 8000
# Copy the https URL and set it in Supabase:
# alter database postgres set app.python_server_url = 'https://xxxx.ngrok.io';
```

## Deploying to Render

1. Push this folder to a GitHub repo
2. Create a new **Web Service** in Render
3. Connect your GitHub repo
4. Set environment variables (see .env.example)
5. Set Health Check Path to `/health`
6. Deploy

## File Type Routing

| File Type | Parser       | Cost          |
|-----------|-------------|---------------|
| .txt      | built-in    | free          |
| .md       | built-in    | free          |
| .csv      | built-in    | free          |
| .docx     | mammoth     | free          |
| .xlsx     | openpyxl    | free          |
| .pdf      | Unstructured| uses quota    |
| .doc      | Unstructured| uses quota    |
| .pptx     | Unstructured| uses quota    |

## Environment Variables

| Variable                  | Required | Description                          |
|---------------------------|----------|--------------------------------------|
| SUPABASE_URL              | ✅       | Your Supabase project URL            |
| SUPABASE_SERVICE_ROLE_KEY | ✅       | Service role key (not anon key)      |
| OPENAI_API_KEY            | ✅       | OpenAI API key                       |
| UNSTRUCTURED_API_KEY      | ✅       | Unstructured.io API key              |
| WEBHOOK_SECRET            | ✅       | Shared secret with Supabase trigger  |
| STORAGE_BUCKET            | ❌       | Bucket name (default: user-files)    |
| CHUNK_SIZE_TOKENS         | ❌       | Tokens per chunk (default: 512)      |
| CHUNK_OVERLAP_TOKENS      | ❌       | Overlap tokens (default: 64)         |
| EMBEDDING_BATCH_SIZE      | ❌       | OpenAI batch size (default: 100)     |
| MAX_FILE_SIZE_MB          | ❌       | Max upload size in MB (default: 25)  |
