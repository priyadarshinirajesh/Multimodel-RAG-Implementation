# Setup Guide for Clinical Decision Support System

## Prerequisites

1. **Python 3.9+** installed
2. **Qdrant Vector Store** running (for vector embeddings)
3. **GROQ API Key** (for LLM reasoning)

---

## Step 1: Get Your GROQ API Key

1. Go to [console.groq.com](https://console.groq.com/)
2. Sign up or log in
3. Create an API key
4. Copy the key

---

## Step 2: Configure Environment Variables

### Option A: Using `.env` File (Recommended)

1. Open `.env` file in the project root
2. Replace `your-groq-api-key-here` with your actual GROQ API key:

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
QDRANT_URL=http://localhost:6333
```

3. Save the file

### Option B: Manual Environment Variable (Windows)

**PowerShell:**
```powershell
$env:GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

**CMD:**
```cmd
set GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Mac/Linux:**
```bash
export GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

---

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
pip install python-dotenv  # For .env support
```

---

## Step 4: Start Qdrant (Vector Store)

```bash
# Using Docker (recommended)
docker run -p 6333:6333 qdrant/qdrant

# Or using Qdrant locally (if installed)
qdrant
```

---

## Step 5: Start the Backend API

```bash
cd /path/to/Multimodel-RAG-Implementation
uvicorn backend_api.main:app --reload --port 8000
```

You should see:
```
✅ Loaded 37 pathologies from metadata
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

---

## Step 6: Start the Frontend

In a **new terminal**:

```bash
cd frontend
python -m http.server 3000
```

Then open: **http://localhost:3000**

---

## Troubleshooting

### Error: `GROQ_API_KEY not found`

**Solution:** Make sure you've set the environment variable or added it to `.env` file.

### Error: `Connection refused on 127.0.0.1:6333`

**Solution:** Make sure Qdrant is running (see Step 4 above).

### Error: `ModuleNotFoundError`

**Solution:** Install all requirements:
```bash
pip install -r requirements.txt
pip install python-dotenv fastapi uvicorn
```

---

## Next Steps

- Modify the UI by editing `/frontend/index.html`, `/frontend/css/styles.css`, `/frontend/js/app.js`
- Modify API behavior in `/backend_api/main.py`
- Add new agents or modify existing ones in `/agents/`

---

For more help, check the README.md file.
