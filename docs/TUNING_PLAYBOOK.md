# RAG Tuning Playbook

> **How to make your RAG actually good, not just "it runs".**

---

## 1. Retrieval Not Accurate Enough

### Symptom
- Answers are wrong
- It doesn't find the right documents
- Users say "答非所问"

### What to Tune

#### Chunk Size
- **Too small** (<300): Loses context, fragments the meaning
- **Too large** (>1500): Noisy, dilutes relevance
- **Sweet spot**: 500-1000 tokens for technical docs, 800 is a good default

```python
# config/settings.py
CHUNK_SIZE = 800      # try 500, 800, 1200
CHUNK_OVERLAP = 100   # ~10-15% of chunk_size
```

#### Reranking
- **What it does**: Takes Top 20 candidates, picks Top 5 that actually matter
- **Why it matters**: Vector search is good, but not great. Reranker is a precision boost.
- **Model**: `BAAI/bge-reranker-base` or `cross-encoder/ms-marco-MiniLM-L-6-v2`

```python
RETRIEVER_TOP_K = 20   # retrieve more candidates
RERANK_TOP_N = 5       # then rerank down to 5
```

#### Hybrid Search Weight
- **Vector weight**: Good for semantic similarity
- **BM25 weight**: Good for keyword matching
- **Default**: 0.6 vector, 0.4 BM25
- **If your docs are technical (lots of jargon)**: Increase BM25 weight → 0.5/0.5
- **If your docs are conversational**: Increase vector weight → 0.7/0.3

```python
# In file_rag/engine.py
EnsembleRetriever(
    retrievers=[dense_retriever, bm25_retriever],
    weights=[0.6, 0.4],  # tune this
)
```

---

## 2. Hallucination / Making Things Up

### Symptom
- Answers sound confident but are wrong
- Users catch it making up facts

### What to Do

#### Add Guardrail
- **Pre-check**: Is the context actually relevant?
- **Post-check**: Is the answer actually grounded in the context?

```python
# core/generators/guardrail.py
USE_GUARDRAIL = True
```

#### Lower Temperature
- **Temperature=0**: Most factual
- **Temperature=0.2**: Good default for RAG
- **Temperature=0.7**: Creative but risky for factual Q&A

```python
LLM_TEMPERATURE = 0.2  # not 0.7!
```

#### Cite Sources
- Make the model cite which document the answer came from
- This forces it to be more careful

```python
prompt = """Answer based ONLY on the context.
Cite sources as [1], [2], etc.
If you don't know, say "I don't have enough information."
"""
```

---

## 3. It's Too Slow

### Symptom
- First token takes 3+ seconds
- Users get impatient

### What to Do

#### Add Caching
- Cache frequent queries
- Same question → same answer → no need to re-run RAG

```python
USE_CACHE = True  # Redis, 1 hour TTL
```

#### Async Processing
- Document upload → background task
- Don't block the API request

```python
# core/tasks/document_tasks.py
# Upload → queue → process in background
```

#### Streaming Output
- Stream tokens as they come
- User sees text appearing, feels faster

```python
# Use streaming=True in the LLM call
```

---

## 4. Concurrent Users / It Crashes

### Symptom
- 10 users → it works
- 100 users → it crashes

### What to Do

#### Add Task Queue
- Use Celery + Redis
- Queue document processing tasks
- Workers process them in background

```python
# docker-compose.yml
celery:
    build: .
    command: celery -A worker worker --loglevel=info
    depends_on:
        - redis
```

#### Add Cache
- Cache frequent queries
- Less load on LLM API

```python
# core/cache/redis_cache.py
# TTL = 1 hour
```

#### Rate Limiting
- Prevent abuse
- 100 req/min/IP

```python
# core/middleware/rate_limit.py
RATE_LIMIT_REQUESTS = 100
```

---

## 5. How Do You Know It's Actually Better?

### The Problem
- You changed chunk size from 800 to 600
- Is it actually better? Or just different?
- **You can't tell by guessing.**

### The Solution: Build an Eval Set

```python
# core/eval/evaluator.py

# 1. Write 50-100 test questions
# 2. Run them through RAG
# 3. Score 3 metrics:
#    - Context Precision (did we retrieve the right docs?)
#    - Faithfulness (is the answer grounded?)
#    - Answer Relevancy (does it answer the question?)

# 4. Every time you change a parameter, run the eval again
# 5. Compare scores
```

### What "Good" Looks Like

| Score | Level |
|-------|-------|
| >0.8 | Production-ready |
| 0.6-0.8 | Okay for internal use |
| <0.6 | Needs work |

---

## Tuning Checklist

Before you go to production:

- [ ] Chunk size tuned (tested 3 values)
- [ ] Reranker working (Cross-Encoder, not keyword)
- [ ] Hybrid search weight tuned
- [ ] Guardrail enabled
- [ ] Temperature <= 0.2
- [ ] Caching enabled
- [ ] Async tasks for heavy work
- [ ] Rate limiting enabled
- [ ] Eval set built and scored >0.7
- [ ] Monitoring (LangSmith or logs)

---

**Remember: A RAG that "runs" is easy. A RAG that is actually accurate, fast, and reliable is hard. That's what companies pay you $15k/month for.**
