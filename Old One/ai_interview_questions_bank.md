# AI System Design Interview Questions Bank
## Categorized Questions with Model Answers

---

# CATEGORY 1: RAG & RETRIEVAL SYSTEMS

## Q1.1: Reducing Hallucinations in RAG
**Question:** "How would you reduce hallucination rates in the Chanakya student feedback system?"

**Model Answer:**

I'd implement a **4-layer defense stack**:

**Layer 1: Retrieval Quality**
- Hybrid search (BM25 + Vector) for both exact and semantic matching
- Cross-encoder re-ranking with threshold > 0.7
- If no relevant chunks found → Return "I don't know" instead of guessing

**Layer 2: Prompt Engineering**
- Citation-based prompting: "Answer ONLY using [SOURCE 1], [SOURCE 2]..."
- Explicit instruction: "If unsure, say 'I cannot verify this'"
- Few-shot examples showing correct vs rejected responses

**Layer 3: Generation Constraints**
- Temperature = 0.0-0.2 for deterministic output
- JSON mode for structured feedback
- Max tokens limit to prevent rambling

**Layer 4: Post-Generation Validation**
- NLI-based groundedness check (BART-MNLI or similar)
- Each generated sentence scored: Entailment > 0.7 = grounded
- RAGAS metrics for offline evaluation (faithfulness, relevancy)

```python
# Groundedness check pseudocode
for sentence in generated_feedback.split(". "):
    score = nli_model.check(context, sentence)
    if score < 0.7:
        flag_for_review(sentence)
```

---

## Q1.2: Chunking Strategy Selection
**Question:** "How do you choose between fixed-size chunking, semantic chunking, and parent document retrieval?"

**Model Answer:**

| Strategy | Use When | Trade-off |
|----------|----------|-----------|
| **Fixed-size** | High volume, low stakes, time-critical | Fast but loses context |
| **Semantic** | Essays, documents with natural structure | Better accuracy, higher cost |
| **Parent Doc** | Need precise retrieval + full context | Best of both, complex setup |

For Chanakya (educational content):
- **Student essays** → Semantic chunking (natural paragraph breaks)
- **Solutions/explanations** → Parent doc retrieval (find error, show full solution)
- **FAQ/simple queries** → Fixed-size is acceptable

---

## Q1.3: Scaling RAG to Millions of Documents
**Question:** "How would you scale a RAG system from 100K to 10M documents?"

**Model Answer:**

**Indexing Tier:**
- Horizontal sharding by topic/subject (e.g., Math, Science, English)
- Approximate Nearest Neighbor (ANN) indexes (HNSW, IVF)
- Async indexing pipeline with Celery workers

**Retrieval Tier:**
- Two-stage retrieval: Fast ANN → Accurate re-ranking
- Metadata filtering before vector search (reduce search space)
- Caching layer for frequent queries (30-50% hit rate)

**Infrastructure:**
- Milvus/Qdrant cluster with replicas
- Read replicas for query traffic
- Separate write path for indexing

**Metrics to Monitor:**
- P99 search latency < 100ms
- Index freshness < 5 minutes
- Recall@10 > 0.9

---

## Q1.4: Hybrid Search Implementation
**Question:** "Explain how you'd implement hybrid search combining keyword and vector retrieval."

**Model Answer:**

```python
# Architecture
┌─────────────────────────────────────────┐
│  Query: "quadratic formula errors"      │
│              │                          │
│    ┌─────────┴─────────┐                │
│    ▼                   ▼                │
│  BM25              Vector               │
│  (keyword)         (semantic)           │
│    │                   │                │
│    └─────────┬─────────┘                │
│              ▼                          │
│     RRF Fusion (k=60)                   │
│              │                          │
│              ▼                          │
│     Cross-Encoder Re-rank               │
│              │                          │
│              ▼                          │
│         Top 5 Results                   │
└─────────────────────────────────────────┘
```

**Key decisions:**
- **Fusion method**: Reciprocal Rank Fusion (RRF) with k=60
- **Weight**: 40% BM25, 60% Vector (tune per domain)
- **Re-ranker**: Cross-encoder (ms-marco-MiniLM) for final top-5

**Why hybrid wins:**
- "Error E-4023" → BM25 finds exact match
- "login problems" → Vector finds "authentication issues"

---

# CATEGORY 2: AI AGENTS & ORCHESTRATION

## Q2.1: Stateful Agent Persistence
**Question:** "Design a stateful Agent that can pause a task (e.g., grading 1000 assignments) and resume it days later. How do you persist the state?"

**Model Answer:**

**Architecture: Checkpointer Pattern**

```
Agent Runtime ←→ Postgres (JSONB)
                 ├─ agent_id (PK)
                 ├─ user_id
                 ├─ state_json
                 ├─ status (running/paused/completed)
                 └─ checkpoint_version
```

**Implementation approach:**

1. **State Serialization**
```python
@dataclass
class GradingState:
    task_id: str
    total: int
    completed: int
    current_index: int
    grades: dict[str, dict]
    messages: list[dict]
    error_log: list[str]
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))
```

2. **Checkpointing Strategy**
   - Save every N iterations (e.g., 10)
   - Save on explicit pause signal
   - Save on error/timeout

3. **Resume Flow**
```python
def resume(task_id):
    state = checkpointer.load(task_id)
    # Hydrate state
    # Continue from state.current_index
    for i in range(state.current_index, total):
        process(i)
```

**Why Postgres over Redis:**
- JSONB supports queries on state
- Durable (Redis with TTL loses data)
- Transaction support for complex updates

---

## Q2.2: Tool Failure Handling
**Question:** "How do you handle API failures in agent tool execution?"

**Model Answer:**

```python
class RobustToolExecutor:
    def execute(self, tool_name: str, args: dict) -> ToolResult:
        for attempt in range(max_retries):
            try:
                # Timeout protection
                result = timeout(30s, tools[tool_name](**args))
                return ToolResult(status="success", data=result)
            except TimeoutError:
                return ToolResult(
                    status="retry",
                    error="Timeout",
                    suggestion="Break into smaller operations"
                )
            except Exception as e:
                if attempt < max_retries - 1:
                    sleep(2 ** attempt)  # Exponential backoff
                    continue
                return ToolResult(
                    status="error",
                    error=str(e),
                    suggestion="Try alternative approach"
                )
```

**Key patterns:**
1. **Timeout wrapper** (30s default, configurable per tool)
2. **Exponential backoff** (1s, 2s, 4s)
3. **Informative errors** to help LLM recover
4. **Graceful degradation** (return partial results if possible)

---

## Q2.3: Multi-Agent Routing
**Question:** "When would you use a Router pattern vs a State Graph for multi-agent systems?"

**Model Answer:**

| Pattern | Use When | Example |
|---------|----------|---------|
| **Router** | Tasks are independent, no inter-agent communication | "Is this a grading or question generation request?" |
| **State Graph** | Complex workflows, conditional paths, feedback loops | "Generate → Review → Revise → Approve" |

**Router Pattern:**
```
User Query → Router (fast LLM) → Specialized Agent
                    ↓
     ┌──────────────┼──────────────┐
     ▼              ▼              ▼
  Grading      Questions      Curriculum
   Agent        Agent          Agent
```

**State Graph Pattern:**
```
Generate → Review ──→ Approved → Format → Done
              ↓
          Rejected
              ↓
          Generate (loop back)
```

**Decision criteria:**
- Fan-out only? → Router
- Need iteration/loops? → State Graph
- Shared state between steps? → State Graph

---

## Q2.4: Agent Memory Design
**Question:** "How do you implement memory for an AI agent that needs both conversation context and long-term knowledge?"

**Model Answer:**

**Two-tier memory architecture:**

```python
class AgentMemory:
    def __init__(self):
        # Short-term: Last N messages or token budget
        self.short_term = TokenBufferMemory(max_tokens=4000)
        
        # Long-term: Vector store for recall
        self.long_term = VectorStoreMemory(Qdrant, OpenAI_Embeddings)
    
    def get_context(self, query: str) -> str:
        # Recent conversation
        recent = self.short_term.get_messages()
        
        # Relevant past knowledge
        memories = self.long_term.recall(query, k=3)
        
        return format_context(recent, memories)
```

**Short-term options:**
- `ConversationBufferMemory(max_messages=20)` - Simple
- `TokenBufferMemory(max_tokens=4000)` - Precise

**Long-term storage:**
- Store: Important facts, user preferences, past decisions
- Recall: Semantic search on query
- Forget: TTL or explicit cleanup

---

# CATEGORY 3: LLMOps & SCALING

## Q3.1: Cost Optimization at Scale
**Question:** "You need to scale Chanakya AI to 100K concurrent users. OpenAI bill is $800K/month. How do you reduce costs?"

**Model Answer:**

**Hybrid Architecture (98% cost reduction):**

```
Request → Semantic Cache (40% hits, $0)
              ↓ miss
          Complexity Classifier
              ↓
    ┌─────────┼─────────┐
    ▼         ▼         ▼
  Simple   Moderate   Complex
  Llama-8B Llama-70B  GPT-4
  (50%)    (8%)       (2%)
```

**Cost breakdown:**
| Layer | Traffic | Monthly Cost |
|-------|---------|--------------|
| Cache (semantic) | 40% | $0 |
| Llama 3 8B | 50% | $8,000 |
| Llama 3 70B | 8% | $8,000 |
| GPT-4 fallback | 2% | $2,000 |
| **Total** | 100% | **$18,000** |

**Implementation:**
1. **Semantic cache**: 0.95 similarity threshold, 24hr TTL
2. **Complexity classifier**: Query length, math symbols, reasoning words
3. **Self-hosted**: vLLM on EC2 p4d (A100 GPUs)

---

## Q3.2: Latency vs Throughput
**Question:** "When would you use streaming vs batching for LLM inference?"

**Model Answer:**

| Mode | Latency | Throughput | Use Case |
|------|---------|------------|----------|
| **Streaming** | TTFT ~100ms | Low | Chat, real-time UI |
| **Small batch** (4-8) | TTFT ~300ms | Medium | API endpoints |
| **Large batch** (32+) | TTFT ~2s | High | Offline processing |

**Streaming implementation:**
```python
async def stream_response(prompt):
    stream = client.chat.completions.create(
        model="gpt-4", 
        messages=[...],
        stream=True
    )
    for chunk in stream:
        yield chunk.choices[0].delta.content
```

**When to batch:**
- Grading 1000 assignments overnight → Batch 32
- Student asking a question → Stream
- API returning feedback → Small batch (4-8)

---

## Q3.3: RAG vs Fine-Tuning
**Question:** "When should you use RAG vs fine-tuning vs both?"

**Model Answer:**

**Choose RAG when:**
- Knowledge changes frequently (curriculum updates)
- Need citations/attribution
- Limited training data (< 10K examples)
- Privacy: data can't be in model weights

**Choose Fine-Tuning when:**
- Consistent style/format needed (grading rubric)
- Specialized reasoning not in base model
- Reducing prompt size (behavior in weights)
- High volume (amortize training cost)

**Combine both for Chanakya:**
- Fine-tune for teacher communication style
- RAG for curriculum content retrieval

```python
# Combined approach
def generate_feedback(student_answer, question):
    # RAG: Get relevant curriculum context
    context = retriever.get(question)
    
    # Fine-tuned model: Generate in teacher voice
    return finetuned_llm.generate(
        prompt=f"Context: {context}\nStudent: {student_answer}"
    )
```

---

## Q3.4: Self-Hosting LLMs
**Question:** "How would you deploy Llama 3 70B for production use?"

**Model Answer:**

**Stack:**
- **Model**: Llama 3 70B with AWQ 4-bit quantization
- **Inference**: vLLM with PagedAttention
- **Hardware**: 2x A100 80GB (tensor parallel)
- **Hosting**: EC2 p4d.24xlarge or SageMaker

**vLLM deployment:**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-70B-Instruct \
    --tensor-parallel-size 2 \
    --quantization awq \
    --max-model-len 8192 \
    --port 8000
```

**Why vLLM:**
- PagedAttention: Efficient KV-cache (no memory waste)
- Continuous batching: 10-24x throughput vs HuggingFace
- OpenAI-compatible API: Drop-in replacement

**Scaling:**
- 3 vLLM nodes behind load balancer
- ~150 concurrent requests capacity
- Auto-scale based on queue depth

---

# CATEGORY 4: SYSTEM DESIGN SCENARIOS

## Q4.1: End-to-End Exam Grading System
**Question:** "Design an automated exam grading system that handles 10,000 students submitting simultaneously."

**Model Answer:**

```
┌─────────────────────────────────────────────────────────┐
│                  EXAM GRADING SYSTEM                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Student Uploads ──→ API Gateway ──→ SQS Queue          │
│                                          │               │
│                                          ▼               │
│                                    ┌─────────────┐       │
│                                    │   Celery    │       │
│                                    │   Workers   │       │
│                                    │   (n=50)    │       │
│                                    └──────┬──────┘       │
│                                           │              │
│            ┌──────────────────────────────┼──────────┐   │
│            ▼                              ▼          ▼   │
│       ┌─────────┐                  ┌───────────┐  Cache  │
│       │   OCR   │                  │    LLM    │   Hit   │
│       │ (if img)│                  │  Grading  │    │    │
│       └────┬────┘                  └─────┬─────┘    │    │
│            │                             │          │    │
│            └──────────────┬──────────────┘          │    │
│                           ▼                         │    │
│                     ┌──────────┐                    │    │
│                     │ Postgres │ ←──────────────────┘    │
│                     │ (grades) │                         │
│                     └──────────┘                         │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Key decisions:**
1. **Queue-based**: Handle burst traffic (SQS/RabbitMQ)
2. **Worker pool**: 50 Celery workers, auto-scale
3. **Semantic cache**: 30% students have similar answers
4. **Progressive feedback**: Store intermediate results
5. **Idempotency**: Retry without duplicate grading

---

## Q4.2: Real-Time Student Feedback System
**Question:** "Design a system that provides real-time feedback as students type their answers."

**Model Answer:**

**Requirements:**
- < 500ms latency for feedback
- Handle 10K concurrent students
- Graceful degradation

**Architecture:**
```
Student Types ──→ WebSocket ──→ Debounce (300ms)
                                     │
                                     ▼
                              ┌──────────────┐
                              │   Feedback   │
                              │   Service    │
                              └──────┬───────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
              Semantic          Small LLM      Full Analysis
               Cache           (Llama 8B)      (if paused)
              (< 5ms)          (< 200ms)       (on demand)
```

**Key patterns:**
1. **Debounce**: Wait 300ms after typing stops
2. **Tiered response**: Quick hint first, detailed later
3. **Streaming**: Show feedback as generated
4. **Fallback**: If LLM slow, show cached similar feedback

---

## Q4.3: Multi-Subject Question Generation
**Question:** "Design an agent that can generate balanced exams across multiple subjects with difficulty gradation."

**Model Answer:**

**State Graph approach:**

```
START
  │
  ▼
Analyze Requirements ──→ {subject, difficulty, count}
  │
  ▼
Generate by Subject (parallel)
  │
  ├── Math Agent ──→ 5 questions
  ├── Science Agent ──→ 5 questions
  └── English Agent ──→ 5 questions
  │
  ▼
Balance Check
  │
  ├── Pass ──→ Format Output ──→ END
  │
  └── Fail ──→ Regenerate weak areas ──→ Balance Check
```

**Quality constraints:**
- Bloom's taxonomy distribution (Remember, Apply, Analyze)
- Difficulty curve (Easy → Medium → Hard)
- No duplicate concepts

**Implementation:**
```python
state = {
    "target": {"math": 5, "science": 5, "english": 5},
    "current": {"math": [], "science": [], "english": []},
    "difficulty_distribution": {"easy": 0.3, "medium": 0.5, "hard": 0.2}
}

# Parallel generation with specialist agents
# Coordinator checks balance and iterates
```

---

# QUICK REFERENCE: KEY NUMBERS

| Metric | Target Value |
|--------|--------------|
| Vector search latency | < 10ms |
| Cross-encoder re-ranking | 50-100ms |
| LLM API TTFT | 500ms-2s |
| Semantic cache hit rate | 30-50% |
| Llama 70B 4-bit memory | ~35GB |
| vLLM throughput | ~500 tok/s per A100 |
| Cost savings (self-hosted) | 90-95% |

---

*Interview Questions Bank for Santosh Kumar*
*System Design for AI - Tier-1 Tech Companies*
