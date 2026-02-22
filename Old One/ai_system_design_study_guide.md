# Deep Dive System Design Study Guide for AI
## For Lead Backend Engineers Targeting Tier-1 Tech Companies

---

# DOMAIN 1: Advanced RAG System Design
*Focus: Chanakya Learning Use Case (Student Uploads → OCR → RAG → Feedback)*

---

## 1. Architecture Deep Dive: Naive RAG vs. Modular RAG

### Naive RAG (What You've Likely Built)

```
┌─────────────────────────────────────────────────────────────┐
│                      NAIVE RAG PIPELINE                      │
├─────────────────────────────────────────────────────────────┤
│  Document → Fixed Chunk → Embed → Vector DB → Top-K → LLM  │
└─────────────────────────────────────────────────────────────┘
```

**Problems at Scale (10M+ documents):**
- **Retrieval Degradation**: Top-K returns semantically similar but contextually irrelevant chunks
- **Lost Context**: Fixed chunking breaks logical document structure
- **No Quality Gates**: Garbage in, garbage out - no validation layer
- **Single Retrieval Shot**: No iterative refinement based on initial results

### Modular RAG (Production Architecture)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        MODULAR RAG ARCHITECTURE                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────┐    ┌──────────────────────────────────────────────────┐ │
│  │   INDEXING   │    │                 RETRIEVAL MODULE                 │ │
│  │   MODULE     │    ├──────────────────────────────────────────────────┤ │
│  ├─────────────┤    │  Query Transformation                             │ │
│  │ • Semantic  │    │       ↓                                           │ │
│  │   Chunking  │    │  Hybrid Search (BM25 + Vector)                    │ │
│  │ • Metadata  │    │       ↓                                           │ │
│  │   Extraction│    │  Cross-Encoder Re-ranking                         │ │
│  │ • Parent Doc│    │       ↓                                           │ │
│  │   Indexing  │    │  Relevance Threshold Filter                       │ │
│  └─────────────┘    └──────────────────────────────────────────────────┘ │
│         ↓                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                        GENERATION MODULE                             │ │
│  ├─────────────────────────────────────────────────────────────────────┤ │
│  │  Context Compression → Prompt Assembly → LLM → Groundedness Check   │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
```

**Why This Matters for Chanakya:**
- Student uploads are messy (handwritten → OCR errors, mixed formatting)
- You need multiple retrieval strategies for different content types
- Feedback quality directly impacts learning outcomes (high stakes)

---

## 2. Key Patterns to Study

### 2.1 Indexing Strategies

#### Fixed-Size Chunking (Baseline)
```python
# Your current approach (likely)
def fixed_chunk(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks
```

**Trade-offs:**
| Metric | Fixed Chunking | When to Use |
|--------|----------------|-------------|
| **Latency** | ✅ Lowest | Time-critical pipelines |
| **Accuracy** | ❌ Lowest | Simple Q&A, FAQ bots |
| **Cost** | ✅ Lowest | High-volume, low-stakes |

#### Semantic Chunking (Recommended for Chanakya)
```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

def semantic_chunk(text: str) -> list[str]:
    """
    Chunks based on semantic similarity between sentences.
    Breakpoints occur where embedding distance exceeds threshold.
    """
    chunker = SemanticChunker(
        embeddings=OpenAIEmbeddings(),
        breakpoint_threshold_type="percentile",  # or "standard_deviation", "interquartile"
        breakpoint_threshold_amount=95  # Top 5% dissimilarity = breakpoint
    )
    return chunker.create_documents([text])
```

**How It Works Internally:**
1. Embed each sentence
2. Calculate cosine distance between consecutive sentence embeddings
3. Create breakpoint where distance > threshold
4. Result: Semantically coherent chunks

#### Parent Document Retrieval (Best for Educational Content)
```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma

def setup_parent_doc_retriever():
    """
    Index small chunks for precision, retrieve parent docs for context.
    Critical for: Student answers that need full question context.
    """
    vectorstore = Chroma(collection_name="child_chunks", embedding_function=embeddings)
    store = InMemoryStore()  # Use Redis/Postgres in production
    
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)  # Small for retrieval
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)  # Large for context
    
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    return retriever
```

**Why This Pattern Wins:**
- Small chunks = High retrieval precision
- Parent docs = Full context for LLM generation
- Perfect for: "Find the specific error in student's answer, but show full solution"

### 2.2 Retrieval: Hybrid Search + Re-ranking

#### The Two-Stage Retrieval Pattern
```
┌─────────────────────────────────────────────────────────────┐
│           STAGE 1: CANDIDATE GENERATION (Fast)              │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────┐      ┌───────────────┐                   │
│  │   BM25/TF-IDF │ ───→ │   Top 50      │                   │
│  │   (Keyword)   │      │   Candidates  │                   │
│  └───────────────┘      └───────┬───────┘                   │
│                                 │                            │
│  ┌───────────────┐              │                            │
│  │   Vector      │ ───→ ────────┴────────→ Merge + Dedupe   │
│  │   (Semantic)  │                              │            │
│  └───────────────┘                              ↓            │
├─────────────────────────────────────────────────────────────┤
│           STAGE 2: RE-RANKING (Accurate but Slow)           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────┐                    │
│  │   Cross-Encoder Re-ranker           │                    │
│  │   (e.g., ms-marco-MiniLM-L-12-v2)   │                    │
│  │                                     │                    │
│  │   Input: (query, candidate) pairs   │                    │
│  │   Output: Relevance score 0-1       │                    │
│  └─────────────────────────────────────┘                    │
│                        ↓                                     │
│              Return Top 5-10 Re-ranked                       │
└─────────────────────────────────────────────────────────────┘
```

#### Implementation with LangChain
```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever

def create_hybrid_retriever(documents: list, vectorstore):
    """
    Production hybrid retrieval with re-ranking.
    """
    # Stage 1: Dual retrieval
    bm25_retriever = BM25Retriever.from_documents(documents, k=25)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 25})
    
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]  # Tune based on your data
    )
    
    # Stage 2: Cross-encoder re-ranking
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-12-v2")
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=5)
    
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble
    )
```

**Why Hybrid Beats Pure Vector:**
| Query Type | BM25 Wins | Vector Wins |
|------------|-----------|-------------|
| "Error code E-4023" | ✅ Exact match | ❌ Semantic drift |
| "How to fix login issues" | ❌ No keyword match | ✅ Intent understanding |
| "quadratic formula mistakes" | ⚖️ Both needed | ⚖️ Both needed |

### 2.3 Optimization: Lost in the Middle & Context Management

#### The "Lost in the Middle" Problem
```
┌────────────────────────────────────────────────────────────────┐
│              LLM ATTENTION DISTRIBUTION                         │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Attention                                                      │
│     ▲                                                           │
│  100%├──●                                            ●──        │
│     │   \                                          /            │
│  75%│    \                                        /             │
│     │     \                                      /              │
│  50%│      \                                    /               │
│     │       \                                  /                │
│  25%│        \●────────────────────────────●─/                  │
│     │                                                           │
│   0%└────────────────────────────────────────────────────────   │
│      Beginning    ←── LOST IN MIDDLE ──→        End             │
│      of context                                 of context      │
└────────────────────────────────────────────────────────────────┘
```

**Mitigation Strategies:**

```python
def reorder_chunks_for_llm(chunks: list[str], query: str) -> list[str]:
    """
    Place most relevant chunks at START and END of context.
    Middle chunks should be supporting/less critical.
    """
    # Assume chunks are sorted by relevance score (descending)
    n = len(chunks)
    reordered = []
    
    # Interleave: best at edges, worst in middle
    for i in range(n):
        if i % 2 == 0:
            reordered.insert(0, chunks[i])  # Prepend (start)
        else:
            reordered.append(chunks[i])     # Append (end)
    
    return reordered

# Alternative: Use LangChain's LongContextReorder
from langchain.document_transformers import LongContextReorder
reordering = LongContextReorder()
reordered_docs = reordering.transform_documents(docs)
```

#### Context Window Management
```python
import tiktoken

def manage_context_budget(
    chunks: list[str],
    query: str,
    max_context_tokens: int = 6000,  # Leave room for query + response
    model: str = "gpt-4"
) -> list[str]:
    """
    Fit maximum relevant chunks within token budget.
    """
    encoder = tiktoken.encoding_for_model(model)
    
    selected = []
    current_tokens = len(encoder.encode(query))
    
    for chunk in chunks:
        chunk_tokens = len(encoder.encode(chunk))
        if current_tokens + chunk_tokens <= max_context_tokens:
            selected.append(chunk)
            current_tokens += chunk_tokens
        else:
            break  # Stop when budget exhausted
    
    return selected
```

---

## 3. Interview Question & Model Answer

### Question:
> "How would you reduce hallucination rates in the Chanakya student feedback system?"

### Model Answer (Technical):

**Framework: Defense in Depth (4 Layers)**

```
┌──────────────────────────────────────────────────────────────┐
│                 ANTI-HALLUCINATION STACK                      │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Layer 1: RETRIEVAL QUALITY                                   │
│  ├─ Hybrid search (BM25 + Vector) for precision               │
│  ├─ Cross-encoder re-ranking (threshold > 0.7)                │
│  └─ If no relevant chunks found → Return "I don't know"       │
│                                                               │
│  Layer 2: PROMPT ENGINEERING                                  │
│  ├─ Citation-based prompting                                  │
│  │   "Answer ONLY using [SOURCE 1], [SOURCE 2]..."            │
│  ├─ Explicit instruction: "If unsure, say 'I cannot verify'"  │
│  └─ Few-shot examples of correct + rejected responses         │
│                                                               │
│  Layer 3: GENERATION CONSTRAINTS                              │
│  ├─ Temperature = 0.0-0.2 (deterministic)                     │
│  ├─ Structured output (JSON mode) for feedback                │
│  └─ Max tokens limit to prevent rambling                      │
│                                                               │
│  Layer 4: POST-GENERATION VALIDATION                          │
│  ├─ Groundedness Check (NLI classifier)                       │
│  │   • Premise: Retrieved chunks                              │
│  │   • Hypothesis: Each generated sentence                    │
│  │   • Score: Entailment probability                          │
│  └─ RAGAS metrics for offline evaluation                      │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

**Implementation - Groundedness Check:**
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class GroundednessChecker:
    """
    Uses NLI (Natural Language Inference) to verify each claim.
    """
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "facebook/bart-large-mnli"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
    
    def check(self, context: str, generated_claim: str) -> float:
        """
        Returns probability that claim is entailed by context.
        Threshold: > 0.7 = grounded, < 0.5 = hallucination
        """
        inputs = self.tokenizer(
            context, generated_claim,
            return_tensors="pt", truncation=True, max_length=512
        )
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        
        # Labels: 0=contradiction, 1=neutral, 2=entailment
        entailment_score = probs[0][2].item()
        return entailment_score

def validate_feedback(context: str, generated_feedback: str) -> dict:
    """
    Production validation pipeline.
    """
    checker = GroundednessChecker()
    sentences = generated_feedback.split(". ")
    
    results = []
    for sentence in sentences:
        score = checker.check(context, sentence)
        results.append({
            "sentence": sentence,
            "grounded": score > 0.7,
            "score": score
        })
    
    # If any claim is ungrounded, flag for human review
    overall_grounded = all(r["grounded"] for r in results)
    return {"grounded": overall_grounded, "details": results}
```

**RAGAS Evaluation (Offline):**
```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,      # Is response grounded in context?
    answer_relevancy,  # Does response address the question?
    context_precision, # Are retrieved chunks relevant?
    context_recall     # Did we retrieve all needed info?
)

def evaluate_rag_pipeline(test_dataset):
    """
    Run on a golden test set (human-labeled Q&A pairs).
    """
    result = evaluate(
        test_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
    )
    return result.to_pandas()
```

---

## Key Takeaways for the Interview (Domain 1)

| Concept | What to Say | What NOT to Say |
|---------|-------------|-----------------|
| **Chunking** | "I'd use semantic chunking for student essays, parent-doc retrieval for solutions" | "I use 512-token fixed chunks" |
| **Retrieval** | "Hybrid BM25 + Vector with cross-encoder re-ranking" | "Just cosine similarity on embeddings" |
| **Hallucination** | "4-layer defense: retrieval quality, citation prompts, low temp, NLI groundedness" | "I use temperature = 0" |
| **Evaluation** | "RAGAS for offline, groundedness checks for online" | "We check manually" |
| **Trade-off** | "Re-ranking adds 50-100ms latency but improves precision by 15-20%" | No mention of trade-offs |

---

# DOMAIN 2: AI Agent Architecture & Orchestration
*Focus: B2B Teacher Dashboard (Automated Exam Creation)*

---

## 1. Architecture Deep Dive: The ReAct Pattern

### What is ReAct (Reasoning + Acting)?

ReAct is the foundational pattern for autonomous AI agents. It alternates between:
- **Reasoning**: LLM thinks about what to do next
- **Acting**: LLM calls a tool/function
- **Observing**: System returns tool output
- **Repeat**: Until task complete or max iterations

```
┌───────────────────────────────────────────────────────────────┐
│                    ReAct LOOP ARCHITECTURE                     │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│   User Query: "Create a 10-question math exam on Quadratics"  │
│                           │                                    │
│                           ▼                                    │
│   ┌─────────────────────────────────────────────────────────┐ │
│   │  WHILE not (task_complete OR max_iterations):           │ │
│   │  │                                                      │ │
│   │  │  1. REASON: LLM generates thought + action          │ │
│   │  │     "I need to search the question bank for          │ │
│   │  │      quadratic equations. Action: search_questions"  │ │
│   │  │                           │                          │ │
│   │  │                           ▼                          │ │
│   │  │  2. ACT: Execute tool call                          │ │
│   │  │     search_questions(topic="quadratics", limit=20)  │ │
│   │  │                           │                          │ │
│   │  │                           ▼                          │ │
│   │  │  3. OBSERVE: Receive tool output                    │ │
│   │  │     [Question objects returned...]                  │ │
│   │  │                           │                          │ │
│   │  │                           ▼                          │ │
│   │  │  4. UPDATE: Add observation to context              │ │
│   │  │     Continue loop with updated state                │ │
│   │  │                                                      │ │
│   └──┴──────────────────────────────────────────────────────┘ │
│                           │                                    │
│                           ▼                                    │
│              Final Answer: Generated Exam                      │
└───────────────────────────────────────────────────────────────┘
```

### Python Implementation (Production-Grade)

```python
from dataclasses import dataclass, field
from typing import Callable, Any
from enum import Enum
import json

class AgentState(Enum):
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    COMPLETE = "complete"
    FAILED = "failed"

@dataclass
class AgentMessage:
    role: str  # "user", "assistant", "tool"
    content: str
    tool_call_id: str | None = None
    tool_name: str | None = None

@dataclass
class AgentContext:
    """Maintains full agent state for persistence/resumption."""
    messages: list[AgentMessage] = field(default_factory=list)
    current_state: AgentState = AgentState.THINKING
    iteration: int = 0
    max_iterations: int = 10
    
    def to_dict(self) -> dict:
        """Serialize for persistence (Postgres/Redis)."""
        return {
            "messages": [vars(m) for m in self.messages],
            "current_state": self.current_state.value,
            "iteration": self.iteration,
            "max_iterations": self.max_iterations
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AgentContext":
        """Hydrate from stored state."""
        ctx = cls()
        ctx.messages = [AgentMessage(**m) for m in data["messages"]]
        ctx.current_state = AgentState(data["current_state"])
        ctx.iteration = data["iteration"]
        ctx.max_iterations = data["max_iterations"]
        return ctx

class ReActAgent:
    """
    Production ReAct agent with:
    - Retry logic for tool failures
    - State persistence for pause/resume
    - Configurable stopping conditions
    """
    
    def __init__(
        self,
        llm_client,  # OpenAI client or similar
        tools: dict[str, Callable],
        tool_schemas: list[dict],  # OpenAI function schemas
        system_prompt: str,
        max_iterations: int = 10,
        max_retries: int = 3
    ):
        self.llm = llm_client
        self.tools = tools
        self.tool_schemas = tool_schemas
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.max_retries = max_retries
    
    def run(self, user_query: str, context: AgentContext | None = None) -> AgentContext:
        """
        Main ReAct loop. Can resume from existing context.
        """
        if context is None:
            context = AgentContext(max_iterations=self.max_iterations)
            context.messages.append(AgentMessage(role="user", content=user_query))
        
        while context.iteration < context.max_iterations:
            context.iteration += 1
            
            # REASON: Get LLM response with potential tool call
            context.current_state = AgentState.THINKING
            response = self._call_llm(context)
            
            # Check if task complete (no tool call)
            if not response.tool_calls:
                context.messages.append(
                    AgentMessage(role="assistant", content=response.content)
                )
                context.current_state = AgentState.COMPLETE
                return context
            
            # ACT: Execute tool calls
            context.current_state = AgentState.ACTING
            for tool_call in response.tool_calls:
                observation = self._execute_tool_with_retry(
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments)
                )
                
                # OBSERVE: Add result to context
                context.current_state = AgentState.OBSERVING
                context.messages.append(AgentMessage(
                    role="tool",
                    content=observation,
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.function.name
                ))
        
        context.current_state = AgentState.FAILED
        return context
    
    def _execute_tool_with_retry(self, tool_name: str, args: dict) -> str:
        """
        Execute tool with exponential backoff retry.
        Critical for production: APIs fail!
        """
        for attempt in range(self.max_retries):
            try:
                result = self.tools[tool_name](**args)
                return json.dumps(result) if not isinstance(result, str) else result
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return json.dumps({
                        "error": str(e),
                        "suggestion": "Tool failed after retries. Consider alternative approach."
                    })
                import time
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
        
        return json.dumps({"error": "Max retries exceeded"})
    
    def _call_llm(self, context: AgentContext):
        """Format messages and call LLM."""
        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in context.messages:
            messages.append({"role": msg.role, "content": msg.content})
        
        return self.llm.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=self.tool_schemas,
            tool_choice="auto"
        )
```

---

## 2. Key Patterns to Study

### 2.1 Memory Management

#### Short-Term Memory (Conversation Buffer)
```python
from collections import deque

class ConversationBufferMemory:
    """
    Fixed-size sliding window of recent messages.
    Use for: Current conversation context.
    """
    def __init__(self, max_messages: int = 20):
        self.buffer = deque(maxlen=max_messages)
    
    def add(self, message: AgentMessage):
        self.buffer.append(message)
    
    def get_context(self) -> list[AgentMessage]:
        return list(self.buffer)
    
    def clear(self):
        self.buffer.clear()

class TokenBufferMemory:
    """
    Keeps messages until token limit exceeded.
    More precise than message count.
    """
    def __init__(self, max_tokens: int = 4000, model: str = "gpt-4"):
        import tiktoken
        self.max_tokens = max_tokens
        self.encoder = tiktoken.encoding_for_model(model)
        self.messages = []
    
    def add(self, message: AgentMessage):
        self.messages.append(message)
        self._trim_to_limit()
    
    def _trim_to_limit(self):
        while self._total_tokens() > self.max_tokens and self.messages:
            self.messages.pop(0)  # Remove oldest
    
    def _total_tokens(self) -> int:
        return sum(len(self.encoder.encode(m.content)) for m in self.messages)
```

#### Long-Term Memory (Vector Store)
```python
from datetime import datetime
import uuid

class LongTermMemory:
    """
    Persistent memory using vector store.
    Use for: Past conversations, learned facts, user preferences.
    """
    def __init__(self, vectorstore, embedding_model):
        self.vectorstore = vectorstore
        self.embedder = embedding_model
    
    def store(self, content: str, metadata: dict | None = None):
        """Store a memory with auto-generated ID and timestamp."""
        doc_id = str(uuid.uuid4())
        meta = metadata or {}
        meta.update({
            "timestamp": datetime.utcnow().isoformat(),
            "doc_id": doc_id
        })
        
        self.vectorstore.add_texts(
            texts=[content],
            metadatas=[meta],
            ids=[doc_id]
        )
        return doc_id
    
    def recall(self, query: str, k: int = 5, filter_metadata: dict | None = None) -> list[str]:
        """Retrieve relevant memories."""
        results = self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter_metadata
        )
        return [doc.page_content for doc in results]
    
    def forget(self, doc_id: str):
        """Remove a specific memory."""
        self.vectorstore.delete([doc_id])

# Usage in Agent
class AgentWithMemory:
    def __init__(self, ...):
        self.short_term = ConversationBufferMemory(max_messages=20)
        self.long_term = LongTermMemory(vectorstore, embedder)
    
    def run(self, query: str):
        # Recall relevant long-term memories
        relevant_memories = self.long_term.recall(query, k=3)
        
        # Inject into system prompt
        memory_context = "\n".join([f"- {m}" for m in relevant_memories])
        enhanced_prompt = f"{self.system_prompt}\n\nRelevant memories:\n{memory_context}"
        
        # ... rest of agent logic
```

### 2.2 Tool Use: Handling Failures

```python
from dataclasses import dataclass
from typing import Literal
import traceback

@dataclass
class ToolResult:
    status: Literal["success", "error", "retry"]
    data: Any
    error_message: str | None = None
    retry_suggestion: str | None = None

class RobustToolExecutor:
    """
    Production tool executor with:
    - Timeout handling
    - Graceful degradation
    - Informative error messages for LLM
    """
    
    def __init__(self, tools: dict[str, Callable], timeout_seconds: int = 30):
        self.tools = tools
        self.timeout = timeout_seconds
    
    def execute(self, tool_name: str, args: dict) -> ToolResult:
        if tool_name not in self.tools:
            return ToolResult(
                status="error",
                data=None,
                error_message=f"Unknown tool: {tool_name}. Available: {list(self.tools.keys())}"
            )
        
        try:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Tool {tool_name} exceeded {self.timeout}s timeout")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout)
            
            result = self.tools[tool_name](**args)
            signal.alarm(0)  # Cancel alarm
            
            return ToolResult(status="success", data=result)
            
        except TimeoutError as e:
            return ToolResult(
                status="retry",
                data=None,
                error_message=str(e),
                retry_suggestion="Consider breaking this into smaller operations"
            )
        except ValueError as e:
            return ToolResult(
                status="error",
                data=None,
                error_message=f"Invalid arguments: {e}",
                retry_suggestion="Check argument types and formats"
            )
        except Exception as e:
            return ToolResult(
                status="error",
                data=None,
                error_message=f"{type(e).__name__}: {e}",
                retry_suggestion="Unexpected error. Try alternative approach."
            )
```

### 2.3 Multi-Agent Patterns

#### Router Pattern (Simple)
```
┌────────────────────────────────────────────────────────────┐
│                     ROUTER PATTERN                          │
├────────────────────────────────────────────────────────────┤
│                                                             │
│                      ┌──────────────┐                       │
│    User Query ────→  │   ROUTER     │                       │
│                      │   (LLM)      │                       │
│                      └──────┬───────┘                       │
│                             │                               │
│           ┌─────────────────┼─────────────────┐             │
│           ▼                 ▼                 ▼             │
│   ┌───────────────┐ ┌───────────────┐ ┌───────────────┐    │
│   │  Question     │ │  Grading      │ │  Curriculum   │    │
│   │  Generator    │ │  Agent        │ │  Planner      │    │
│   └───────────────┘ └───────────────┘ └───────────────┘    │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

```python
class RouterAgent:
    """
    Routes queries to specialized sub-agents.
    Use when: Tasks are clearly separable, no inter-agent communication needed.
    """
    
    def __init__(self, agents: dict[str, ReActAgent], router_llm):
        self.agents = agents
        self.router_llm = router_llm
        self.routing_prompt = """
        Classify the user request into one of these categories:
        - question_generation: Creating exams, quizzes, practice problems
        - grading: Evaluating student answers, providing feedback
        - curriculum: Planning lessons, creating syllabi
        
        Respond with ONLY the category name.
        """
    
    def route(self, query: str) -> str:
        response = self.router_llm.chat.completions.create(
            model="gpt-3.5-turbo",  # Fast, cheap for routing
            messages=[
                {"role": "system", "content": self.routing_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    
    def run(self, query: str) -> AgentContext:
        category = self.route(query)
        if category not in self.agents:
            category = "question_generation"  # Default fallback
        
        return self.agents[category].run(query)
```

#### State Graph Pattern (LangGraph Style)
```
┌─────────────────────────────────────────────────────────────────┐
│                    STATE GRAPH PATTERN                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐                                                │
│   │   START     │                                                │
│   └──────┬──────┘                                                │
│          │                                                       │
│          ▼                                                       │
│   ┌─────────────┐    questions_ready     ┌─────────────┐        │
│   │  Generate   │ ─────────────────────→ │   Review    │        │
│   │  Questions  │                        │   Quality   │        │
│   └─────────────┘                        └──────┬──────┘        │
│          ▲                                      │                │
│          │                                      │                │
│          │ needs_more                   ┌───────┴───────┐       │
│          │                              ▼               ▼       │
│          │                         approved         rejected    │
│          │                              │               │       │
│          │                              ▼               │       │
│          │                       ┌─────────────┐        │       │
│          │                       │   Format    │        │       │
│          │                       │   Output    │        │       │
│          └───────────────────────┤             │◄───────┘       │
│                                  └──────┬──────┘                │
│                                         │                        │
│                                         ▼                        │
│                                  ┌─────────────┐                │
│                                  │    END      │                │
│                                  └─────────────┘                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
from typing import TypedDict, Literal
from dataclasses import dataclass

class ExamState(TypedDict):
    """Shared state passed between nodes."""
    topic: str
    difficulty: str
    num_questions: int
    questions: list[dict]
    review_feedback: str | None
    final_output: str | None
    iteration: int

@dataclass
class GraphNode:
    name: str
    fn: Callable[[ExamState], ExamState]

class StateGraph:
    """
    Simplified LangGraph-style state machine.
    Each node transforms state, edges determine next node.
    """
    
    def __init__(self):
        self.nodes: dict[str, GraphNode] = {}
        self.edges: dict[str, dict[str, str]] = {}  # node -> {condition: next_node}
        self.entry_point: str | None = None
    
    def add_node(self, name: str, fn: Callable):
        self.nodes[name] = GraphNode(name=name, fn=fn)
    
    def add_edge(self, from_node: str, to_node: str, condition: str = "default"):
        if from_node not in self.edges:
            self.edges[from_node] = {}
        self.edges[from_node][condition] = to_node
    
    def add_conditional_edges(self, from_node: str, router_fn: Callable[[ExamState], str]):
        """Router function returns the condition string."""
        self.edges[from_node] = {"_router": router_fn}
    
    def set_entry_point(self, node: str):
        self.entry_point = node
    
    def run(self, initial_state: ExamState, max_iterations: int = 10) -> ExamState:
        state = initial_state.copy()
        current_node = self.entry_point
        iterations = 0
        
        while current_node and current_node != "END" and iterations < max_iterations:
            iterations += 1
            state["iteration"] = iterations
            
            # Execute node
            node = self.nodes[current_node]
            state = node.fn(state)
            
            # Determine next node
            edge_config = self.edges.get(current_node, {})
            
            if "_router" in edge_config:
                condition = edge_config["_router"](state)
                current_node = self.edges[current_node].get(condition, "END")
            elif "default" in edge_config:
                current_node = edge_config["default"]
            else:
                current_node = "END"
        
        return state

# Example: Exam Creation Graph
def create_exam_graph() -> StateGraph:
    graph = StateGraph()
    
    def generate_questions(state: ExamState) -> ExamState:
        # Call LLM to generate questions
        questions = llm_generate_questions(
            topic=state["topic"],
            difficulty=state["difficulty"],
            count=state["num_questions"]
        )
        state["questions"] = questions
        return state
    
    def review_quality(state: ExamState) -> ExamState:
        # LLM reviews question quality
        feedback = llm_review_questions(state["questions"])
        state["review_feedback"] = feedback
        return state
    
    def format_output(state: ExamState) -> ExamState:
        # Format final exam document
        state["final_output"] = format_exam(state["questions"])
        return state
    
    def quality_router(state: ExamState) -> str:
        if "approved" in state["review_feedback"].lower():
            return "approved"
        elif state["iteration"] >= 3:
            return "approved"  # After 3 tries, accept anyway
        return "rejected"
    
    graph.add_node("generate", generate_questions)
    graph.add_node("review", review_quality)
    graph.add_node("format", format_output)
    
    graph.add_edge("generate", "review")
    graph.add_conditional_edges("review", quality_router)
    graph.edges["review"]["approved"] = "format"
    graph.edges["review"]["rejected"] = "generate"
    graph.add_edge("format", "END")
    
    graph.set_entry_point("generate")
    
    return graph
```

---

## 3. Interview Question & Model Answer

### Question:
> "Design a stateful Agent that can pause a task (e.g., grading 1000 assignments) and resume it days later. How do you persist the state?"

### Model Answer:

**Architecture: Checkpointer Pattern**

```
┌────────────────────────────────────────────────────────────────────┐
│               STATEFUL AGENT PERSISTENCE ARCHITECTURE              │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────────────┐      ┌──────────────────────────────┐  │
│   │     AGENT RUNTIME    │      │     PERSISTENCE LAYER        │  │
│   ├──────────────────────┤      ├──────────────────────────────┤  │
│   │                      │      │                               │  │
│   │  AgentState          │◄────►│  Postgres (JSONB)            │  │
│   │  ├─ messages[]       │      │  ├─ agent_id (PK)            │  │
│   │  ├─ current_step     │      │  ├─ user_id                  │  │
│   │  ├─ iteration        │      │  ├─ state_json               │  │
│   │  ├─ partial_results  │      │  ├─ created_at               │  │
│   │  └─ checkpoint_id    │      │  ├─ updated_at               │  │
│   │                      │      │  └─ status (enum)            │  │
│   └──────────────────────┘      └──────────────────────────────┘  │
│                                                                     │
│   WORKFLOW:                                                         │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │  1. User starts task → Create agent_id, init state         │  │
│   │  2. Every N iterations → Checkpoint (save to Postgres)      │  │
│   │  3. User pauses/timeout → Save final checkpoint             │  │
│   │  4. Resume → Load checkpoint → Hydrate state → Continue     │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
import json
import uuid
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Any
import psycopg2
from psycopg2.extras import Json

class TaskStatus(Enum):
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class GradingState:
    """Complete state for grading task."""
    task_id: str
    total_assignments: int
    completed_count: int
    current_index: int
    grades: dict[str, dict]  # assignment_id -> {grade, feedback}
    messages: list[dict]  # Conversation history
    partial_results: list[dict]
    error_log: list[str]
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, json_str: str) -> "GradingState":
        data = json.loads(json_str)
        return cls(**data)

class PostgresCheckpointer:
    """
    Persistent checkpointing with Postgres.
    Supports: save, load, list checkpoints, cleanup.
    """
    
    def __init__(self, connection_string: str):
        self.conn = psycopg2.connect(connection_string)
        self._init_table()
    
    def _init_table(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS agent_checkpoints (
                    id VARCHAR(36) PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    task_type VARCHAR(100) NOT NULL,
                    state_json JSONB NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    checkpoint_version INT DEFAULT 1,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    metadata JSONB
                );
                CREATE INDEX IF NOT EXISTS idx_user_status 
                    ON agent_checkpoints(user_id, status);
            """)
            self.conn.commit()
    
    def save(self, state: GradingState, user_id: str, task_type: str, 
             status: TaskStatus, metadata: dict | None = None) -> str:
        """Save or update checkpoint."""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO agent_checkpoints 
                    (id, user_id, task_type, state_json, status, metadata, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (id) DO UPDATE SET
                    state_json = EXCLUDED.state_json,
                    status = EXCLUDED.status,
                    checkpoint_version = agent_checkpoints.checkpoint_version + 1,
                    updated_at = NOW()
            """, (state.task_id, user_id, task_type, 
                  Json(json.loads(state.to_json())), status.value, 
                  Json(metadata or {})))
            self.conn.commit()
        return state.task_id
    
    def load(self, task_id: str) -> GradingState | None:
        """Load checkpoint by task_id."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT state_json FROM agent_checkpoints WHERE id = %s",
                (task_id,)
            )
            row = cur.fetchone()
            if row:
                return GradingState.from_json(json.dumps(row[0]))
        return None
    
    def list_paused(self, user_id: str) -> list[dict]:
        """List all paused tasks for a user."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT id, task_type, 
                       state_json->>'completed_count' as completed,
                       state_json->>'total_assignments' as total,
                       updated_at
                FROM agent_checkpoints 
                WHERE user_id = %s AND status = 'paused'
                ORDER BY updated_at DESC
            """, (user_id,))
            return [
                {"task_id": r[0], "type": r[1], "progress": f"{r[2]}/{r[3]}", 
                 "paused_at": r[4]}
                for r in cur.fetchall()
            ]

class ResumableGradingAgent:
    """
    Agent that can pause and resume grading tasks.
    """
    
    def __init__(self, llm_client, checkpointer: PostgresCheckpointer,
                 checkpoint_interval: int = 10):
        self.llm = llm_client
        self.checkpointer = checkpointer
        self.checkpoint_interval = checkpoint_interval
        self._should_pause = False
    
    def request_pause(self):
        """External signal to pause (e.g., from API endpoint)."""
        self._should_pause = True
    
    def start_grading(self, user_id: str, assignments: list[dict]) -> str:
        """Start a new grading task."""
        state = GradingState(
            task_id=str(uuid.uuid4()),
            total_assignments=len(assignments),
            completed_count=0,
            current_index=0,
            grades={},
            messages=[],
            partial_results=[],
            error_log=[]
        )
        
        self.checkpointer.save(state, user_id, "grading", TaskStatus.RUNNING)
        return self._run_grading_loop(state, user_id, assignments)
    
    def resume_grading(self, task_id: str, user_id: str, 
                       assignments: list[dict]) -> str:
        """Resume a paused grading task."""
        state = self.checkpointer.load(task_id)
        if not state:
            raise ValueError(f"No checkpoint found for task {task_id}")
        
        # Continue from where we left off
        return self._run_grading_loop(state, user_id, assignments)
    
    def _run_grading_loop(self, state: GradingState, user_id: str,
                          assignments: list[dict]) -> str:
        """Main grading loop with checkpointing."""
        
        for i in range(state.current_index, len(assignments)):
            # Check for pause signal
            if self._should_pause:
                state.current_index = i
                self.checkpointer.save(state, user_id, "grading", TaskStatus.PAUSED)
                self._should_pause = False
                return f"PAUSED at {i}/{len(assignments)}"
            
            # Grade assignment
            assignment = assignments[i]
            try:
                grade_result = self._grade_single(assignment, state.messages)
                state.grades[assignment["id"]] = grade_result
                state.completed_count += 1
                state.current_index = i + 1
                
            except Exception as e:
                state.error_log.append(f"Assignment {assignment['id']}: {e}")
            
            # Periodic checkpoint
            if (i + 1) % self.checkpoint_interval == 0:
                self.checkpointer.save(state, user_id, "grading", TaskStatus.RUNNING)
        
        # Final save
        self.checkpointer.save(state, user_id, "grading", TaskStatus.COMPLETED)
        return f"COMPLETED: {state.completed_count}/{state.total_assignments}"
    
    def _grade_single(self, assignment: dict, history: list[dict]) -> dict:
        """Grade a single assignment using LLM."""
        # ... LLM grading logic
        pass
```

**Key Design Decisions:**

| Decision | Rationale |
|----------|-----------|
| **Postgres JSONB** | Flexible schema, queryable, battle-tested for production |
| **Checkpoint interval** | Balance between durability and performance (every 10 items) |
| **External pause signal** | Non-blocking; agent checks flag each iteration |
| **Version tracking** | `checkpoint_version` for debugging and audit |
| **Separate error log** | Don't lose progress on individual failures |

---

## Key Takeaways for the Interview (Domain 2)

| Concept | What to Say | What NOT to Say |
|---------|-------------|-----------------|
| **ReAct Loop** | "While loop with Reason→Act→Observe phases, max iterations, and explicit stopping conditions" | "The LLM figures it out" |
| **Memory** | "Short-term (token buffer) for context, long-term (vector store) for recall" | "I keep all messages" |
| **Tool Failures** | "Exponential backoff retry, timeout handling, informative error messages for LLM recovery" | "I catch exceptions" |
| **Multi-Agent** | "Router for independent tasks, State Graph for complex workflows with conditional edges" | "I chain multiple prompts" |
| **Persistence** | "Postgres JSONB with periodic checkpointing, hydration on resume" | "Redis with TTL" (loses data) |

---

# DOMAIN 3: LLM Operations (LLMOps) & Inference
*Focus: Scaling Chanakya AI to 100K Users with Cost Optimization*

---

## 1. Key Concepts to Study

### 1.1 Latency vs. Throughput Trade-offs

```
┌────────────────────────────────────────────────────────────────────┐
│            LATENCY vs THROUGHPUT OPTIMIZATION SPECTRUM             │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ← LOW LATENCY                              HIGH THROUGHPUT →     │
│                                                                     │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│   │  Streaming   │    │  Small Batch │    │  Large Batch │        │
│   │  (batch=1)   │    │  (batch=4-8) │    │  (batch=32+) │        │
│   └──────────────┘    └──────────────┘    └──────────────┘        │
│                                                                     │
│   Use Case:          Use Case:           Use Case:                 │
│   • Interactive      • API endpoints    • Offline grading          │
│   • Chat             • Student feedback • Bulk processing          │
│   • Real-time        • Balance latency  • Cost optimization        │
│                        and throughput                               │
│                                                                     │
│   TTFT: ~100ms       TTFT: ~200-500ms   TTFT: ~1-5s               │
│   Throughput: Low    Throughput: Med    Throughput: High          │
│   Cost/token: High   Cost/token: Med    Cost/token: Low           │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘

TTFT = Time To First Token
```

#### Streaming Implementation
```python
from openai import OpenAI
import asyncio
from typing import AsyncGenerator

async def stream_response(prompt: str) -> AsyncGenerator[str, None]:
    """
    Stream tokens as they're generated.
    Use for: Chat interfaces, real-time feedback.
    """
    client = OpenAI()
    
    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Example: FastAPI streaming endpoint
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/chat/stream")
async def chat_stream(prompt: str):
    async def generate():
        async for token in stream_response(prompt):
            yield f"data: {token}\n\n"  # SSE format
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

#### Batching Implementation
```python
import asyncio
from dataclasses import dataclass
from typing import Callable
from collections import deque
import time

@dataclass
class BatchRequest:
    prompt: str
    future: asyncio.Future
    timestamp: float

class DynamicBatcher:
    """
    Collect requests and process in batches for throughput.
    Use for: API endpoints with moderate traffic.
    """
    
    def __init__(
        self,
        process_fn: Callable[[list[str]], list[str]],
        max_batch_size: int = 8,
        max_wait_ms: float = 100
    ):
        self.process_fn = process_fn
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue: deque[BatchRequest] = deque()
        self._lock = asyncio.Lock()
        self._processing = False
    
    async def add_request(self, prompt: str) -> str:
        """Add a request and wait for batched result."""
        future = asyncio.get_event_loop().create_future()
        request = BatchRequest(prompt=prompt, future=future, timestamp=time.time())
        
        async with self._lock:
            self.queue.append(request)
            
            if not self._processing:
                self._processing = True
                asyncio.create_task(self._process_batch())
        
        return await future
    
    async def _process_batch(self):
        """Process accumulated requests as a batch."""
        await asyncio.sleep(self.max_wait_ms / 1000)  # Wait for more requests
        
        async with self._lock:
            batch = []
            while self.queue and len(batch) < self.max_batch_size:
                batch.append(self.queue.popleft())
            self._processing = len(self.queue) > 0
        
        if batch:
            prompts = [r.prompt for r in batch]
            results = await asyncio.to_thread(self.process_fn, prompts)
            
            for request, result in zip(batch, results):
                request.future.set_result(result)
        
        if self._processing:
            asyncio.create_task(self._process_batch())
```

### 1.2 Semantic Caching (GPTCache Pattern)

```
┌─────────────────────────────────────────────────────────────────┐
│                    SEMANTIC CACHING FLOW                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   User Query: "What is the quadratic formula?"                  │
│                     │                                            │
│                     ▼                                            │
│   ┌────────────────────────────────────────┐                    │
│   │  1. Embed query                        │                    │
│   │  2. Search cache (similarity > 0.95)  │                    │
│   └────────────────────┬───────────────────┘                    │
│                        │                                         │
│            ┌───────────┴───────────┐                            │
│            ▼                       ▼                            │
│      ┌──────────┐            ┌──────────┐                       │
│      │  CACHE   │            │  CACHE   │                       │
│      │   HIT    │            │   MISS   │                       │
│      └────┬─────┘            └────┬─────┘                       │
│           │                       │                              │
│           ▼                       ▼                              │
│   Return cached answer      Call LLM → Cache result             │
│   (Latency: ~5ms)          (Latency: ~500ms+)                   │
│   (Cost: $0)               (Cost: $0.01+)                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Implementation
```python
import hashlib
import json
from datetime import datetime, timedelta
from typing import Any
import numpy as np

class SemanticCache:
    """
    Cache LLM responses based on semantic similarity.
    Saves 30-50% on repeated/similar queries.
    """
    
    def __init__(
        self,
        vectorstore,
        embedding_model,
        similarity_threshold: float = 0.95,
        ttl_hours: int = 24
    ):
        self.vectorstore = vectorstore
        self.embedder = embedding_model
        self.similarity_threshold = similarity_threshold
        self.ttl_hours = ttl_hours
    
    def _get_cache_key(self, query: str, model: str, params: dict) -> str:
        """Generate deterministic cache key."""
        key_data = {
            "query": query,
            "model": model,
            "params": params
        }
        return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def get(self, query: str, model: str = "gpt-4", params: dict = None) -> dict | None:
        """
        Check cache for semantically similar query.
        Returns cached response or None.
        """
        params = params or {}
        
        # Search by semantic similarity
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=1,
            filter={"model": model}
        )
        
        if not results:
            return None
        
        doc, score = results[0]
        
        # Check similarity threshold
        if score < self.similarity_threshold:
            return None
        
        # Check TTL
        cached_at = datetime.fromisoformat(doc.metadata.get("cached_at", ""))
        if datetime.utcnow() - cached_at > timedelta(hours=self.ttl_hours):
            return None
        
        return {
            "response": doc.metadata["response"],
            "cache_hit": True,
            "original_query": doc.page_content
        }
    
    def set(self, query: str, response: str, model: str = "gpt-4", 
            params: dict = None, usage: dict = None):
        """Cache a query-response pair."""
        params = params or {}
        
        self.vectorstore.add_texts(
            texts=[query],
            metadatas=[{
                "response": response,
                "model": model,
                "params": json.dumps(params),
                "cached_at": datetime.utcnow().isoformat(),
                "usage": json.dumps(usage or {})
            }]
        )
    
    def invalidate(self, pattern: str = None):
        """Invalidate cache entries matching pattern."""
        # Implementation depends on vectorstore capabilities
        pass

# Usage with LLM
class CachedLLM:
    def __init__(self, llm_client, cache: SemanticCache):
        self.llm = llm_client
        self.cache = cache
    
    def generate(self, prompt: str, **kwargs) -> dict:
        # Check cache first
        cached = self.cache.get(prompt, model=kwargs.get("model", "gpt-4"))
        if cached:
            return cached
        
        # Cache miss - call LLM
        response = self.llm.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        
        result = response.choices[0].message.content
        
        # Store in cache
        self.cache.set(
            query=prompt,
            response=result,
            model=kwargs.get("model", "gpt-4"),
            usage=dict(response.usage)
        )
        
        return {"response": result, "cache_hit": False}
```

### 1.3 RAG vs. Fine-Tuning Decision Framework

```
┌───────────────────────────────────────────────────────────────────────┐
│                  RAG vs FINE-TUNING DECISION MATRIX                   │
├───────────────────────────────────────────────────────────────────────┤
│                                                                        │
│                        CHOOSE RAG WHEN:                                │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │ ✅ Knowledge changes frequently (daily/weekly updates)          │  │
│  │ ✅ Need citations and source attribution                        │  │
│  │ ✅ Data is proprietary and can't be in model weights            │  │
│  │ ✅ Want to avoid hallucination through grounding                │  │
│  │ ✅ Limited training data (< 10K examples)                       │  │
│  │ ✅ Privacy regulations require data separation                  │  │
│  │                                                                  │  │
│  │ Examples: Customer support with product docs, Legal research,   │  │
│  │           Chanakya's curriculum-based Q&A                       │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│                     CHOOSE FINE-TUNING WHEN:                          │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │ ✅ Need consistent style/format (brand voice, JSON output)      │  │
│  │ ✅ Task requires specialized reasoning not in base model        │  │
│  │ ✅ Reducing prompt size/cost (behavior in weights, not prompts) │  │
│  │ ✅ High inference volume (amortize training cost)               │  │
│  │ ✅ Static knowledge that rarely changes                         │  │
│  │                                                                  │  │
│  │ Examples: Code completion, Domain-specific classification,      │  │
│  │           Structured data extraction                            │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│                     COMBINE RAG + FINE-TUNING:                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │ ✅ Fine-tune for style/format + RAG for knowledge               │  │
│  │ ✅ Fine-tune retrieval-augmented generation behavior            │  │
│  │                                                                  │  │
│  │ Example: Chanakya feedback system - fine-tune for teacher       │  │
│  │          communication style, RAG for curriculum content        │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                        │
└───────────────────────────────────────────────────────────────────────┘
```

#### PEFT/LoRA for Efficient Fine-Tuning
```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def setup_lora_finetuning(
    base_model: str = "meta-llama/Llama-3-8B",
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.1
):
    """
    LoRA: Low-Rank Adaptation
    - Trains ~0.1% of parameters
    - 10-100x less memory than full fine-tuning
    - Adapters can be swapped at inference
    """
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,                    # Rank of update matrices
        lora_alpha=alpha,          # Scaling factor
        lora_dropout=dropout,
        target_modules=[           # Which layers to adapt
            "q_proj", "k_proj", "v_proj",  # Attention
            "gate_proj", "up_proj", "down_proj"  # MLP
        ],
        bias="none"
    )
    
    # Wrap model with LoRA
    peft_model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable, total = peft_model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    
    return peft_model, tokenizer

# Training example
def train_lora_adapter(
    model,
    tokenizer,
    train_dataset,  # HuggingFace Dataset
    output_dir: str = "./lora_adapter"
):
    from transformers import TrainingArguments, Trainer
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )
    
    trainer.train()
    model.save_pretrained(output_dir)

# Inference with LoRA adapter
def load_model_with_adapter(base_model: str, adapter_path: str):
    from peft import PeftModel
    
    base = AutoModelForCausalLM.from_pretrained(base_model)
    model = PeftModel.from_pretrained(base, adapter_path)
    return model
```

---

## 2. System Design Scenario

### Scenario:
> "Scale Chanakya AI to 100,000 concurrent users. OpenAI API bill is too high. Design a solution using open-source models (Llama 3)."

### Solution Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    CHANAKYA AI - PRODUCTION ARCHITECTURE                      │
│                    (100K Concurrent Users, Open Source LLM)                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │                         LOAD BALANCER (ALB)                          │    │
│   │                              │                                       │    │
│   │              ┌───────────────┼───────────────┐                       │    │
│   │              ▼               ▼               ▼                       │    │
│   │        ┌─────────┐     ┌─────────┐     ┌─────────┐                  │    │
│   │        │ FastAPI │     │ FastAPI │     │ FastAPI │ (Auto-scaling)   │    │
│   │        │ Pod n   │     │ Pod n+1 │     │ Pod n+2 │                  │    │
│   │        └────┬────┘     └────┬────┘     └────┬────┘                  │    │
│   │             │               │               │                        │    │
│   └─────────────┴───────────────┴───────────────┴────────────────────────┘    │
│                                 │                                              │
│   ┌─────────────────────────────┴──────────────────────────────────────┐     │
│   │                        CACHING LAYER                                │     │
│   │  ┌──────────────────┐    ┌──────────────────────────────────────┐  │     │
│   │  │  Semantic Cache  │    │  Redis (Session, Rate Limiting)      │  │     │
│   │  │  (Qdrant/Milvus) │    │                                      │  │     │
│   │  └────────┬─────────┘    └───────────────────────────────────────┘  │     │
│   └───────────┴─────────────────────────────────────────────────────────┘     │
│               │                                                               │
│               │ Cache Miss                                                    │
│               ▼                                                               │
│   ┌───────────────────────────────────────────────────────────────────────┐  │
│   │                    INFERENCE LAYER (vLLM Cluster)                      │  │
│   │                                                                         │  │
│   │   ┌─────────────────────────────────────────────────────────────────┐  │  │
│   │   │                    vLLM Load Balancer                           │  │  │
│   │   │                           │                                      │  │  │
│   │   │    ┌──────────────────────┼──────────────────────┐              │  │  │
│   │   │    ▼                      ▼                      ▼              │  │  │
│   │   │ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐    │  │  │
│   │   │ │   vLLM Node 1   │ │   vLLM Node 2   │ │   vLLM Node 3   │    │  │  │
│   │   │ │   A100 (80GB)   │ │   A100 (80GB)   │ │   A100 (80GB)   │    │  │  │
│   │   │ │                 │ │                 │ │                 │    │  │  │
│   │   │ │  Llama 3 70B    │ │  Llama 3 70B    │ │  Llama 3 70B    │  │  │  │
│   │   │ │  4-bit Quant.   │ │  4-bit Quant.   │ │  4-bit Quant.   │  │  │  │
│   │   │ │  PagedAttention │ │  PagedAttention │ │  PagedAttention │  │  │  │
│   │   │ └─────────────────┘ └─────────────────┘ └─────────────────┘    │  │  │
│   │   │                                                                  │  │  │
│   │   │  Continuous Batching: 50-100 concurrent requests per node      │  │  │
│   │   └──────────────────────────────────────────────────────────────────┘  │  │
│   └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│   HOSTING OPTIONS:                                                            │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│   │  AWS SageMaker  │  │   RunPod/Modal  │  │   Self-hosted   │             │
│   │  (Managed)      │  │   (Serverless)  │  │   (EC2 p4d)     │             │
│   │  $$$            │  │  $$             │  │  $              │             │
│   │  Easy ops       │  │  Auto-scale     │  │  Full control   │             │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘             │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Key Technologies

#### 1. Quantization (4-bit/8-bit)
```python
# Load Llama 3 70B in 4-bit quantization
# Memory: 70B * 4bits = ~35GB (fits on single A100 80GB)

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat 4-bit
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True        # Double quantization
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-70B-Instruct",
    quantization_config=quantization_config,
    device_map="auto"
)

# Quality impact: ~1-3% degradation on benchmarks
# Memory savings: 4x (280GB → 70GB) or 8x with 4-bit
```

#### 2. vLLM with PagedAttention
```python
# vLLM achieves 10-24x higher throughput than HuggingFace
# PagedAttention: Efficient KV-cache memory management

# Server deployment
from vllm import LLM, SamplingParams

# Initialize vLLM engine
llm = LLM(
    model="meta-llama/Llama-3-70B-Instruct",
    tensor_parallel_size=2,        # Distribute across 2 GPUs
    quantization="awq",            # AWQ quantization
    max_model_len=8192,            # Max context length
    gpu_memory_utilization=0.9,    # Use 90% of GPU memory
    enforce_eager=False            # Use CUDA graphs
)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=512
)

# Generate (handles batching automatically)
outputs = llm.generate(["prompt1", "prompt2", "prompt3"], sampling_params)

# For production: Run as OpenAI-compatible API server
# python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Llama-3-70B-Instruct \
#     --tensor-parallel-size 2 \
#     --port 8000
```

#### 3. AWS SageMaker Deployment
```python
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel

def deploy_to_sagemaker():
    """
    Deploy Llama 3 on SageMaker with auto-scaling.
    """
    role = sagemaker.get_execution_role()
    
    # HuggingFace model configuration
    hub_config = {
        "HF_MODEL_ID": "meta-llama/Llama-3-70B-Instruct",
        "HF_TASK": "text-generation",
        "SM_NUM_GPUS": "4",  # Use 4 GPUs for tensor parallelism
        "MAX_INPUT_LENGTH": "4096",
        "MAX_TOTAL_TOKENS": "8192",
        "QUANTIZE": "bitsandbytes"
    }
    
    huggingface_model = HuggingFaceModel(
        image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.0.1-tgi1.0.3-gpu-py39-cu118-ubuntu20.04",
        env=hub_config,
        role=role
    )
    
    # Deploy with auto-scaling
    predictor = huggingface_model.deploy(
        initial_instance_count=2,
        instance_type="ml.p4d.24xlarge",  # 8x A100 40GB
        endpoint_name="chanakya-llama3-prod"
    )
    
    # Configure auto-scaling
    client = boto3.client("application-autoscaling")
    
    client.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=f"endpoint/chanakya-llama3-prod/variant/AllTraffic",
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=2,
        MaxCapacity=10
    )
    
    # Scale based on invocations per instance
    client.put_scaling_policy(
        PolicyName="Invocations-ScalingPolicy",
        ServiceNamespace="sagemaker",
        ResourceId=f"endpoint/chanakya-llama3-prod/variant/AllTraffic",
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": 100,  # 100 invocations per instance
            "PredefinedMetricSpecification": {
                "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
            }
        }
    )
    
    return predictor
```

### Cost Analysis

```
┌───────────────────────────────────────────────────────────────────────────┐
│                        COST COMPARISON (100K Users)                        │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ASSUMPTIONS:                                                              │
│  • 100K concurrent users, avg 10 requests/user/day                        │
│  • Each request: ~500 input tokens, ~200 output tokens                    │
│  • Monthly requests: 100K × 10 × 30 = 30M requests                        │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  OPTION 1: OpenAI GPT-4                                             │  │
│  │  ─────────────────────────                                          │  │
│  │  Input:  30M × 500 tokens × $0.03/1K = $450,000/month              │  │
│  │  Output: 30M × 200 tokens × $0.06/1K = $360,000/month              │  │
│  │  TOTAL: ~$810,000/month                                             │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  OPTION 2: Self-hosted Llama 3 70B (vLLM on AWS)                   │  │
│  │  ─────────────────────────────────────────────                      │  │
│  │  Infrastructure:                                                     │  │
│  │  • 4x p4d.24xlarge (8x A100): $12.50/hr × 4 × 720hr = $36,000     │  │
│  │  • Load balancer, networking: ~$2,000                               │  │
│  │  • Ops, monitoring: ~$5,000                                         │  │
│  │  TOTAL: ~$43,000/month                                              │  │
│  │                                                                      │  │
│  │  SAVINGS: 95% reduction ($767,000/month)                            │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  OPTION 3: Hybrid (Cache + Smaller Model + Fallback)               │  │
│  │  ─────────────────────────────────────────────────                  │  │
│  │  • 40% cache hits (semantic cache): $0                              │  │
│  │  • 50% simple requests (Llama 3 8B): $8,000 (2x g5.12xlarge)       │  │
│  │  • 10% complex (Llama 3 70B or GPT-4 fallback): $10,000            │  │
│  │  TOTAL: ~$18,000/month                                              │  │
│  │                                                                      │  │
│  │  SAVINGS: 98% reduction                                              │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
└───────────────────────────────────────────────────────────────────────────┘
```

### Hybrid Routing Implementation
```python
from enum import Enum
from dataclasses import dataclass

class ModelTier(Enum):
    CACHE = "cache"
    SMALL = "llama-3-8b"
    LARGE = "llama-3-70b"
    PREMIUM = "gpt-4"

@dataclass
class RoutingDecision:
    tier: ModelTier
    reason: str

class IntelligentRouter:
    """
    Route requests to appropriate model based on complexity.
    """
    
    def __init__(self, cache: SemanticCache, complexity_classifier):
        self.cache = cache
        self.classifier = complexity_classifier
    
    def route(self, query: str, context: str = "") -> RoutingDecision:
        # Layer 1: Check cache
        cached = self.cache.get(query)
        if cached:
            return RoutingDecision(ModelTier.CACHE, "Cache hit")
        
        # Layer 2: Classify complexity
        features = self._extract_features(query, context)
        complexity = self.classifier.predict(features)
        
        if complexity < 0.3:
            return RoutingDecision(ModelTier.SMALL, "Simple query")
        elif complexity < 0.7:
            return RoutingDecision(ModelTier.LARGE, "Moderate complexity")
        else:
            return RoutingDecision(ModelTier.PREMIUM, "High complexity")
    
    def _extract_features(self, query: str, context: str) -> dict:
        """Extract features for complexity classification."""
        return {
            "query_length": len(query.split()),
            "context_length": len(context.split()),
            "has_math": bool(re.search(r'[\d+\-*/=]', query)),
            "has_code": bool(re.search(r'```|def |class |function', query)),
            "question_type": self._classify_question_type(query),
            "requires_reasoning": "why" in query.lower() or "explain" in query.lower()
        }
    
    def _classify_question_type(self, query: str) -> str:
        """Simple question type classification."""
        query_lower = query.lower()
        if any(w in query_lower for w in ["what is", "define", "meaning"]):
            return "factual"
        elif any(w in query_lower for w in ["how to", "steps", "process"]):
            return "procedural"
        elif any(w in query_lower for w in ["why", "reason", "explain"]):
            return "analytical"
        elif any(w in query_lower for w in ["compare", "difference", "versus"]):
            return "comparative"
        return "general"
```

---

## Key Takeaways for the Interview (Domain 3)

| Concept | What to Say | What NOT to Say |
|---------|-------------|-----------------|
| **Latency vs Throughput** | "Streaming for interactive, batching for bulk. Measure TTFT, TPS, P99 latency." | "I use the API directly" |
| **Caching** | "Semantic cache with 0.95 similarity threshold, 30-50% cost savings on repeated queries" | "I cache exact matches" |
| **RAG vs Fine-tuning** | "RAG for dynamic knowledge + citations, fine-tuning for style/format. Often combine both." | "Fine-tuning is always better" |
| **Quantization** | "4-bit NF4 or AWQ, 1-3% quality loss, 4x memory reduction, fits 70B on single A100" | "I use full precision" |
| **vLLM** | "PagedAttention for KV-cache, continuous batching, 10-24x throughput vs HuggingFace" | "I use transformers.generate()" |
| **Cost Optimization** | "Hybrid routing: cache → small model → large model → premium fallback. 90%+ savings." | "We use GPT-4 for everything" |

---

# QUICK REFERENCE CHEAT SHEET

## Interview Response Framework

When asked "How would you design X at scale?", use this structure:

1. **Clarify Requirements** (30 seconds)
   - "What's the expected QPS/concurrency?"
   - "What's the latency SLA?"
   - "What's the accuracy requirement vs cost budget?"

2. **High-Level Architecture** (2 minutes)
   - Draw the main components
   - Show data flow
   - Identify bottlenecks

3. **Deep Dive on Bottleneck** (5 minutes)
   - Pick the hardest problem
   - Discuss 2-3 approaches with trade-offs
   - State your recommendation with rationale

4. **Operational Considerations** (2 minutes)
   - Monitoring (latency, accuracy, cost metrics)
   - Failure modes and recovery
   - Scaling strategy

## Key Numbers to Remember

| Metric | Value | Context |
|--------|-------|---------|
| **LLM API Latency** | 500ms-2s | GPT-4 TTFT (Time To First Token) |
| **Vector Search** | <10ms | Per query with proper indexing |
| **Cross-Encoder Reranking** | 50-100ms | Per 10 candidates |
| **Embedding Generation** | 5-10ms | Per document chunk |
| **Semantic Cache Hit Rate** | 30-50% | Production systems with repeated queries |
| **Llama 3 70B Memory** | 140GB FP16, 70GB INT8, 35GB INT4 | GPU VRAM needed |
| **vLLM Throughput** | ~500 tokens/s | Per A100 with 70B model |
| **Cost Savings (Self-hosted)** | 90-95% | vs OpenAI at scale |

---

*Study Guide Generated for Santosh Kumar - Lead Backend Engineer*
*Target: System Design for AI Interviews at Tier-1 Tech Companies*
