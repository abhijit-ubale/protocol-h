# File Index: Hierarchical Agentic RAG Repository

## Directory Structure

```
protocol-h/
├── config/                          # Configuration files
│   ├── agents.yaml                 # Agent prompts and behaviors
│   └── connections.yaml            # Database connection templates
│
├── src/                            # Source code (main package)
│   ├── __init__.py                # Package initialization
│   │
│   ├── graph/                      # Orchestration engine (LangGraph)
│   │   ├── __init__.py
│   │   ├── state.py               # AgentState definition (TypedDict)
│   │   ├── supervisor.py          # Supervisor node and routing logic
│   │   └── workflow.py            # StateGraph construction
│   │
│   ├── agents/                     # Worker agent implementations
│   │   ├── __init__.py
│   │   ├── sql_agent.py           # SQL Worker for database queries
│   │   └── vector_agent.py        # Vector Worker for semantic search
│   │
│   ├── tools/                      # Data connectors (Adapter Pattern)
│   │   ├── __init__.py
│   │   ├── base_connector.py      # Abstract BaseConnector interface
│   │   ├── snowflake_tools.py     # Snowflake connector (production)
│   │   └── vector_store_tools.py  # Pinecone connector
│   │
│   └── utils/                      # Utility modules
│       ├── __init__.py
│       └── llm_factory.py         # LLM initialization factory
│
├── notebooks/
│   └── demo_ent_qa.ipynb          # Interactive demo & Ent-QA benchmarks
│
├── root files
│   ├── main.py                    # CLI entry point
│   ├── requirements.txt           # Python dependencies
│   ├── .env.example               # Environment variable template
│   ├── Dockerfile                 # Container definition
│   ├── LICENSE                    # MIT License
│   ├── README.md                  # User documentation
│   ├── BUILD_SUMMARY.md           # This build summary
│   ├── FILE_INDEX.md              # This file
│   └── notes.txt                  # Original research paper
```

## Core Files Explained

### State Management
**Location**: `src/graph/state.py`
**Purpose**: Define the unified state representation for the orchestration workflow
**Key Classes**:
- `AgentState`: TypedDict with fields:
  - `messages`: Conversation history (LangChain BaseMessages)
  - `next_step`: Routing decision (str: "sql_agent" | "vector_agent" | "FINISH")
  - `final_answer`: Synthesized output (Optional[str])
  - `query_type`: Query classification (Optional[str])
  - `retry_count`: Error retry counter (int)
  - `error_message`: Latest error (Optional[str])
- `WorkerResult`: Standard inter-agent result structure

### Orchestration Engine
**Location**: `src/graph/supervisor.py`
**Purpose**: Implement the Supervisor orchestrator and reflective retry mechanism
**Key Functions**:
- `supervisor_node(state)`: Route queries to appropriate workers
  - Uses LLM to decompose queries
  - Structured output parsing for deterministic routing
  - Returns routing decision and instructions
- `reflective_retry_node(state)`: Autonomous error recovery
  - Analyzes worker failures
  - Formulates corrections
  - Routes recovery action
- `_prepare_messages_summary()`: Compress conversation for context

### Workflow Assembly
**Location**: `src/graph/workflow.py`
**Purpose**: Assemble the complete StateGraph with all nodes and conditional edges
**Key Classes**:
- `WorkflowBuilder`: Builder for workflow construction
  - `build_workflow()`: Assemble StateGraph
  - `get_compiled_app()`: Return compiled LangGraph application
**Key Functions**:
- `synthesizer_node(state)`: Combine worker outputs into final answer
- `create_orchestrator()`: Factory to create new orchestrator instance

**Graph Structure**:
```
[Supervisor] 
  ├─→ [SQL Worker] ─→ [Reflective Retry] ─→ [Supervisor]
  ├─→ [Vector Worker] ─→ [Reflective Retry] ─→ [Supervisor]
  └─→ [FINISH] ─→ [Synthesizer] ─→ [END]
```

### SQL Worker Agent
**Location**: `src/agents/sql_agent.py`
**Purpose**: Execute database queries with schema awareness
**Key Function**: `sql_worker_node(state)`
**Tools**:
- `schema_introspector`: Read table definitions
- `list_tables`: Enumerate available tables
- `query_executor`: Execute read-only SQL
**Agent Type**: ReAct (Reasoning + Acting)
**Max Iterations**: 5
**Temperature**: 0.0 (deterministic)

### Vector Worker Agent
**Location**: `src/agents/vector_agent.py`
**Purpose**: Perform semantic search over document collections
**Key Function**: `vector_worker_node(state)`
**Tools**:
- `semantic_search`: Query embedding + similarity search
- `keyword_search`: Keyword-based retrieval
**Agent Type**: ReAct
**Max Iterations**: 5
**Temperature**: 0.2 (slightly creative for summarization)

### Database Connectors

#### Base Connector (Adapter Pattern)
**Location**: `src/tools/base_connector.py`
**Purpose**: Define cloud-agnostic database interface
**Key Classes**:
- `BaseConnector` (ABC): Abstract interface with methods:
  - `connect()`: Establish connection
  - `disconnect()`: Close connection
  - `list_tables()`: Enumerate tables
  - `get_table_schema(table_name)`: Get column definitions
  - `execute_query(sql)`: Run read-only queries
  - `test_connection()`: Validate connectivity
- `TableSchema`: Schema information dataclass
- `QueryResult`: Query result dataclass
- `ConnectorFactory`: Factory pattern for connector creation

#### Snowflake Connector
**Location**: `src/tools/snowflake_tools.py`
**Purpose**: Production Snowflake implementation
**Key Class**: `SnowflakeConnector(BaseConnector)`
**Features**:
- SQLAlchemy for connection management
- Snowflake-specific optimizations
- Read-only enforcement
- Query result formatting
- Error handling with detailed messages
**Auto-Registration**: Registers with ConnectorFactory on import

#### Vector Store Connector
**Location**: `src/tools/vector_store_tools.py`
**Purpose**: Pinecone integration for semantic search
**Key Class**: `PineconeConnector`
**Features**:
- Query embedding via OpenAI
- Hybrid semantic-keyword search
- Metadata filtering
- Batch document retrieval
- Relevance scoring
**Result Classes**: `VectorSearchResult`

### LLM Factory
**Location**: `src/utils/llm_factory.py`
**Purpose**: Unified LLM initialization across providers
**Key Class**: `LLMFactory`
**Static Methods**:
- `create_llm(provider, model, temperature, ...)`: Create LLM instance
  - Supports: "openai", "azure"
  - Model selection: "gpt-4o", "gpt-3.5-turbo", etc.
- `create_supervisor_llm()`: LLM for routing (T=0.1)
- `create_worker_llm(worker_type)`: LLM for specific worker (T=0.0 or 0.2)
**Env Vars**: OPENAI_API_KEY, AZURE_OPENAI_*

## Configuration Files

### Agent Configuration
**Location**: `config/agents.yaml`
**Purpose**: Define system prompts and behavior for each agent
**Sections**:
- `supervisor`: Routing prompts, model, temperature
- `sql_agent`: SQL generation prompts, model, temperature
- `vector_agent`: Search prompts, model, temperature
- `reflective_retry`: Error handling configuration

### Connection Configuration
**Location**: `config/connections.yaml`
**Purpose**: Database connection parameters (all environment-based)
**Sections**:
- `snowflake`: Account, user, warehouse, database, schema
- `redshift`: Alternative RDBMS configuration
- `bigquery`: Google Cloud alternative
- `pinecone`: Vector store API and configuration
- `redis`: State persistence (optional)
- `llm`: LLM provider and model settings

## Entry Points

### CLI Entry Point
**Location**: `main.py`
**Purpose**: Command-line interface for the orchestrator
**Usage**: `python main.py "your query" [--verbose] [--output-json]`
**Functions**:
- `main()`: Parse args and execute orchestration
- Supports: single query, JSON output, verbose logging

### Jupyter Demo
**Location**: `notebooks/demo_ent_qa.ipynb`
**Purpose**: Interactive demonstration and benchmarking
**Sections**:
1. Setup & imports
2. Orchestrator initialization
3. Example queries (single-hop, multi-hop, cross-modal)
4. Ent-QA benchmark dataset
5. Performance comparison table
6. Hallucination analysis
7. Cost-per-correct-answer analysis
8. Architecture deep dive
9. Cloud deployment options
10. Conclusion

## Supporting Files

### Requirements
**Location**: `requirements.txt`
**Key Dependencies**:
- `langgraph==0.2.46`: Orchestration engine
- `langchain==0.3.9`: LLM framework
- `langchain-core==0.3.19`: Core abstractions
- `langchain-openai==0.2.5`: OpenAI integration
- `snowflake-connector-python==3.10.1`: Snowflake
- `pinecone-client==4.1.1`: Pinecone
- `pydantic==2.7.1`: Data validation

### Environment Template
**Location**: `.env.example`
**Purpose**: Template for setting up environment variables
**Variables**:
- OPENAI_API_KEY, OPENAI_MODEL
- SNOWFLAKE_* (account, user, password, warehouse, etc.)
- PINECONE_* (api_key, index, environment)
- REDIS_* (optional)

### Docker Definition
**Location**: `Dockerfile`
**Base Image**: python:3.11-slim
**Components**:
- System dependencies (build-essential, curl)
- Python package installation
- Application code copying
- Health check endpoint
- Default command (Uvicorn server or Jupyter)

### License
**Location**: `LICENSE`
**Type**: MIT
**Allows**: Commercial use, modification, distribution
**Requires**: License and copyright notice

## Documentation Files

### User README
**Location**: `README.md`
**Audience**: End users
**Contents**:
- Overview of Protocol-H
- Quick start guide
- Architecture diagram
- Installation instructions
- Configuration guide
- Usage examples
- Performance benchmarks
- Troubleshooting
- Deployment options

### Build Summary
**Location**: `BUILD_SUMMARY.md`
**Audience**: Developers/Implementers
**Contents**:
- Repository structure
- Files created (with descriptions)
- Key components implemented
- Technology stack
- Performance benchmarks
- Design highlights
- Configuration options
- Quick start guide
- Production readiness checklist
- Next steps

### Research Paper
**Location**: `notes.txt`
**Audience**: Academic/Technical audience
**Contents**:
- Abstract on hierarchical agentic RAG
- Design goals and requirements
- Framework architecture details
- Implementation notes
- Experimental setup (Ent-QA)
- Results and analysis
- Conclusion and future work

## File Statistics

| Category | Count | Examples |
|----------|-------|----------|
| Python modules | 15 | state.py, supervisor.py, sql_agent.py |
| Configuration | 2 | agents.yaml, connections.yaml |
| Documentation | 5 | README.md, BUILD_SUMMARY.md, etc. |
| Notebooks | 1 | demo_ent_qa.ipynb |
| Infrastructure | 2 | Dockerfile, requirements.txt |
| Support | 2 | .env.example, LICENSE |
| **Total** | **27** | — |

## Code Organization Principles

1. **Separation of Concerns**
   - `graph/`: Orchestration logic only
   - `agents/`: Worker implementations
   - `tools/`: Data access (Adapter Pattern)
   - `utils/`: Shared utilities

2. **Adapter Pattern**
   - `base_connector.py`: Abstract interface
   - `snowflake_tools.py`, `vector_store_tools.py`: Implementations
   - Easy to add new connectors without modifying agents

3. **Factory Pattern**
   - `LLMFactory`: Create LLMs by provider
   - `ConnectorFactory`: Create connectors by type

4. **Type Safety**
   - `AgentState`: TypedDict for state shape
   - `WorkerResult`: Standard result format
   - Pydantic models for structured outputs

5. **Error Handling**
   - Reflective Retry Mechanism
   - Logging at each step
   - Graceful degradation

## Development Workflow

1. **Modify Agent Behavior**: Edit `config/agents.yaml` prompts
2. **Add New Worker**: Create `src/agents/new_worker.py`
3. **Add New Database**: Create `src/tools/new_db_tools.py`
4. **Update Workflow**: Modify `src/graph/workflow.py` nodes/edges
5. **Test**: Use `notebooks/demo_ent_qa.ipynb`
6. **Deploy**: Use `Dockerfile` or `main.py` entry point

---

**Last Updated**: January 3, 2026
**Repository**: protocol-h
**Status**: Production-Ready
