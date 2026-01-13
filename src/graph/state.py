"""
State management for the hierarchical agentic RAG system.

This module defines the AgentState TypedDict that represents the complete
state of the orchestration workflow. It persists across all nodes in the graph.
"""

from typing import TypedDict, Optional, List, Annotated
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    """
    Unified state dictionary for the hierarchical agentic RAG system.
    
    This state is passed between nodes and updated as agents execute.
    It maintains the conversation history, tracks the next worker to execute,
    and accumulates intermediate results.
    """
    
    # Conversation history: All messages exchanged during the workflow
    messages: Annotated[List[BaseMessage], operator.add]
    
    # Router decision: Which agent should execute next
    # Valid values: "sql_agent", "vector_agent", "FINISH"
    next_step: str
    
    # Final synthesized answer to return to the user
    final_answer: Optional[str]
    
    # Metadata tracking
    query_type: Optional[str]  # "single_hop", "multi_hop", "cross_modal"
    retry_count: int  # Number of times the current task has been retried
    error_message: Optional[str]  # Latest error from a worker (if any)


class WorkerResult(TypedDict):
    """
    Standard result structure returned by worker agents.
    Allows the Supervisor to consistently interpret outputs.
    """
    
    success: bool
    data: Optional[str]
    error: Optional[str]
    worker_name: str
    timestamp: Optional[str]
