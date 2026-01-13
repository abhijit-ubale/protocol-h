"""
Supervisor Agent - The Meta-Cognitive Orchestrator.

The Supervisor is the "boss" of the hierarchical agentic system.
It receives user queries, decomposes them into sub-tasks, and delegates
to specialized workers. It also manages the "Reflective Retry" mechanism
for self-correction.
"""

import logging
from typing import Dict, Any, Literal
from datetime import datetime

from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
)
from langchain_core.pydantic_v1 import BaseModel, Field

from src.graph.state import AgentState
from src.utils.llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class SupervisorDecision(BaseModel):
    """Structured output from supervisor's decision."""
    
    next_worker: Literal["sql_agent", "vector_agent", "FINISH"] = Field(
        ...,
        description="Which worker should handle the next step"
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of the routing decision"
    )
    task_description: str = Field(
        default="",
        description="Specific instruction for the next worker"
    )


def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """
    Supervisor node that routes queries to appropriate workers.
    
    This is the "meta-cognitive" layer of Protocol-H. The Supervisor:
    1. Analyzes the conversation history
    2. Determines which worker(s) should execute next
    3. Formulates clear instructions for delegation
    4. Manages the overall reasoning flow
    
    Args:
        state: Current AgentState containing conversation history
    
    Returns:
        Dictionary update containing next_step and potentially other state updates
    """
    
    # Initialize LLM for supervisor
    llm = LLMFactory.create_supervisor_llm()
    
    # Prepare conversation context
    messages_summary = _prepare_messages_summary(state["messages"])
    
    # Construct supervisor prompt
    supervisor_prompt = f"""You are the Supervisor Agent in a hierarchical multi-agent RAG system.

Your role is to orchestrate a team of specialized workers to answer complex business questions.

Available Workers:
1. sql_agent: Queries structured databases (Snowflake, Redshift, etc.) for quantitative data
   - Use this for: SQL queries, data lookups, aggregations, counts, joins
   - This worker has schema awareness and can validate column names

2. vector_agent: Searches document collections for qualitative insights
   - Use this for: Document retrieval, PDF searches, text analysis, qualitative data
   - This worker performs semantic and keyword search

3. FINISH: When you have enough information to synthesize the final answer
   - Use this when all required information has been gathered
   - The final answer will be synthesized and returned to the user

Current Conversation:
{messages_summary}

Decision Rules:
- If the query requires both SQL and document data, start with SQL (to get facts), then Vector (to get context)
- If a worker previously failed (error in state), try routing to a different worker or retry with modified instructions
- Only choose FINISH when you have sufficient information to answer the original user question
- Be explicit about what information you need from each worker

Your Decision (output ONLY valid JSON):
"""
    
    # Bind the supervisor LLM to structured output
    structured_llm = llm.with_structured_output(SupervisorDecision)
    
    try:
        # Get supervisor decision
        decision = structured_llm.invoke([
            SystemMessage(content=supervisor_prompt)
        ])
        
        logger.info(f"Supervisor decision: {decision.next_worker} - {decision.reasoning}")
        
        # Update state with supervisor decision
        update_dict = {
            "next_step": decision.next_worker,
            "messages": [
                AIMessage(
                    content=f"[SUPERVISOR] Routing to {decision.next_worker}: {decision.reasoning}",
                    name="supervisor"
                )
            ]
        }
        
        return update_dict
    
    except Exception as e:
        logger.error(f"Supervisor decision failed: {str(e)}")
        # Fallback: return error state
        return {
            "next_step": "FINISH",
            "error_message": f"Supervisor error: {str(e)}",
            "final_answer": "I encountered an error while processing your query. Please try again.",
            "messages": [
                AIMessage(
                    content=f"[SUPERVISOR_ERROR] {str(e)}",
                    name="supervisor"
                )
            ]
        }


def reflective_retry_node(state: AgentState) -> Dict[str, Any]:
    """
    Reflective Retry Mechanism - Closed-loop error handling.
    
    When a worker returns an error, this node:
    1. Analyzes the error
    2. Formulates a correction strategy
    3. Sends the task back to the worker (or a different worker)
    
    This is the "self-healing" aspect of Protocol-H.
    
    Args:
        state: Current AgentState with error_message set
    
    Returns:
        Updated state with retry instruction or failure signal
    """
    
    if not state.get("error_message") or state["retry_count"] >= 3:
        # Max retries reached or no error
        return {
            "next_step": "FINISH",
            "final_answer": "I was unable to retrieve the required information after multiple attempts."
        }
    
    llm = LLMFactory.create_supervisor_llm()
    
    error_analysis_prompt = f"""You are analyzing an error from a worker agent.

Worker Error:
{state['error_message']}

Last Message from User:
{state['messages'][-2].content if len(state['messages']) > 1 else 'N/A'}

Options:
1. Retry the same task with a different approach
2. Route to a different worker
3. Abort and inform user

Suggest the best corrective action (be brief):
"""
    
    try:
        analysis = llm.invoke([
            SystemMessage(content=error_analysis_prompt)
        ])
        
        logger.info(f"Reflective retry analysis: {analysis.content}")
        
        # Route back to supervisor for re-planning
        return {
            "next_step": "sql_agent",  # Or "vector_agent" depending on analysis
            "retry_count": state["retry_count"] + 1,
            "error_message": None,
            "messages": [
                AIMessage(
                    content=f"[RETRY_MECHANISM] Attempting recovery: {analysis.content}",
                    name="reflective_retry"
                )
            ]
        }
    
    except Exception as e:
        logger.error(f"Reflective retry analysis failed: {str(e)}")
        return {
            "next_step": "FINISH",
            "final_answer": "An error occurred and could not be recovered automatically."
        }


def _prepare_messages_summary(messages: list[BaseMessage]) -> str:
    """
    Prepare a concise summary of conversation messages for the supervisor.
    
    Args:
        messages: List of messages in conversation
    
    Returns:
        Formatted string summary
    """
    if not messages:
        return "No messages yet."
    
    summary_lines = []
    for i, msg in enumerate(messages[-10:]):  # Last 10 messages
        if isinstance(msg, HumanMessage):
            summary_lines.append(f"User: {msg.content[:200]}")
        elif isinstance(msg, AIMessage):
            name = getattr(msg, "name", "Assistant")
            summary_lines.append(f"{name}: {msg.content[:200]}")
        elif isinstance(msg, ToolMessage):
            name = getattr(msg, "name", "Tool")
            summary_lines.append(f"{name} Result: {msg.content[:200]}")
    
    return "\n".join(summary_lines)
