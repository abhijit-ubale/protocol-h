"""
Workflow Graph - The Main Orchestration Engine.

This module assembles the hierarchical agentic RAG system using LangGraph's StateGraph.
It defines:
- All nodes (Supervisor, Workers, Synthesizer)
- State transitions and routing logic
- Conditional edges for decision-making
- The complete DAG topology
"""

import logging
from typing import Literal

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from src.graph.state import AgentState
from src.graph.supervisor import supervisor_node, reflective_retry_node
from src.agents.sql_agent import sql_worker_node
from src.agents.vector_agent import vector_worker_node

logger = logging.getLogger(__name__)


class WorkflowBuilder:
    """
    Builder class for constructing the hierarchical agentic RAG workflow.
    
    Uses the Builder pattern to allow flexible configuration and testing
    of the workflow graph.
    """
    
    def __init__(self):
        """Initialize the workflow builder."""
        self.graph = None
        self._compiled_app = None
    
    def build_workflow(self) -> StateGraph:
        """
        Build and compile the complete workflow graph.
        
        This method constructs the StateGraph with all nodes and edges,
        returning a ready-to-execute orchestration engine.
        
        Returns:
            Compiled LangGraph application ready to invoke
        """
        
        # Initialize the StateGraph
        workflow = StateGraph(AgentState)
        
        # Add all nodes
        # ============================================================
        
        # 1. Supervisor Node - The orchestrator
        workflow.add_node("supervisor", supervisor_node)
        
        # 2. Worker Nodes
        workflow.add_node("sql_agent", sql_worker_node)
        workflow.add_node("vector_agent", vector_worker_node)
        
        # 3. Reflective Retry Node - Error recovery
        workflow.add_node("reflective_retry", reflective_retry_node)
        
        # 4. Synthesizer Node - Final answer composition
        workflow.add_node("synthesizer", synthesizer_node)
        
        # Define edges and routing logic
        # ============================================================
        
        # Entry point: Start with the supervisor
        workflow.set_entry_point("supervisor")
        
        # Supervisor routes to workers or finish
        def supervisor_router(state: AgentState) -> str:
            """Route based on supervisor's next_step decision."""
            next_step = state.get("next_step", "FINISH")
            
            if next_step in ["sql_agent", "vector_agent"]:
                return next_step
            elif next_step == "FINISH":
                return "synthesizer"
            else:
                logger.warning(f"Unknown next_step: {next_step}, routing to synthesizer")
                return "synthesizer"
        
        workflow.add_conditional_edges(
            "supervisor",
            supervisor_router,
            {
                "sql_agent": "sql_agent",
                "vector_agent": "vector_agent",
                "synthesizer": "synthesizer",
            }
        )
        
        # SQL Agent error handling
        def sql_agent_router(state: AgentState) -> str:
            """Route SQL agent output."""
            if state.get("error_message"):
                return "reflective_retry"
            else:
                return "supervisor"
        
        workflow.add_conditional_edges(
            "sql_agent",
            sql_agent_router,
            {
                "reflective_retry": "reflective_retry",
                "supervisor": "supervisor",
            }
        )
        
        # Vector Agent error handling
        def vector_agent_router(state: AgentState) -> str:
            """Route vector agent output."""
            if state.get("error_message"):
                return "reflective_retry"
            else:
                return "supervisor"
        
        workflow.add_conditional_edges(
            "vector_agent",
            vector_agent_router,
            {
                "reflective_retry": "reflective_retry",
                "supervisor": "supervisor",
            }
        )
        
        # Retry logic
        def retry_router(state: AgentState) -> str:
            """Route retry decisions."""
            next_step = state.get("next_step")
            
            if next_step == "FINISH":
                return "synthesizer"
            elif next_step in ["sql_agent", "vector_agent"]:
                return next_step
            else:
                return "supervisor"
        
        workflow.add_conditional_edges(
            "reflective_retry",
            retry_router,
            {
                "sql_agent": "sql_agent",
                "vector_agent": "vector_agent",
                "supervisor": "supervisor",
                "synthesizer": "synthesizer",
            }
        )
        
        # Synthesizer ends the workflow
        workflow.add_edge("synthesizer", END)
        
        # Compile the graph
        self._compiled_app = workflow.compile()
        self.graph = workflow
        
        logger.info("Workflow graph compiled successfully")
        return self._compiled_app
    
    def get_compiled_app(self):
        """Get the compiled LangGraph application."""
        if not self._compiled_app:
            self.build_workflow()
        return self._compiled_app


def synthesizer_node(state: AgentState) -> dict:
    """
    Synthesizer Node - Composes the final answer.
    
    This node:
    1. Collects information from worker outputs
    2. Synthesizes a coherent answer
    3. Cites sources
    4. Prepares the response for the user
    
    Args:
        state: Complete AgentState with all intermediate results
    
    Returns:
        Final state with synthesized answer
    """
    
    from src.utils.llm_factory import LLMFactory
    
    # Collect all worker outputs
    messages_text = []
    for msg in state["messages"][-10:]:  # Last 10 messages
        if hasattr(msg, "name"):
            messages_text.append(f"[{msg.name}] {msg.content}")
        else:
            messages_text.append(msg.content)
    
    context = "\n".join(messages_text)
    
    # If we already have a final answer, return it
    if state.get("final_answer"):
        return {"final_answer": state["final_answer"]}
    
    # Synthesize new answer
    llm = LLMFactory.create_supervisor_llm()
    
    synthesis_prompt = f"""You are synthesizing a final answer based on information gathered from multiple sources.

Information gathered:
{context}

Original User Query:
{state['messages'][0].content if state['messages'] else 'N/A'}

Your task:
1. Synthesize a coherent, accurate answer
2. Cite which sources (SQL database, documents, etc.) provided each piece of information
3. Note any limitations or uncertainties
4. Format the answer clearly for the user

Provide your final answer:
"""
    
    try:
        response = llm.invoke([
            AIMessage(content=synthesis_prompt)
        ])
        
        final_answer = response.content
        
        return {
            "final_answer": final_answer,
            "messages": [
                AIMessage(
                    content=f"[SYNTHESIZER] Final Answer:\n{final_answer}",
                    name="synthesizer"
                )
            ]
        }
    
    except Exception as e:
        logger.error(f"Synthesis failed: {str(e)}")
        return {
            "final_answer": f"I gathered information but encountered an error during synthesis: {str(e)}. Here's what was collected: {context}"
        }


def create_orchestrator() -> WorkflowBuilder:
    """
    Factory function to create a new orchestrator instance.
    
    Returns:
        WorkflowBuilder with compiled workflow
    """
    builder = WorkflowBuilder()
    builder.build_workflow()
    return builder
