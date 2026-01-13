"""
Vector Worker Agent - Specialized for semantic document search.

This agent handles:
- Semantic search over document collections
- Hybrid keyword-semantic retrieval
- Document summarization
- Relevance scoring and filtering
"""

import logging
from typing import Dict, Any, Optional
import json

from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, ToolMessage
)
from langchain_core.tools import Tool
from langchain import agents, hub
from langchain_openai import OpenAIEmbeddings

from src.graph.state import AgentState
from src.utils.llm_factory import LLMFactory
from src.tools.vector_store_tools import PineconeConnector

logger = logging.getLogger(__name__)


def vector_worker_node(state: AgentState) -> Dict[str, Any]:
    """
    Vector Worker Node - Performs semantic search over documents.
    
    This specialized agent handles:
    1. Query embedding
    2. Semantic similarity search
    3. Document chunk retrieval
    4. Result summarization
    
    Args:
        state: Current AgentState
    
    Returns:
        Updated state with search results
    """
    
    # Initialize vector store connector
    try:
        vector_connector = PineconeConnector(
            api_key=__get_env("PINECONE_API_KEY"),
            environment=__get_env("PINECONE_ENVIRONMENT", "us-west-2-aws"),
            index_name=__get_env("PINECONE_INDEX", "ent-qa"),
            top_k=int(__get_env("PINECONE_TOP_K", "5")),
            namespace=__get_env("PINECONE_NAMESPACE", "default"),
        )
        vector_connector.connect()
    except Exception as e:
        logger.error(f"Failed to initialize vector connector: {str(e)}")
        return {
            "next_step": "FINISH",
            "error_message": f"Vector store connection failed: {str(e)}",
            "final_answer": "I could not connect to the document store. Please check your configuration.",
            "messages": [
                AIMessage(
                    content=f"[VECTOR_WORKER_ERROR] Connection failed: {str(e)}",
                    name="vector_agent"
                )
            ]
        }
    
    # Initialize embeddings and LLM
    try:
        embeddings = OpenAIEmbeddings(
            api_key=__get_env("OPENAI_API_KEY"),
            model="text-embedding-3-small"
        )
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {str(e)}")
        return {
            "error_message": f"Embedding initialization failed: {str(e)}",
            "final_answer": "I encountered an embedding service error.",
            "messages": [
                AIMessage(
                    content=f"[VECTOR_WORKER_ERROR] Embeddings failed: {str(e)}",
                    name="vector_agent"
                )
            ]
        }
    
    llm = LLMFactory.create_worker_llm("vector")
    
    # Define tools for vector search
    def semantic_search(query: str, top_k: Optional[int] = None) -> str:
        """Perform semantic search over documents."""
        try:
            # Embed the query
            query_embedding = embeddings.embed_query(query)
            
            # Search
            result = vector_connector.similarity_search(
                query_embedding=query_embedding,
                query_text=query,
                top_k=top_k,
                include_metadata=True
            )
            
            # Format results
            if result.total_matches == 0:
                return f"No relevant documents found for: '{query}'"
            
            formatted_results = []
            for match in result.matches[:5]:  # Top 5
                metadata = match.get("metadata", {})
                score = match.get("score", 0.0)
                text = metadata.get("text", "")[:500]  # First 500 chars
                
                formatted_results.append({
                    "relevance_score": score,
                    "chunk_id": match.get("id"),
                    "text": text,
                    "source": metadata.get("source", "unknown")
                })
            
            return json.dumps({
                "query": query,
                "total_matches": result.total_matches,
                "top_results": formatted_results
            })
        
        except Exception as e:
            return f"Search Error: {str(e)}"
    
    def keyword_search(keyword: str) -> str:
        """Search with specific keywords."""
        try:
            keyword_embedding = embeddings.embed_query(keyword)
            
            # With keyword filtering
            result = vector_connector.similarity_search(
                query_embedding=keyword_embedding,
                query_text=keyword,
                filters={"keyword": keyword},
                top_k=5,
                include_metadata=True
            )
            
            if result.total_matches == 0:
                return f"No documents with keyword '{keyword}' found."
            
            chunks = []
            for match in result.matches[:5]:
                metadata = match.get("metadata", {})
                chunks.append(metadata.get("text", "")[:500])
            
            return "\n---\n".join(chunks)
        
        except Exception as e:
            return f"Keyword Search Error: {str(e)}"
    
    # Create tools
    tools = [
        Tool(
            name="semantic_search",
            func=semantic_search,
            description="Perform semantic similarity search over document collection. Input a natural language query.",
        ),
        Tool(
            name="keyword_search",
            func=keyword_search,
            description="Search for documents containing specific keywords.",
        ),
    ]
    
    # Get task from latest message
    last_message = state["messages"][-1]
    user_task = last_message.content
    
    # System prompt
    vector_system_prompt = """You are a Document Retrieval Specialist.

Your approach:
1. Analyze the user's information need
2. Reformulate as a semantic search query
3. Search the document collection
4. Synthesize relevant findings into a clear answer
5. Cite sources where appropriate

The user's request:
{user_task}

Begin by searching for relevant documents, then summarize your findings.
"""
    
    # Use ReAct agent
    prompt = hub.pull("hwchase17/react")
    
    agent = agents.create_react_agent(
        llm,
        tools,
        prompt,
    )
    
    agent_executor = agents.AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True
    )
    
    try:
        # Execute the agent
        result = agent_executor.invoke({
            "input": user_task,
        })
        
        vector_result = result.get("output", "No results found")
        
        return {
            "messages": [
                AIMessage(
                    content=f"[VECTOR_WORKER] Search Result:\n{vector_result}",
                    name="vector_agent"
                )
            ]
        }
    
    except Exception as e:
        logger.error(f"Vector agent execution failed: {str(e)}")
        return {
            "error_message": str(e),
            "messages": [
                AIMessage(
                    content=f"[VECTOR_WORKER_ERROR] {str(e)}",
                    name="vector_agent"
                )
            ]
        }
    
    finally:
        vector_connector.disconnect()


def __get_env(key: str, default: str = "") -> str:
    """Helper to get environment variables."""
    import os
    return os.getenv(key, default)
