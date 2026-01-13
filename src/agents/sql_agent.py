"""
SQL Worker Agent - Specialized for database queries.

This agent handles all SQL-based data retrieval with:
- Schema introspection
- Query validation
- Error handling and retry logic
- Dialect-specific optimizations
"""

import logging
from typing import Dict, Any, Optional
import json

from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, ToolMessage
)
from langchain_core.tools import Tool
from langchain import agents, hub
from langchain_core.pydantic_v1 import BaseModel

from src.graph.state import AgentState
from src.utils.llm_factory import LLMFactory
from src.tools.base_connector import ConnectorFactory

logger = logging.getLogger(__name__)


class SchemaInfo(BaseModel):
    """Response from schema introspection."""
    table_name: str
    columns: list
    row_count: Optional[int]
    description: Optional[str]


class QueryExecution(BaseModel):
    """Response from query execution."""
    success: bool
    data: Optional[list] = None
    error: Optional[str] = None
    row_count: int = 0


def sql_worker_node(state: AgentState) -> Dict[str, Any]:
    """
    SQL Worker Node - Executes database queries.
    
    This specialized agent handles:
    1. Schema introspection
    2. SQL query generation
    3. Query execution with error handling
    4. Result formatting
    
    Args:
        state: Current AgentState
    
    Returns:
        Updated state with query results
    """
    
    # Initialize database connector
    try:
        db_connector = ConnectorFactory.create(
            "snowflake",
            account=__get_env("SNOWFLAKE_ACCOUNT"),
            user=__get_env("SNOWFLAKE_USER"),
            password=__get_env("SNOWFLAKE_PASSWORD"),
            warehouse=__get_env("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
            database=__get_env("SNOWFLAKE_DATABASE", "DEV_DB"),
            schema=__get_env("SNOWFLAKE_SCHEMA", "PUBLIC"),
        )
        db_connector.connect()
    except Exception as e:
        logger.error(f"Failed to initialize database connector: {str(e)}")
        return {
            "next_step": "FINISH",
            "error_message": f"Database connection failed: {str(e)}",
            "final_answer": "I could not connect to the database. Please check your configuration.",
            "messages": [
                AIMessage(
                    content=f"[SQL_WORKER_ERROR] Connection failed: {str(e)}",
                    name="sql_agent"
                )
            ]
        }
    
    # Initialize LLM for this worker
    llm = LLMFactory.create_worker_llm("sql")
    
    # Define tools for the SQL agent
    def schema_introspector(table_name: str) -> str:
        """Get schema information for a table."""
        try:
            schema = db_connector.get_table_schema(table_name)
            return json.dumps({
                "table_name": schema.table_name,
                "columns": schema.columns,
                "row_count": schema.row_count
            })
        except Exception as e:
            return f"Error: Could not find table '{table_name}'. {str(e)}"
    
    def list_available_tables() -> str:
        """List all available tables."""
        try:
            tables = db_connector.list_tables()
            return f"Available tables: {', '.join(tables)}"
        except Exception as e:
            return f"Error listing tables: {str(e)}"
    
    def query_executor(sql_query: str) -> str:
        """Execute a SQL query and return results."""
        try:
            result = db_connector.execute_query(sql_query)
            
            if not result.success:
                return f"Query Error: {result.error}"
            
            # Format results
            if result.row_count == 0:
                return "Query executed successfully but returned no rows."
            
            # Return first 10 rows as JSON
            data_sample = result.data[:10] if result.data else []
            return json.dumps({
                "rows_returned": result.row_count,
                "sample_data": data_sample,
                "execution_time_ms": result.execution_time_ms
            })
        
        except Exception as e:
            return f"Execution Error: {str(e)}"
    
    # Create tools
    tools = [
        Tool(
            name="schema_introspector",
            func=schema_introspector,
            description="Get the schema (columns, types) of a database table. Useful to understand table structure before writing queries.",
        ),
        Tool(
            name="list_tables",
            func=list_available_tables,
            description="List all available tables in the current database schema.",
        ),
        Tool(
            name="query_executor",
            func=query_executor,
            description="Execute a READ-ONLY SQL query. Only SELECT statements allowed.",
        ),
    ]
    
    # Get the task from the latest message
    last_message = state["messages"][-1]
    user_task = last_message.content
    
    # Construct the system prompt for SQL generation
    sql_system_prompt = """You are a SQL Expert specialized in generating and executing database queries.

Your approach:
1. Understand the user's data request
2. Use the schema_introspector tool to understand table structures
3. Generate appropriate SQL (Snowflake-compatible)
4. Execute the query using the query_executor tool
5. Interpret and explain the results

IMPORTANT RULES:
- Only generate SELECT queries (read-only)
- Always validate table and column names exist before executing
- If a query fails, analyze the error and suggest a fix
- Be precise with data types and NULL handling
- Optimize queries for clarity and performance

The user's request:
{user_task}

Begin by exploring the available tables and schema, then construct and execute your query.
"""
    
    # Use ReAct-style agent for SQL generation
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
            "system_prompt": sql_system_prompt
        })
        
        sql_result = result.get("output", "No output")
        
        return {
            "next_step": "sql_agent",  # Return to supervisor
            "messages": [
                AIMessage(
                    content=f"[SQL_WORKER] Query Result:\n{sql_result}",
                    name="sql_agent"
                )
            ]
        }
    
    except Exception as e:
        logger.error(f"SQL agent execution failed: {str(e)}")
        return {
            "next_step": "sql_agent",
            "error_message": str(e),
            "messages": [
                AIMessage(
                    content=f"[SQL_WORKER_ERROR] {str(e)}",
                    name="sql_agent"
                )
            ]
        }
    
    finally:
        db_connector.disconnect()


def __get_env(key: str, default: str = "") -> str:
    """Helper to get environment variables."""
    import os
    return os.getenv(key, default)
