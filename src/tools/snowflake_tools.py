"""
Snowflake-specific database connector implementation.

Leverages the Snowflake connector and SQLAlchemy for schema introspection
and query execution. Implements the BaseConnector interface.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

import snowflake.connector
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

from src.tools.base_connector import BaseConnector, TableSchema, QueryResult

logger = logging.getLogger(__name__)


class SnowflakeConnector(BaseConnector):
    """
    Cloud-agnostic connector for Snowflake data warehouses.
    
    Features:
    - Automatic schema introspection
    - Query validation (read-only enforcement)
    - Connection pooling
    - Error handling and logging
    """
    
    def __init__(
        self,
        account: str,
        user: str,
        password: str,
        warehouse: str,
        database: str,
        schema: str,
        role: Optional[str] = "SYSADMIN",
        read_only: bool = True
    ):
        """
        Initialize Snowflake connector.
        
        Args:
            account: Snowflake account identifier
            user: Username
            password: Password
            warehouse: Warehouse name
            database: Database name
            schema: Schema name
            role: Role to assume
            read_only: Enforce read-only queries
        """
        self.account = account
        self.user = user
        self.password = password
        self.warehouse = warehouse
        self.database = database
        self.schema = schema
        self.role = role
        self.read_only = read_only
        
        self.engine = None
        self.connection = None
        self._inspector = None
    
    def connect(self) -> None:
        """Establish connection to Snowflake."""
        try:
            connection_string = (
                f"snowflake://{self.user}:{self.password}@{self.account}/"
                f"{self.database}/{self.schema}?"
                f"warehouse={self.warehouse}&role={self.role}"
            )
            
            self.engine = create_engine(connection_string)
            self.connection = self.engine.connect()
            self._inspector = inspect(self.engine)
            
            logger.info(f"Connected to Snowflake: {self.account}/{self.database}.{self.schema}")
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {str(e)}")
            raise
    
    def disconnect(self) -> None:
        """Close the database connection."""
        if self.connection:
            self.connection.close()
        if self.engine:
            self.engine.dispose()
        logger.info("Disconnected from Snowflake")
    
    def list_tables(self) -> List[str]:
        """
        List all available tables in the current schema.
        
        Returns:
            List of table names
        """
        if not self._inspector:
            raise RuntimeError("Not connected to database")
        
        tables = self._inspector.get_table_names(schema=self.schema)
        return [table.upper() for table in tables]
    
    def get_table_schema(self, table_name: str) -> TableSchema:
        """
        Retrieve schema information for a specific table.
        
        Args:
            table_name: Name of the table
        
        Returns:
            TableSchema object
        """
        if not self._inspector:
            raise RuntimeError("Not connected to database")
        
        try:
            # Get columns
            columns_info = self._inspector.get_columns(
                table_name.upper(), 
                schema=self.schema
            )
            
            columns = []
            for col in columns_info:
                columns.append({
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col.get("nullable", True),
                    "description": col.get("comment", "")
                })
            
            # Get row count
            row_count = None
            try:
                result = self.connection.execute(
                    text(f"SELECT COUNT(*) as cnt FROM {self.schema}.{table_name.upper()}")
                )
                row_count = result.fetchone()[0]
            except Exception as e:
                logger.warning(f"Could not get row count for {table_name}: {str(e)}")
            
            return TableSchema(
                table_name=table_name.upper(),
                columns=columns,
                row_count=row_count,
                description=""
            )
        
        except Exception as e:
            logger.error(f"Failed to get schema for table {table_name}: {str(e)}")
            raise ValueError(f"Table not found: {table_name}")
    
    def execute_query(self, sql: str, timeout: int = 30) -> QueryResult:
        """
        Execute a SQL query against Snowflake.
        
        Args:
            sql: SQL query string
            timeout: Query timeout in seconds
        
        Returns:
            QueryResult with data or error
        """
        if not self.connection:
            return QueryResult(
                success=False,
                error="Not connected to database"
            )
        
        # Validate read-only constraint
        if self.read_only:
            dangerous_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE"]
            sql_upper = sql.strip().upper()
            for keyword in dangerous_keywords:
                if sql_upper.startswith(keyword):
                    return QueryResult(
                        success=False,
                        error=f"Write operations not allowed: {keyword}"
                    )
        
        try:
            start_time = time.time()
            
            result = self.connection.execute(text(sql))
            rows = result.fetchall()
            
            # Convert rows to list of dicts
            columns = result.keys()
            data = [dict(zip(columns, row)) for row in rows]
            
            execution_time = (time.time() - start_time) * 1000  # milliseconds
            
            return QueryResult(
                success=True,
                data=data,
                row_count=len(data),
                execution_time_ms=execution_time
            )
        
        except SQLAlchemyError as e:
            logger.error(f"SQL execution error: {str(e)}")
            return QueryResult(
                success=False,
                error=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error executing query: {str(e)}")
            return QueryResult(
                success=False,
                error=str(e)
            )
    
    def test_connection(self) -> bool:
        """Test if connection is valid."""
        try:
            result = self.connection.execute(text("SELECT 1"))
            return result.fetchone()[0] == 1
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False


# Register the Snowflake connector
from src.tools.base_connector import ConnectorFactory
ConnectorFactory.register("snowflake", SnowflakeConnector)
