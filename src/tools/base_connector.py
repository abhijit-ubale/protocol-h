"""
Base connector interface for cloud-agnostic database access.

Implements the Adapter Pattern to abstract database implementations
and allow the same agentic code to work across different platforms
(Snowflake, Redshift, BigQuery, etc.).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class TableSchema:
    """Schema information for a database table."""
    table_name: str
    columns: List[Dict[str, str]]  # List of {name, type, nullable, description}
    row_count: Optional[int] = None
    description: Optional[str] = None


@dataclass
class QueryResult:
    """Result of a database query execution."""
    success: bool
    data: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    row_count: int = 0
    execution_time_ms: float = 0.0


class BaseConnector(ABC):
    """
    Abstract base class for database connectors.
    
    Defines the interface that all cloud-specific implementations must adhere to.
    This ensures that worker agents can interact with any database backend
    without knowing implementation details.
    """
    
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the database."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close the database connection."""
        pass
    
    @abstractmethod
    def get_table_schema(self, table_name: str) -> TableSchema:
        """
        Retrieve schema information for a specific table.
        
        Args:
            table_name: Name of the table to introspect
        
        Returns:
            TableSchema object containing column definitions
        
        Raises:
            ValueError: If table not found
        """
        pass
    
    @abstractmethod
    def list_tables(self) -> List[str]:
        """
        List all available tables in the current schema/database.
        
        Returns:
            List of table names
        """
        pass
    
    @abstractmethod
    def execute_query(self, sql: str, timeout: int = 30) -> QueryResult:
        """
        Execute a SQL query against the database.
        
        Args:
            sql: SQL query string (read-only queries only)
            timeout: Query timeout in seconds
        
        Returns:
            QueryResult containing data or error information
        
        Raises:
            ValueError: If query appears to be non-read-only (INSERT, UPDATE, DELETE, etc.)
        """
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test if the connector can successfully connect to the database.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass


class ConnectorFactory:
    """
    Factory for creating database connector instances.
    
    Abstracts the instantiation logic and allows registration
    of new connector types.
    """
    
    _connectors: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, connector_class: type) -> None:
        """Register a new connector type."""
        cls._connectors[name.lower()] = connector_class
    
    @classmethod
    def create(cls, connector_type: str, **kwargs) -> BaseConnector:
        """
        Create a connector instance.
        
        Args:
            connector_type: Type of connector ("snowflake", "redshift", "bigquery")
            **kwargs: Configuration parameters for the connector
        
        Returns:
            Instantiated connector
        
        Raises:
            ValueError: If connector type not recognized
        """
        connector_class = cls._connectors.get(connector_type.lower())
        if not connector_class:
            raise ValueError(f"Unknown connector type: {connector_type}")
        
        return connector_class(**kwargs)
