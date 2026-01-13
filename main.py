"""
Main application entry point for the Hierarchical Agentic RAG system.

This module provides a simple CLI interface to the orchestrator.
For production, wrap with FastAPI or similar web framework.
"""

import argparse
import json
import logging
from langchain_core.messages import HumanMessage
from src.graph.workflow import create_orchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    
    parser = argparse.ArgumentParser(
        description="Protocol-H: Enterprise Hierarchical Agentic RAG"
    )
    parser.add_argument(
        "query",
        type=str,
        help="User query for the orchestrator"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="Output result as JSON"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Initializing Hierarchical Agentic RAG Orchestrator...")
    
    # Create orchestrator
    orchestrator = create_orchestrator()
    app = orchestrator.get_compiled_app()
    
    logger.info(f"Processing query: {args.query}")
    
    # Prepare initial state
    initial_state = {
        "messages": [HumanMessage(content=args.query)],
        "next_step": "supervisor",
        "final_answer": None,
        "query_type": None,
        "retry_count": 0,
        "error_message": None,
    }
    
    # Execute orchestration
    try:
        result = app.invoke(initial_state)
        
        if args.output_json:
            output = {
                "query": args.query,
                "answer": result.get("final_answer", ""),
                "query_type": result.get("query_type"),
                "retry_count": result.get("retry_count", 0),
            }
            print(json.dumps(output, indent=2))
        else:
            print("\n" + "="*70)
            print("FINAL ANSWER:")
            print("="*70)
            print(result.get("final_answer", "No answer generated"))
            print("="*70)
        
    except Exception as e:
        logger.error(f"Orchestration failed: {str(e)}")
        if args.output_json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print(f"\nError: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
