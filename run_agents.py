
"""
Run Multi-Agent Regression Pipeline.

"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def main():
    """Run the multi-agent pipeline."""
    parser = argparse.ArgumentParser(description="Run Multi-Agent ML Pipeline")
    
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to training data CSV",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=None,
        help="Path to test data CSV (optional)",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="target",
        help="Name of target column",
    )
    parser.add_argument(
        "--rag-storage",
        type=str,
        default="./rag_storage",
        help="Path to RAG storage directory",
    )
    parser.add_argument(
        "--working-dir",
        type=str,
        default="./workspace",
        help="Working directory for code execution",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Maximum retry attempts per agent",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Execution timeout in seconds",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen/qwen2.5-7b-instruct",
        help="LLM model for code generation",
    )
    parser.add_argument(
        "--max-feedback-iterations",
        type=int,
        default=2,
        help="Maximum number of evaluator-driven feedback iterations",
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip EDA agent",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip Train agent",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip Eval agent",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON (optional)",
    )
    
    args = parser.parse_args()
    
    # Validate data path
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("Multi-Agent Regression Pipeline")
    logger.info("=" * 60)
    logger.info(f"Data path: {data_path}")
    logger.info(f"Test path: {args.test_path or 'N/A'}")
    logger.info(f"Target column: {args.target_column}")
    logger.info(f"RAG storage: {args.rag_storage}")
    logger.info(f"Working directory: {args.working_dir}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Max attempts: {args.max_attempts}")
    logger.info(f"Timeout: {args.timeout}s")
    logger.info("=" * 60)
    
    # Check RAG storage
    rag_path = Path(args.rag_storage)
    if not rag_path.exists():
        logger.warning(f"RAG storage not found: {rag_path}")
        logger.warning("RAG search will not work until you index some examples.")
        logger.warning("Run: python -m langflow_components.rag.indexer")
    else:
        logger.info(f"RAG storage found: {rag_path}")
    
    # Import and create supervisor
    from agents.supervisor_agent import SupervisorAgent
    
    supervisor = SupervisorAgent(
        rag_storage_path=args.rag_storage,
        working_dir=args.working_dir,
        max_attempts=args.max_attempts,
        timeout=args.timeout,
        model=args.model,
        max_feedback_iterations=args.max_feedback_iterations,
    )
    
    # Run pipeline
    results = supervisor.run_pipeline(
        data_path=str(data_path),
        target_column=args.target_column,
        test_path=args.test_path,
        run_eda=not args.skip_eda,
        run_train=not args.skip_train,
        run_eval=not args.skip_eval,
    )
    
    # Save results if requested
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert any non-serializable values
        serializable_results = {}
        for key, value in results.items():
            if key == "final_state" and value:
                serializable_results[key] = {
                    "data_path": value.get("data_path"),
                    "model_path": value.get("model_path"),
                    "metrics": value.get("metrics", {}),
                    "artifacts": value.get("artifacts", []),
                    "errors": value.get("errors", []),
                }
            else:
                serializable_results[key] = value
        
        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2, default=str)
        logger.info(f"Results saved to: {output_path}")
    
    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Session: {results.get('session_dir', 'N/A')}")
    logger.info(f"Agents run: {results.get('agents_run', [])}")
    logger.info(f"Success: {results.get('success', False)}")
    
    if results.get("errors"):
        logger.warning(f"Errors: {len(results['errors'])}")
        for i, error in enumerate(results["errors"], 1):
            logger.warning(f"  {i}. {error}")
    
    if results.get("final_state"):
        state = results["final_state"]
        logger.info(f"Model: {state.get('model_path', 'N/A')}")
        logger.info(f"Metrics: {state.get('metrics', {})}")
        logger.info(f"Artifacts: {state.get('artifacts', [])}")
    
    logger.info("=" * 60)
    
    sys.exit(0 if results.get("success") else 1)


if __name__ == "__main__":
    main()
