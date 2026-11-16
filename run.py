import sys
import os
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(
        description="GlycoGuard AI – Clinical Diabetes Decision Support System Runner"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the pipeline, perform hyperparameter tuning, and export artifacts"
    )
    parser.add_argument(
        "--server",
        action="store_true",
        default=True,
        help="Start the FastAPI backend and serve the Web UI (Default)"
    )
    parser.add_argument(
        "--streamlit",
        action="store_true",
        help="Start the Streamlit analytical dashboard"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Execute the pytest automated test suite"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number (default: 8000)"
    )

    args, unknown = parser.parse_known_args()

    # Train action
    if args.train:
        print("[*] Starting training pipeline...")
        from src.train import train_and_export
        train_and_export()
        return

    # Test action
    if args.test:
        print("[*] Running automated test suite...")
        subprocess.run([sys.executable, "-m", "pytest", "-v", "tests/test_pipeline.py"])
        return

    # Streamlit action
    if args.streamlit:
        print(f"[*] Launching Streamlit dashboard on http://{args.host}:8501 ...")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.address", args.host, "--server.port", "8501"
        ])
        return

    # Default: FastAPI Server + Web UI
    print(f"================================================================")
    print(f"  🩺 GlycoGuard AI – Clinical Decision Support Platform")
    print(f"================================================================")
    print(f"  Web Dashboard : http://{args.host}:{args.port}")
    print(f"  Swagger Docs  : http://{args.host}:{args.port}/docs")
    print(f"  ReDoc         : http://{args.host}:{args.port}/redoc")
    print(f"================================================================")
    
    import uvicorn
    uvicorn.run("backend.app:app", host=args.host, port=args.port, reload=False)

if __name__ == "__main__":
    main()
