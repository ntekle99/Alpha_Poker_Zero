#!/usr/bin/env python3
"""Start both frontend and backend servers."""

import subprocess
import sys
import os
import time
import signal
from pathlib import Path

# Colors for terminal output
GREEN = '\033[0;32m'
BLUE = '\033[0;34m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
NC = '\033[0m'  # No Color

backend_process = None
frontend_process = None


def cleanup(signum=None, frame=None):
    """Cleanup function to stop both servers."""
    print(f"\n{YELLOW}Shutting down servers...{NC}")
    if backend_process:
        backend_process.terminate()
        backend_process.wait()
    if frontend_process:
        frontend_process.terminate()
        frontend_process.wait()
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


def check_dependencies():
    """Check and install dependencies if needed."""
    # Check backend dependencies
    try:
        import flask
        import flask_cors
    except ImportError:
        print(f"{YELLOW}Installing backend dependencies...{NC}")
        subprocess.run([sys.executable, "-m", "pip", "install", "flask", "flask-cors"], check=True)
    
    # Check frontend dependencies
    frontend_dir = Path(__file__).parent / "frontend"
    node_modules = frontend_dir / "node_modules"
    if not node_modules.exists():
        print(f"{YELLOW}Installing frontend dependencies...{NC}")
        subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)


def main():
    """Start both servers."""
    global backend_process, frontend_process
    
    print(f"{BLUE}🚀 Starting Alpha Poker Zero...{NC}\n")
    
    # Check dependencies
    check_dependencies()
    
    # Get project root directory
    project_root = Path(__file__).parent
    frontend_dir = project_root / "frontend"
    
    # Start backend
    print(f"{BLUE}Starting backend server on port 5001...{NC}")
    backend_process = subprocess.Popen(
        [sys.executable, "backend.py"],
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Wait a moment for backend to start
    time.sleep(2)
    
    # Check if backend is still running
    if backend_process.poll() is not None:
        print(f"{RED}Backend failed to start!{NC}")
        if backend_process.stdout:
            print(backend_process.stdout.read())
        sys.exit(1)
    
    # Start frontend
    print(f"{BLUE}Starting frontend server on port 3000...{NC}")
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=frontend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Wait a moment for frontend to start
    time.sleep(3)
    
    print(f"\n{GREEN}✅ Both servers are running!{NC}\n")
    print(f"📡 Backend:  http://localhost:5001")
    print(f"🌐 Frontend: http://localhost:3000")
    print(f"\n{YELLOW}Press Ctrl+C to stop both servers{NC}\n")
    
    # Print output from both processes
    try:
        while True:
            # Check backend output
            if backend_process.stdout and backend_process.stdout.readable():
                line = backend_process.stdout.readline()
                if line:
                    print(f"[Backend] {line.rstrip()}")
            
            # Check frontend output
            if frontend_process.stdout and frontend_process.stdout.readable():
                line = frontend_process.stdout.readline()
                if line:
                    print(f"[Frontend] {line.rstrip()}")
            
            # Check if processes are still running
            if backend_process.poll() is not None:
                print(f"{RED}Backend process ended!{NC}")
                break
            if frontend_process.poll() is not None:
                print(f"{RED}Frontend process ended!{NC}")
                break
            
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()


if __name__ == "__main__":
    main()

