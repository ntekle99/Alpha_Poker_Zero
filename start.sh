#!/bin/bash

# Start Alpha Poker Zero - Frontend and Backend

echo "🚀 Starting Alpha Poker Zero..."
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down servers...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit
}

# Trap Ctrl+C
trap cleanup SIGINT SIGTERM

# Check if backend dependencies are installed
if ! python3 -c "import flask" 2>/dev/null; then
    echo -e "${YELLOW}Installing backend dependencies...${NC}"
    pip3 install flask flask-cors
fi

# Start backend
echo -e "${BLUE}Starting backend server on port 5001...${NC}"
cd "$(dirname "$0")"
python3 backend.py > backend.log 2>&1 &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 2

# Check if backend started successfully
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${YELLOW}Backend failed to start. Check backend.log for errors.${NC}"
    exit 1
fi

# Start frontend
echo -e "${BLUE}Starting frontend server on port 3000...${NC}"
cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    npm install
fi

npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!

# Wait a moment for frontend to start
sleep 3

echo ""
echo -e "${GREEN}✅ Both servers are running!${NC}"
echo ""
echo "📡 Backend:  http://localhost:5001"
echo "🌐 Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Wait for processes
wait

