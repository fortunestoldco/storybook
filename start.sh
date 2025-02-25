#!/bin/bash

# Print header
echo "=== Storybook Service Launcher ==="
echo "Started at: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "User: $USER"
echo "=========================="

# Create and activate Python virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install/upgrade langgraph-cli if needed
echo "Installing/upgrading langgraph-cli..."
pip install --upgrade langgraph-cli[inmem]

# Start langgraph in the background
echo "Starting langgraph server..."
langgraph up &
LANGGRAPH_PID=$!

# Wait a moment for langgraph to start
sleep 2

# Change to frontend directory
echo "Changing to frontend directory..."
cd frontend

# Install frontend dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    yarn install
fi

# Start Next.js on port 8080
echo "Starting Next.js frontend on port 8080..."
yarn start -p 8080 &
NEXTJS_PID=$!

# Function to handle shutdown
cleanup() {
    echo "Shutting down services..."
    kill $LANGGRAPH_PID
    kill $NEXTJS_PID
    deactivate
    exit 0
}

# Set up trap for cleanup on script termination
trap cleanup SIGINT SIGTERM

# Keep script running
echo "Services are running. Press Ctrl+C to stop."
wait
