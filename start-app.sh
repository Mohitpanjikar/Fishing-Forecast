#!/bin/bash

# Start the Python ML backend
echo "Starting Python ML backend..."
cd api
python3 -m venv venv 2>/dev/null || true
source venv/bin/activate
pip install -r requirements.txt
python app.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start and check if it's running
echo "Waiting for backend to start..."
sleep 5
if ! curl -s http://localhost:5000/api/health > /dev/null; then
  echo "WARNING: Backend is not responding. Models will use simulation mode."
  echo "Check if port 5000 is available and Python dependencies are installed correctly."
else
  echo "Backend is running and healthy!"
fi

# Start the frontend
echo "Starting frontend application..."
npm run dev &
FRONTEND_PID=$!

# Handle graceful shutdown
function cleanup {
  echo "Shutting down services..."
  kill $BACKEND_PID
  kill $FRONTEND_PID
  exit 0
}

trap cleanup SIGINT

echo "Both services are running!"
echo "ML Backend: http://localhost:5000/api"
echo "Frontend: http://localhost:5173"
echo "Press Ctrl+C to stop all services"

# Keep the script running
wait 