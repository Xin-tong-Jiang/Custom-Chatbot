#!/bin/bash
# Start the React frontend
cd frontend
echo "Installing React dependencies..."
npm install
echo ""
echo "Starting React development server..."
echo "Frontend will be available at: http://localhost:3000"
echo "Press Ctrl+C to stop the server"
echo "----------------------------------------"
npm start
