#!/bin/bash

# Quick start script for backdoor detection experiment

echo "=================================================="
echo "Backdoor Detection in RL-Controlled EV Charging"
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Test setup
echo ""
echo "Testing setup..."
python test_setup.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "Setup complete! Running experiment..."
    echo "=================================================="
    echo ""
    cd experiments
    python run_experiment.py

    echo ""
    echo "=================================================="
    echo "Generating visualizations..."
    echo "=================================================="
    python visualize_results.py
else
    echo ""
    echo "Setup test failed. Please check errors above."
    exit 1
fi
