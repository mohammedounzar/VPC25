#!/bin/bash

##############################################
# DO NOT MODIFY THIS FILE
##############################################

# Script for running your anonymization algorithm
##############################################

run() {
    echo "Running evaluation ..."

    # Check if virtual environment exists
    if [ ! -d ".venv" ]; then
        echo "Virtual environment not found! Creating one..."
        python3.12 -m venv .venv
    fi

    # Activate virtual environment
    source .venv/bin/activate

    # Ensure required packages are installed
    pip install -r requirements.txt

    # Run the evaluation
    python evaluation.py

    echo "... end of evaluation run"
}

run
