#!/bin/bash

##############################################
# DO NOT MODIFY THIS FILE
##############################################

# Script for running the evaluation of your anonymization algorithm
##############################################

run() {
    echo "Running evaluation ..."

    # Check if virtual environment exists
    # if [ ! -d ".venv" ]; then
    #     echo "Virtual environment not found! Creating one..."
    #     python3.12 -m venv .venv
    # fi

    # # Activate virtual environment
    # source .venv/bin/activate

    # Ensure required packages are installed
    pip install -r requirements.txt --quiet

    # Set evaluation directory (default: "evaluation_data/")
    EVAL_DIR=${1:-"evaluation_data/"}

    # Run the evaluation with the specified directory
    python evaluation.py "$EVAL_DIR"

    echo "... end of evaluation run"
}

run "$@"