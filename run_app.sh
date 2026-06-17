#!/usr/bin/env bash
# Launch the Streamlit demo (Git Bash).
# Run from the project root:  ./run_app.sh
set -e
cd "$(dirname "$0")"

# Use the locally-downloaded 1.5B weights (avoids re-pulling from HF hub).
export TARGET_MODEL_ID="models/Qwen2.5-1.5B-Instruct"
# Force UTF-8 so model output containing "₹" etc. doesn't crash the console.
export PYTHONIOENCODING="utf-8"

# Use the project virtualenv's interpreter (has torch/transformers/streamlit),
# NOT the system python which lacks the ML deps.
exec .venv/Scripts/python.exe -m streamlit run app.py --server.port 8501 --server.headless true
