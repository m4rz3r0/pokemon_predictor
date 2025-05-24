#!/usr/bin/env fish
source venv/bin/activate.fish
python -m pytest tests/ -v --tb=short
