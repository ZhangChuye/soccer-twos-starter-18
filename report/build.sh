#!/usr/bin/env bash
# Build the report with tectonic (no external LaTeX install required).
set -e
cd "$(dirname "$0")"
~/.local/bin/tectonic -X compile report.tex
echo "Built: $(pwd)/report.pdf"
