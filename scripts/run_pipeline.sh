#!/usr/bin/env bash
set -euo pipefail

python -m dirty_signal_lab.cli run --symbol DEMO --n 20000 --seed 7
