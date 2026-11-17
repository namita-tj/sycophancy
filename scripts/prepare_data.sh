#!/usr/bin/env bash
# Download and convert the evaluation data into data/eval/.
#   pressure.jsonl    600 MMLU questions for factual pressure, corrections and multi-turn pressure
#   capability.jsonl  300 different MMLU questions for the capability guard
#   opinion.jsonl     300 items from each of the three Perez et al. sycophancy files
# Needs: pip install -e ".[data]"
set -euo pipefail

OUT=${1:-data/eval}
RAW=data/raw
mkdir -p "$OUT" "$RAW"

BASE=https://raw.githubusercontent.com/anthropics/evals/main/sycophancy
for f in sycophancy_on_nlp_survey sycophancy_on_philpapers2020 sycophancy_on_political_typology_quiz; do
  [ -f "$RAW/$f.jsonl" ] || curl -sSfL "$BASE/$f.jsonl" -o "$RAW/$f.jsonl"
done

syco data perez "$RAW"/sycophancy_on_*.jsonl --shuffle-seed 0 --limit 300 --out "$OUT/opinion.jsonl"
syco data mcq --hf cais/mmlu:all:test --shuffle-seed 0 --limit 600 --source mmlu --out "$OUT/pressure.jsonl"
syco data mcq --hf cais/mmlu:all:test --shuffle-seed 0 --skip 600 --limit 300 --source mmlu --out "$OUT/capability.jsonl"
syco data synthetic --n 4000 --seed 0 --out data/train/synthetic.jsonl
