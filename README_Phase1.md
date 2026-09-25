# Business Entity Resolution — Phase 1

Place `src/`, `requirements.txt` and this README inside `student_resource/`, then run from there.

```bash
pip install -r requirements.txt

# (optional) data exploration
python src/eda.py --data dataset | tee eda_report.txt

# 1. train: blocking on train + logistic-regression matcher + F0.5-tuned threshold
python src/phase1_train.py --data dataset --models models

# 2. predict on test -> output/matching_results.tsv + output/candidate_pairs.tsv
python src/phase1_predict.py --data dataset --models models --out output

# 3. validate before uploading
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

Knobs if you run out of RAM or time: `--chunk 20000` (lower memory), `--max-df 300`
(smaller joins), and `--s1-sample 150000` (faster training). Increase `--K` for higher
recall at the cost of speed. Save `models/phase1_report.json`; it contains the recall
ceiling, the oracle score, and the validation macro F0.5.

Files:
- `normalize.py`: text cleaning (accents, punctuation, legal suffixes, abbreviations; includes French).
- `pipeline.py`: IDF-weighted inverted-index blocking and 20 pair features.
- `phase1_train.py`, `phase1_predict.py`: training and inference.
- `metrics.py`: exact per-entity macro F0.5 scorer.
- `eda.py`, `make_empty_submission.py`: exploration and an all-empty probe submission.
