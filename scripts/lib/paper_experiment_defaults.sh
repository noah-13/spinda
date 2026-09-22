#!/usr/bin/env bash
# Shared defaults for the canonical paper-reproduction launchers.

# Every paper launcher uses this training configuration unless explicitly
# overridden for an ablation with TRAINING_CONFIG.
PAPER_TRAINING_CONFIG="${PAPER_TRAINING_CONFIG:-configs/training.json}"

# Keep model comparisons consistent within each language scope. Launchers may
# still override MODEL_SPECS for a deliberately smaller experimental run.
PAPER_ENGLISH_MODEL_SPECS="microsoft/deberta-v3-large;roberta-base;bert-base-uncased;Twitter/twhin-bert-base;xlm-roberta-base;bert-base-multilingual-cased;microsoft/infoxlm-base"
PAPER_MULTILINGUAL_MODEL_SPECS="xlm-roberta-base;bert-base-multilingual-cased;microsoft/infoxlm-base"
