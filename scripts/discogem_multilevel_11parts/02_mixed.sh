#!/usr/bin/env bash
set -euo pipefail
d="$(dirname "${BASH_SOURCE[0]}")"
bash "$d/run_part.sh" 'english|roberta-base|soft rel|42' 'english|bert-base-uncased|soft rel|42' 'multilingual|microsoft/infoxlm-base|soft mse|44' 'multilingual|xlm-roberta-base|soft rel|42'
