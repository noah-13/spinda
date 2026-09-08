#!/usr/bin/env bash
set -euo pipefail
d="$(dirname "${BASH_SOURCE[0]}")"
bash "$d/run_part.sh" 'english|xlm-roberta-base|soft_to_hard ce|42' 'multilingual|microsoft/infoxlm-base|soft_to_hard ce|44' 'multilingual|bert-base-multilingual-cased|soft rel|43' 'multilingual|bert-base-multilingual-cased|soft rel|44'
