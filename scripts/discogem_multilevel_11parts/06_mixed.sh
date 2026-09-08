#!/usr/bin/env bash
set -euo pipefail
d="$(dirname "${BASH_SOURCE[0]}")"
bash "$d/run_part.sh" 'english|xlm-roberta-base|soft rel|43' 'english|Twitter/twhin-bert-base|soft rel|43' 'multilingual|microsoft/infoxlm-base|soft rel|42' 'multilingual|xlm-roberta-base|soft_to_hard ce|43'
