#!/usr/bin/env bash
set -euo pipefail
d="$(dirname "${BASH_SOURCE[0]}")"
bash "$d/run_part.sh" 'english|bert-base-multilingual-cased|soft rel|43' 'english|Twitter/twhin-bert-base|soft_to_hard ce|43' 'multilingual|microsoft/infoxlm-base|soft_to_hard ce|42' 'multilingual|bert-base-multilingual-cased|soft_to_hard ce|43'
