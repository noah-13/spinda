#!/usr/bin/env bash
set -euo pipefail
d="$(dirname "${BASH_SOURCE[0]}")"
bash "$d/run_part.sh" 'english|bert-base-multilingual-cased|soft rel|44' 'multilingual|microsoft/infoxlm-base|soft_to_hard ce|43' 'multilingual|bert-base-multilingual-cased|soft rel|42' 'multilingual|bert-base-multilingual-cased|soft_to_hard ce|44'
