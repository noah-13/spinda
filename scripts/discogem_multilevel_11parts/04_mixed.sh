#!/usr/bin/env bash
set -euo pipefail
d="$(dirname "${BASH_SOURCE[0]}")"
bash "$d/run_part.sh" 'english|roberta-base|soft rel|44' 'english|bert-base-uncased|soft rel|44' 'multilingual|microsoft/infoxlm-base|soft jsd|43' 'multilingual|xlm-roberta-base|soft rel|44'
