#!/usr/bin/env bash
set -euo pipefail
d="$(dirname "${BASH_SOURCE[0]}")"
bash "$d/run_part.sh" 'english|roberta-base|soft rel|43' 'english|bert-base-uncased|soft rel|43' 'multilingual|microsoft/infoxlm-base|soft jsd|42' 'multilingual|xlm-roberta-base|soft rel|43'
