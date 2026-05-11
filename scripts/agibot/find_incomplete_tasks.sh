#!/bin/bash
# Find tasks that were started but not completed (no sentinel file)
# Usage: ./find_incomplete_tasks.sh

set -euo pipefail

export CURL_CA_BUNDLE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
export SSL_CERT_FILE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem

GCS_STAGING="${AGIBOT_GCS_STAGING:-gs://pi0-cot/agibot_staging}"
TASKS_JSON="${AGIBOT_BASE:-/n/fs/robot-data/data/agibot}/tasks.json"

# Get all task IDs from tasks.json
ALL_TASKS=$(python3 -c "
import json
with open('$TASKS_JSON') as f:
    tasks = json.load(f)
for i, t in enumerate(tasks):
    print(i, t['task_id'])
")

# Get completed tasks (those with sentinel files)
COMPLETED=$(gsutil ls "${GCS_STAGING}/sentinels/_DONE_*" 2>/dev/null | sed 's/.*_DONE_//' || true)

echo "=== Incomplete Tasks (no sentinel) ==="
echo "array_idx task_id"
echo "$ALL_TASKS" | while read idx tid; do
    if ! echo "$COMPLETED" | grep -q "^${tid}$"; then
        echo "$idx $tid"
    fi
done

echo ""
echo "=== To resume, run: ==="
echo 'sbatch --array=$(./scripts/agibot/find_incomplete_tasks.sh | grep "^[0-9]" | cut -d" " -f1 | paste -sd,) scripts/agibot/submit_process_array.sbatch'
