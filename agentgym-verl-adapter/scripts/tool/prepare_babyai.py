"""Convert AgentGym item_id JSON into official verl v0.5.0 Parquet rows.

Example:
 python scripts/prepare_babyai.py --input /path/babyai_train.json --output data/babyai_train.parquet
"""


import argparse
import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from agentenv.envs.babyai import BabyAIEnvClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    items = json.loads(Path(args.input).read_text(encoding="utf-8"))
    intro = BabyAIEnvClient.conversation_start
    if not isinstance(items, list) or not items:
        raise ValueError("Expected a non-empty JSON list")

    rows = []
    seen = set()
    for item in items:
        item_id = item["item_id"]
        match = re.fullmatch(r"babyai_(\d+)", item_id)
        if match is None or item_id in seen:
            raise ValueError(f"Invalid or duplicate item_id: {item_id!r}")
        seen.add(item_id)
        task_id = int(match.group(1))
        rows.append({
            "data_source": "babyai",
            "agent_name": "babyai_agent",
            "prompt": [
                {"role": "user", "content": intro[0]["value"], "item_id": item_id},
                {"role": "assistant", "content": intro[1]["value"], "item_id": item_id},
            ],
            "reward_model": {"ground_truth": ""},
            "extra_info": {"index": task_id, "item_id": item_id},
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out)
    print(f"Created {out} with {len(rows)} rows, {len(seen)} unique task IDs")
    print(f"Range: {min(int(x['item_id'].split('_')[-1]) for x in items)} .. "
          f"{max(int(x['item_id'].split('_')[-1]) for x in items)}")
    print(f"First row: {rows[0]['extra_info']}")


if __name__ == "__main__":
    main()
