"""Create the Quickr keyspace and tables in the local Cassandra node.

Run after `docker compose up -d` and after Cassandra is healthy:
    .venv\\Scripts\\python.exe scripts\\init_cassandra.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from cassandra.cluster import Cluster, ExecutionProfile, EXEC_PROFILE_DEFAULT
from cassandra.policies import DCAwareRoundRobinPolicy
from dotenv import load_dotenv
import os

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

HOST = os.getenv("CASSANDRA_HOST", "127.0.0.1")
PORT = int(os.getenv("CASSANDRA_PORT", "9043"))
DATACENTER = os.getenv("CASSANDRA_DATACENTER", "datacenter1")
SCHEMA = ROOT / "cassandra" / "schema.cql"


def statements_from_schema(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    lines = []
    for line in raw.splitlines():
        if line.strip().startswith("--"):
            continue
        lines.append(line)
    return [stmt.strip() for stmt in "\n".join(lines).split(";") if stmt.strip()]


def main() -> int:
    profile = ExecutionProfile(
        load_balancing_policy=DCAwareRoundRobinPolicy(local_dc=DATACENTER)
    )
    cluster = Cluster(
        [HOST],
        port=PORT,
        connect_timeout=8,
        control_connection_timeout=8,
        protocol_version=4,
        execution_profiles={EXEC_PROFILE_DEFAULT: profile},
    )
    print(f"Connecting to {HOST}:{PORT} ...")
    deadline = time.time() + 120
    session = None
    while time.time() < deadline:
        try:
            session = cluster.connect()
            break
        except Exception as exc:
            print(f"  Cassandra not ready yet ({type(exc).__name__}); retrying ...")
            time.sleep(5)
    if session is None:
        print("ERROR: Cassandra did not become ready within 120 seconds.")
        cluster.shutdown()
        return 1

    for statement in statements_from_schema(SCHEMA):
        first = " ".join(statement.split())[:80]
        print(f"Executing: {first} ...")
        session.execute(statement)

    rows = session.execute(
        "SELECT table_name FROM system_schema.tables WHERE keyspace_name = 'quickr'"
    )
    tables = sorted(row.table_name for row in rows)
    print(f"\nKeyspace 'quickr' is ready. Tables ({len(tables)}):")
    for table in tables:
        print(f"  - {table}")
    cluster.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
