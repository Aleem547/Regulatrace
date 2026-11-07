"""Read completed chat measurements: python scripts/query_log_report.py."""

import math
from decimal import Decimal
from statistics import median, quantiles

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from regulatrace.db.models import QueryLog
from regulatrace.db.session import get_engine, get_session_factory


def main() -> None:
    engine = get_engine()
    try:
        with get_session_factory(engine)() as session:
            rows = session.execute(select(QueryLog.latency_per_stage, QueryLog.cost)).all()
    except SQLAlchemyError:
        raise SystemExit("Could not read query_logs; check PostgreSQL and migrations.") from None
    finally:
        engine.dispose()

    print(f"Completed query log records: {len(rows)}")
    print(f"{'Stage (ms)':<20} {'Samples':>8} {'p50':>12} {'p95':>12}")
    typical = {}
    for stage in ("total", "rewrite", "retrieval", "reranking", "llm_generation"):
        values = [
            timings[stage]
            for timings, _ in rows
            if timings
            and isinstance(timings.get(stage), int | float)
            and not isinstance(timings[stage], bool)
            and math.isfinite(timings[stage])
            and timings[stage] >= 0
        ]
        if not values:
            print(f"{stage:<20} {0:>8} {'n/a':>12} {'n/a':>12}")
            continue
        p50 = median(values)
        # Inclusive interpolation supports small samples; a single sample repeats itself.
        p95 = quantiles(values, n=100, method="inclusive")[94] if len(values) > 1 else values[0]
        print(f"{stage:<20} {len(values):>8} {p50:>12.2f} {p95:>12.2f}")
        if stage != "total" and p50 > 0:
            typical[stage] = p50
    dominant = max(typical, key=typical.get) if typical else "n/a"
    print(f"Dominant stage by p50: {dominant}")
    costs = [cost for _, cost in rows if cost is not None]
    if costs:
        average = sum(costs, Decimal(0)) / len(costs)
        print(
            f"Estimated average LLM cost/query: ${average:.8f} ({len(costs)}/{len(rows)} records)"
        )
    else:
        print("Estimated average LLM cost/query: unknown (0 records with cost data)")
    print("Skipped stages count as 0 ms. Small samples make p95 unstable; costs are estimates.")


if __name__ == "__main__":
    main()
