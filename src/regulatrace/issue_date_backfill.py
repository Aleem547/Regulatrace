"""One-time payload-only issue-date update for the three existing Qdrant indexes."""

import argparse
from collections import defaultdict

from qdrant_client import QdrantClient, models

from regulatrace.config import get_settings
from regulatrace.ingestion.metadata import normalize_issue_date
from regulatrace.vectorstore import get_qdrant_client


def update_issue_dates(
    client: QdrantClient | None = None, *, apply: bool = False, confirmed: bool = False
) -> dict[str, int]:
    """Validate every point first; optionally set only missing issue_date_iso values."""
    if apply and not confirmed:
        raise ValueError("Apply mode requires explicit confirmation")
    settings = get_settings()
    names = (settings.qdrant_collection, settings.bm25_collection, settings.hybrid_collection)
    if len(set(names)) != 3:
        raise ValueError("Collection names must be distinct")
    q_client = client if client is not None else get_qdrant_client()

    collections = {}
    for name in names:
        if not q_client.collection_exists(name):
            raise ValueError(f"Collection {name!r} does not exist")
        points = {}
        offset = None
        while True:
            page, next_offset = q_client.scroll(
                collection_name=name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in page:
                if point.id in points:
                    raise ValueError(f"Duplicate point ID {point.id} in {name}")
                payload = point.payload or {}
                try:
                    expected = normalize_issue_date(payload.get("issue_date"))
                except ValueError as exc:
                    raise ValueError(f"Invalid issue_date for point {point.id} in {name}") from exc
                existing = payload.get("issue_date_iso")
                if existing is not None and existing != expected:
                    raise ValueError(f"Conflicting issue_date_iso for point {point.id} in {name}")
                points[point.id] = (payload, expected)
            if next_offset is None:
                break
            offset = next_offset
        if len(points) != 669:
            raise ValueError(f"Expected 669 points in {name}, found {len(points)}")
        collections[name] = points

    source = collections[names[0]]
    for name in names[1:]:
        target = collections[name]
        if source.keys() != target.keys():
            raise ValueError(f"Point IDs differ between {names[0]} and {name}")
        for point_id, (payload, expected) in source.items():
            other_payload, other_expected = target[point_id]
            if expected != other_expected or {
                key: value for key, value in payload.items() if key != "issue_date_iso"
            } != {key: value for key, value in other_payload.items() if key != "issue_date_iso"}:
                raise ValueError(f"Payload or date differs for point {point_id} in {name}")

    pending = {}
    for name, points in collections.items():
        groups = defaultdict(list)
        for point_id, (payload, expected) in points.items():
            if payload.get("issue_date_iso") is None:
                groups[expected].append(point_id)
        pending[name] = sum(len(ids) for ids in groups.values())
        if apply:
            for expected, ids in groups.items():
                for index in range(0, len(ids), 100):
                    result = q_client.set_payload(
                        collection_name=name,
                        payload={"issue_date_iso": expected},
                        points=models.PointIdsList(points=ids[index : index + 100]),
                        wait=True,
                    )
                    if result.status != models.UpdateStatus.COMPLETED:
                        raise RuntimeError(f"Payload update did not complete in {name}")
    return pending


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate or update issue_date_iso payloads")
    parser.add_argument(
        "--apply", action="store_true", help="Update missing dates after validation"
    )
    parser.add_argument("--confirm", action="store_true", help="Confirm the authorized write")
    args = parser.parse_args()
    for name, count in update_issue_dates(apply=args.apply, confirmed=args.confirm).items():
        print(f"{name}: 669 points; {count} dates {'updated' if args.apply else 'pending'}")


if __name__ == "__main__":
    main()
