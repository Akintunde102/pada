"""JSON Schema exporter for Jev contract models."""

import json
from pathlib import Path
from pada.schema.request import JevRequest
from pada.schema.response import JevResponse


def export_schemas(output_dir: Path | str | None = None) -> None:
    if output_dir is None:
        target = Path(__file__).resolve().parent / "json_schemas"
    else:
        target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    req_schema = JevRequest.model_json_schema()
    res_schema = JevResponse.model_json_schema()

    with open(target / "jev_request.schema.json", "w", encoding="utf-8") as f:
        json.dump(req_schema, f, indent=2)

    with open(target / "jev_response.schema.json", "w", encoding="utf-8") as f:
        json.dump(res_schema, f, indent=2)

    print(f"Exported JSON schemas to {target}")


if __name__ == "__main__":
    export_schemas()
