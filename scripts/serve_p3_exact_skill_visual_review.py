from __future__ import annotations

import argparse
from functools import partial
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from exam_bank.atomic_json import write_atomic_json
from exam_bank.p3_exact_skill import DEFAULT_REVIEW_BATCH_DIR


class ReviewSaveHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, output_path: Path, **kwargs: Any) -> None:
        self.output_path = output_path
        super().__init__(*args, **kwargs)

    def do_POST(self) -> None:
        if self.path != "/p3-exact-skill-review-responses":
            self.send_error(404, "Unknown endpoint")
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as exc:
            self.send_error(400, f"Invalid JSON: {exc}")
            return
        if not isinstance(payload, dict) or not isinstance(payload.get("responses"), list):
            self.send_error(400, "Expected a response payload with a responses list")
            return
        write_atomic_json(payload, self.output_path, sort_keys=True)
        body = json.dumps({"ok": True, "path": str(self.output_path)}, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve the P3 exact-skill visual review packet and save browser responses into the repo."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--batch-dir", type=Path, default=Path(DEFAULT_REVIEW_BATCH_DIR))
    parser.add_argument("--batch-id", default="batch_0001")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    batch_dir = args.batch_dir if args.batch_dir.is_absolute() else repo_root / args.batch_dir
    output_path = args.output or batch_dir / f"{args.batch_id}_review_responses.v1.json"
    output_path = output_path if output_path.is_absolute() else repo_root / output_path
    html_path = batch_dir / f"{args.batch_id}_visual_review.html"
    handler = partial(ReviewSaveHandler, directory=str(repo_root), output_path=output_path)

    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving repo root: {repo_root}")
    print(f"Open: http://{args.host}:{args.port}/{html_path.relative_to(repo_root).as_posix()}")
    print(f"Saving responses to: {output_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
