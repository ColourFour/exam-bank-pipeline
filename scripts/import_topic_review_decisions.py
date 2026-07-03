from exam_bank.topic_review_loop import add_topic_review_import_cli_arguments, import_topic_review_decisions_from_args

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(description="Import validated automated topic review decisions.")
    add_topic_review_import_cli_arguments(parser)
    report = import_topic_review_decisions_from_args(parser.parse_args())
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
