from exam_bank.topic_review_loop import build_topic_review_batch_from_args, add_topic_review_batch_cli_arguments

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an automated topic review batch.")
    add_topic_review_batch_cli_arguments(parser)
    report = build_topic_review_batch_from_args(parser.parse_args())
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
