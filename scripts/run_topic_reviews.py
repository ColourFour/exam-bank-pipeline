from exam_bank.topic_review_loop import add_topic_review_run_cli_arguments, run_topic_reviews_from_args

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(description="Run automated topic reviews for a batch.")
    add_topic_review_run_cli_arguments(parser)
    report = run_topic_reviews_from_args(parser.parse_args())
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
