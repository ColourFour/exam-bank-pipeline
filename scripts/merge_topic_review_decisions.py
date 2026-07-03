from exam_bank.topic_review_loop import add_topic_review_merge_cli_arguments, merge_topic_review_decisions_from_args

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge topic-bank reviewed decision files.")
    add_topic_review_merge_cli_arguments(parser)
    report = merge_topic_review_decisions_from_args(parser.parse_args())
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
