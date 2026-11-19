"""Resolve the eight manually audited incident examples from the pinned HERB files."""

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REVISION = "a00bca08f9118e482e6de9951fdcb654fbed5343"
SELECTION = (
    ("ActionGenie", 41, "20261230-5-473fe", "20270118-0-0b5c5", "CUST-0088",
     "UI freezing during bulk action suggestions",
     "Synchronous processing blocked the UI thread.",
     "Process suggestions asynchronously and update the UI with a callback."),
    ("ActionGenie", 49, "20270203-1-a41f0", "20270228-0-7dfc3", "CUST-0088",
     "delayed task creation in high-traffic channels",
     "Inefficient prioritization overloaded the backend task queue under high traffic.",
     "Improve task prioritization and backend queue management."),
    ("ActionGenie", 32, "20261117-1-a77c3", "20261123-0-80236", "CUST-0047",
     "duplicate log entries for manual actions",
     "Network retries could trigger the manual-action logger repeatedly.",
     "Log each action once and verify entry uniqueness before logging."),
    ("AnomalyForce", 29, "20260906-3-40719", "20260911-0-64110", "CUST-0025",
     "gradual time-series drift generating false anomaly alerts",
     "The detector was too sensitive to gradual trends and confused them with sudden anomalies.",
     "Improve temporal smoothing and adjust detector sensitivity."),
    ("AnomalyForce", 33, "20260925-5-5564e", "20261003-0-06ef8", "CUST-0058",
     "inconsistent refresh rates across dashboard data sources",
     "Asynchronous data fetching was not synchronized across streams.",
     "Synchronize fetching and coordinate update intervals with a central refresh controller."),
    ("AnomalyForce", 27, "20260830-3-3b439", "20260902-0-e1407", "CUST-0059",
     "incorrect data aggregation across time zones",
     "Timestamps were not normalized before aggregation.",
     "Normalize timestamps before aggregation."),
    ("CollaborationForce", 24, "20261110-1-d2c0f", "20261119-0-28c95", "CUST-0061",
     "slow document retrieval caused by inefficient query plans",
     "Outdated database statistics and missing indexes led to inefficient plans and full table scans.",
     "Update the database statistics and add the missing indexes."),
    ("CollaborationForce", 28, "20261121-3-af6ce", "20261209-0-15934", "CUST-0032",
     "conversation threads appearing out of chronological order",
     "Timestamp parsing mishandled time zones.",
     "Correct time-zone handling in the timestamp parser so chronological sorting works."),
)


def prepare(source_dir):
    files = {}
    samples = []
    for i, (product, number, report_id, update_id, customer, incident, cause, fix) in enumerate(SELECTION, 1):
        if product not in files:
            data = (source_dir / f"{product}.json").read_bytes()
            files[product] = (json.loads(data), hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest())
        doc, digest = files[product]
        messages = {x["Message"]["User"]["utterranceID"]: x for x in doc["slack"]}
        report, update = (messages[x] for x in (report_id, update_id))
        assert report["Channel"]["channelID"] == update["Channel"]["channelID"]
        report_text, update_text = (x["Message"]["User"]["text"].strip() for x in (report, update))
        link = f"https://github.com/salesforce/{product}/pull/{number}"
        pr = next(x for x in doc["prs"] if x["link"].rstrip("/") == link)
        assert customer in report_text and link + "." in update_text
        # Keep the issue and changes. Omit test details following those sections.
        pr_excerpt = re.split(r"\n#{2,4}\s+(?:Test|How to Test|Verification)", pr["summary"], maxsplit=1,
                              flags=re.IGNORECASE)[0].strip()
        assert customer not in pr_excerpt
        assert len(pr_excerpt) > 100
        ids = [report_id, update_id, pr["id"]]
        question = (f"For the {product} incident involving {incident}, which customer ID was affected, "
                    "what was the root cause, and what did the fix change?")
        samples.append(dict(
            id=f"herb_{i:02}", product=product, product_git_blob=digest, question=question,
            evidence=[dict(source="slack", text=report_text + "\n\n" + update_text),
                      dict(source="github_pr", text=link + "\n\n" + pr_excerpt)],
            ground_truth=f"{customer}. {cause} {fix}", required_facts=dict(customer=customer, cause=cause, fix=fix),
            citations=ids, official_qa_citing_all_selected_records=[
                j for j, q in enumerate(doc["answerable_questions"]) if set(ids) <= set(q["citations"])
            ],
        ))
    payload = dict(dataset="Salesforce/HERB", revision=REVISION,
                   selection="Handwritten incident QA using paired Slack issue/closure messages and a linked PR summary.",
                   samples=samples)
    (ROOT / "samples.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    prepare(parser.parse_args().source_dir)
