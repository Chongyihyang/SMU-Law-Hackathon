#!/usr/bin/env python3
"""
Extract parties, effective dates, and expiration dates.

This script is the inference companion for train_legal_bert_contract_fields.py.
It combines:

    - LEGAL-BERT token classification for multiple spans;
    - explicit-label parsing for fields such as Landlord/Tenant;
    - date parsing and date arithmetic for implied expiration dates.

Example:

    python extract_contract_fields.py \
        --model_dir ./legal-bert-contract-fields \
        --input_txt ./contract.txt \
        --output_json ./contract_fields.json

Install python-dateutil for robust date arithmetic:

    pip install python-dateutil
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime
from pathlib import Path


MONTHS = (
    "January|February|March|April|May|June|July|August|"
    "September|October|November|December"
)

DATE_PATTERN = re.compile(
    rf"\b(?:"
    rf"(?:{MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,)?\s+\d{{4}}"
    rf"|\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}"
    rf")\b",
    flags=re.IGNORECASE,
)

PARTY_LABELS = [
    "Landlord",
    "Tenant",
    "Lessor",
    "Lessee",
    "Buyer",
    "Seller",
    "Licensor",
    "Licensee",
    "Employer",
    "Employee",
    "Client",
    "Contractor",
    "Vendor",
    "Customer",
]

SECTION_LABELS = PARTY_LABELS + [
    "Premises",
    "Term",
    "Rent",
    "Rent Increase",
    "Utilities",
    "Repairs",
    "Termination",
    "Effective Date",
    "Expiration Date",
    "Commencement Date",
    "Address",
    "Services",
    "Fees",
    "Payment",
    "Notice",
    "Governing Law",
]


def parse_date_text(value: str) -> date | None:
    """Parse common legal date formats."""

    cleaned = re.sub(
        r"\b(\d{1,2})(st|nd|rd|th)\b",
        r"\1",
        value,
        flags=re.IGNORECASE,
    )

    try:
        from dateutil import parser as date_parser

        return date_parser.parse(
            cleaned,
            fuzzy=False,
            default=datetime(1900, 1, 1),
        ).date()
    except Exception:
        pass

    formats = [
        "%B %d, %Y",
        "%B %d %Y",
        "%b %d, %Y",
        "%b %d %Y",
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%Y-%m-%d",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(cleaned.strip(), fmt).date()
        except ValueError:
            continue

    return None


def find_date_mentions(text: str) -> list[dict]:
    mentions = []

    for match in DATE_PATTERN.finditer(text):
        raw = match.group(0)
        parsed = parse_date_text(raw)

        if parsed is not None:
            mentions.append(
                {
                    "text": raw,
                    "start": match.start(),
                    "end": match.end(),
                    "date": parsed,
                }
            )

    return mentions


def extract_labeled_parties(context: str) -> list[dict]:
    party_pattern = "|".join(
        re.escape(label)
        for label in sorted(PARTY_LABELS, key=len, reverse=True)
    )
    section_pattern = "|".join(
        re.escape(label)
        for label in sorted(SECTION_LABELS, key=len, reverse=True)
    )

    pattern = re.compile(
        rf"\b(?P<role>{party_pattern})\s*:\s*"
        rf"(?P<name>.*?)(?=\s+\b(?:{section_pattern})\s*:|$)",
        flags=re.IGNORECASE,
    )

    parties = []

    for match in pattern.finditer(context):
        name = match.group("name").strip(" .;,\n\t")
        if not name:
            continue

        parties.append(
            {
                "role": match.group("role"),
                "name": name,
                "start": match.start("name"),
                "end": match.end("name"),
                "source": "label-parser",
            }
        )

    return parties


def extract_term(context: str) -> dict | None:
    patterns = [
        re.compile(
            r"\b(?:initial\s+)?term\s*:\s*([^\.\n]+)",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:initial\s+)?term\s+(?:is|shall be|of)\s+"
            r"([^\.\n]+)",
            flags=re.IGNORECASE,
        ),
    ]

    for pattern in patterns:
        match = pattern.search(context)
        if not match:
            continue

        clause = match.group(1).strip()
        duration = re.search(
            r"\b(\d+(?:\.\d+)?)\s*"
            r"(year|years|month|months)\b",
            clause,
            flags=re.IGNORECASE,
        )

        result = {
            "text": clause,
            "start": match.start(1),
            "end": match.end(1),
            "source": "term-parser",
        }

        if duration:
            result["amount"] = float(duration.group(1))
            result["unit"] = duration.group(2).lower()

        return result

    return None


def find_contextual_date(
    context: str,
    mentions: list[dict],
    keywords: list[str],
) -> dict | None:
    keyword_pattern = re.compile(
        r"(?:" + "|".join(keywords) + r")",
        flags=re.IGNORECASE,
    )

    for mention in mentions:
        before = context[max(0, mention["start"] - 100):mention["start"]]
        after = context[mention["end"]:mention["end"] + 40]

        if keyword_pattern.search(before) or keyword_pattern.search(after):
            return {
                **mention,
                "source": "date-context-parser",
            }

    return None


def add_duration(start_date: date, amount: float, unit: str) -> date:
    """Add years or months, using dateutil when available."""

    try:
        from dateutil.relativedelta import relativedelta

        if unit.startswith("year"):
            return start_date + relativedelta(years=amount)
        return start_date + relativedelta(months=amount)
    except Exception:
        if unit.startswith("year"):
            years = int(amount)
            try:
                return start_date.replace(year=start_date.year + years)
            except ValueError:
                return start_date.replace(
                    year=start_date.year + years,
                    day=28,
                )

        months = int(amount)
        year = start_date.year + (start_date.month - 1 + months) // 12
        month = (start_date.month - 1 + months) % 12 + 1
        day = min(start_date.day, 28)
        return date(year, month, day)


def date_result(mention: dict | None, source: str | None = None) -> dict | None:
    if mention is None:
        return None

    result = {
        "text": mention["text"],
        "normalized": mention["date"].isoformat(),
        "start": mention.get("start"),
        "end": mention.get("end"),
        "source": source or mention.get("source", "parser"),
        "computed": False,
    }
    return result


def load_model_labels(model_dir: Path):
    import torch
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        use_fast=True,
    )
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)
    model.eval()

    id2label = {}
    for key, value in model.config.id2label.items():
        id2label[int(key)] = value

    return tokenizer, model, device, id2label


def decode_model_spans(
    context: str,
    tokenizer,
    model,
    device,
    id2label: dict[int, str],
    max_length: int,
    doc_stride: int,
    min_probability: float,
) -> list[dict]:
    import torch

    encoded = tokenizer(
        context,
        truncation=True,
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    candidates = []

    for window_index in range(len(encoded["input_ids"])):
        model_inputs = {
            "input_ids": torch.tensor(
                [encoded["input_ids"][window_index]],
                dtype=torch.long,
                device=device,
            ),
            "attention_mask": torch.tensor(
                [encoded["attention_mask"][window_index]],
                dtype=torch.long,
                device=device,
            ),
        }

        if "token_type_ids" in encoded:
            model_inputs["token_type_ids"] = torch.tensor(
                [encoded["token_type_ids"][window_index]],
                dtype=torch.long,
                device=device,
            )

        with torch.inference_mode():
            logits = model(**model_inputs).logits[0]

        probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
        offsets = encoded["offset_mapping"][window_index]
        attention = encoded["attention_mask"][window_index]

        active_span = None

        def close_active():
            nonlocal active_span
            if active_span is not None:
                start = active_span["start"]
                end = active_span["end"]
                active_span["text"] = context[start:end]
                active_span["score"] = (
                    active_span["score_sum"] / active_span["count"]
                )
                del active_span["score_sum"]
                del active_span["count"]
                candidates.append(active_span)
                active_span = None

        for token_index, token_offset in enumerate(offsets):
            start, end = token_offset

            if attention[token_index] == 0 or end <= start:
                close_active()
                continue

            label_id = max(
                range(len(probabilities[token_index])),
                key=lambda index: probabilities[token_index][index],
            )
            label = str(id2label.get(label_id, "O"))
            probability = probabilities[token_index][label_id]

            if label == "O" or probability < min_probability:
                close_active()
                continue

            if "-" not in label:
                close_active()
                continue

            prefix, category = label.split("-", 1)

            if prefix == "B" or active_span is None:
                close_active()
                active_span = {
                    "category": category,
                    "start": start,
                    "end": end,
                    "score_sum": probability,
                    "count": 1,
                    "source": "token-classification-model",
                }
                continue

            if (
                prefix == "I"
                and active_span["category"] == category
            ):
                active_span["end"] = end
                active_span["score_sum"] += probability
                active_span["count"] += 1
            else:
                close_active()

        close_active()

    # Deduplicate spans produced by overlapping sliding windows.
    unique = {}
    for candidate in candidates:
        key = (
            candidate["category"],
            candidate["start"],
            candidate["end"],
        )
        if (
            key not in unique
            or candidate["score"] > unique[key]["score"]
        ):
            unique[key] = candidate

    candidates = list(unique.values())

    # Remove overlapping lower-scoring spans within one category.
    result = []
    for category in sorted({item["category"] for item in candidates}):
        category_candidates = sorted(
            [item for item in candidates if item["category"] == category],
            key=lambda item: item["score"],
            reverse=True,
        )
        kept = []

        for candidate in category_candidates:
            overlaps = any(
                candidate["start"] < other["end"]
                and other["start"] < candidate["end"]
                for other in kept
            )
            if not overlaps:
                kept.append(candidate)

        result.extend(sorted(kept, key=lambda item: item["start"]))

    return result


def parse_fields(context: str, model_spans: list[dict]) -> dict:
    parties = extract_labeled_parties(context)

    if not parties:
        parties = [
            {
                "name": span["text"],
                "start": span["start"],
                "end": span["end"],
                "source": span["source"],
            }
            for span in model_spans
            if span["category"] == "PARTIES"
        ]

    mentions = find_date_mentions(context)
    term = extract_term(context)

    effective = find_contextual_date(
        context,
        mentions,
        [
            r"effective",
            r"commenc",
            r"start",
            r"begin",
            r"as of",
            r"execution",
        ],
    )

    expiration = find_contextual_date(
        context,
        mentions,
        [
            r"expir",
            r"expir",
            r"end",
            r"until",
            r"through",
        ],
    )

    # Prefer dates explicitly located by the token-classification model when
    # they can be parsed. Otherwise use the contextual date parser.
    for span in model_spans:
        if span["category"] not in {
            "EFFECTIVE_DATE",
            "EXPIRATION_DATE",
        }:
            continue

        parsed = find_date_mentions(span["text"])
        if not parsed:
            continue

        mention = parsed[0].copy()
        mention["start"] += span["start"]
        mention["end"] += span["start"]
        mention["source"] = "token-classification-model"

        if span["category"] == "EFFECTIVE_DATE":
            effective = mention
        else:
            expiration = mention

    effective_result = date_result(effective)
    expiration_result = date_result(expiration)

    if expiration_result is None and effective is not None and term:
        if "amount" in term:
            computed = add_duration(
                effective["date"],
                term["amount"],
                term["unit"],
            )
            expiration_result = {
                "text": computed.isoformat(),
                "normalized": computed.isoformat(),
                "start": None,
                "end": None,
                "source": "computed-from-effective-date-and-term",
                "computed": True,
                "basis": {
                    "effective_date": effective["date"].isoformat(),
                    "amount": term["amount"],
                    "unit": term["unit"],
                },
            }

    return {
        "parties": parties,
        "effective_date": effective_result,
        "term": term,
        "expiration_date": expiration_result,
        "model_spans": model_spans,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default=None,
        help=(
            "Token-classification model directory. If omitted, the script "
            "uses deterministic label/date parsing only."
        ),
    )
    parser.add_argument("--input_txt", required=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--doc_stride", type=int, default=128)
    parser.add_argument("--min_probability", type=float, default=0.50)
    args = parser.parse_args()

    input_path = Path(args.input_txt)
    model_dir = Path(args.model_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if model_dir is not None and not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    context = input_path.read_text(encoding="utf-8")

    if model_dir is None:
        model_spans = []
        print("Using deterministic parser only.")
    else:
        tokenizer, model, device, id2label = load_model_labels(model_dir)
        print(f"Using device: {device}")

        if device.type == "cuda":
            import torch

            print(f"GPU: {torch.cuda.get_device_name(0)}")

        model_spans = decode_model_spans(
            context=context,
            tokenizer=tokenizer,
            model=model,
            device=device,
            id2label=id2label,
            max_length=args.max_length,
            doc_stride=args.doc_stride,
            min_probability=args.min_probability,
        )

    result = {
        "input_file": str(input_path),
        "fields": parse_fields(context, model_spans),
    }

    output_text = json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    )

    if args.output_json:
        Path(args.output_json).write_text(
            output_text + "\n",
            encoding="utf-8",
        )
        print(f"Results saved to: {args.output_json}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
