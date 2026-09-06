#!/usr/bin/env python3
"""Convert contract-extraction JSON into an iCalendar contract-term event.

The JSON may contain fields such as:

    {
      "input_file": "contract.txt",
      "parties": [
        {"role": "Landlord", "name": "ABC Properties"},
        {"role": "Tenant", "name": "XYZ Corp"}
      ],
      "effective_date": "2026-07-01",
      "expiration_date": "2029-07-01"
    }

The current parties-only ``answer_contract.py`` does not produce dates.  For
that output, provide ``--effective_date`` and ``--expiration_date`` when
running this converter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any


MONTHS = {
    name.lower(): number
    for number, name in enumerate(
        (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ),
        start=1,
    )
}


def first_value(payload: dict[str, Any], key: str) -> Any:
    """Read a field from the top level or a nested ``fields`` object."""

    if key in payload:
        return payload[key]
    fields = payload.get("fields")
    if isinstance(fields, dict):
        return fields.get(key)
    return None


def scalar_value(value: Any) -> str | None:
    """Extract a string from common model-output field shapes."""

    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        for key in ("normalized", "date", "value", "text", "answer"):
            if key in value:
                result = scalar_value(value[key])
                if result:
                    return result
    return str(value).strip() or None


def parse_date(value: Any) -> date | None:
    """Parse ISO, US numeric, and common written date formats."""

    text = scalar_value(value)
    if not text:
        return None

    iso_match = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text)
    if iso_match:
        return date(
            int(iso_match.group(1)),
            int(iso_match.group(2)),
            int(iso_match.group(3)),
        )

    numeric_match = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", text)
    if numeric_match:
        return date(
            int(numeric_match.group(3)),
            int(numeric_match.group(1)),
            int(numeric_match.group(2)),
        )

    written_match = re.search(
        r"\b([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?[,]?\s+(\d{4})\b",
        text,
    )
    if written_match:
        month = MONTHS.get(written_match.group(1).lower())
        if month:
            return date(
                int(written_match.group(3)),
                month,
                int(written_match.group(2)),
            )

    return None


def extract_term(value: Any) -> tuple[float, str] | None:
    """Read a duration from a structured or textual term field."""

    if isinstance(value, dict):
        amount = value.get("amount")
        unit = value.get("unit")
        if amount is not None and unit:
            try:
                return float(amount), str(unit).lower()
            except (TypeError, ValueError):
                return None

    text = scalar_value(value)
    if not text:
        return None
    match = re.search(
        r"\b(\d+(?:\.\d+)?)\s*(day|days|week|weeks|month|months|year|years)\b",
        text.lower(),
    )
    if not match:
        return None
    return float(match.group(1)), match.group(2)


def add_term(start: date, amount: float, unit: str) -> date:
    """Calculate a term end date from a start date and duration."""

    normalized_unit = unit.lower().rstrip("s")
    if normalized_unit == "day":
        return start + timedelta(days=amount)
    if normalized_unit == "week":
        return start + timedelta(days=amount * 7)
    if normalized_unit == "month":
        whole_months = int(amount)
        total_months = start.year * 12 + (start.month - 1) + whole_months
        year = total_months // 12
        month = total_months % 12 + 1
        # Clamp dates such as January 31 + 1 month to February's last day.
        next_month = date(year + (month == 12), 1 if month == 12 else month + 1, 1)
        last_day = (next_month - timedelta(days=1)).day
        return date(year, month, min(start.day, last_day))
    if normalized_unit == "year":
        whole_years = int(amount)
        try:
            return start.replace(year=start.year + whole_years)
        except ValueError:
            # February 29 in a leap year becomes February 28.
            return start.replace(year=start.year + whole_years, day=28)
    raise ValueError(f"Unsupported term unit: {unit}")


def choose_expiration_date(
    payload: dict[str, Any],
    effective_date: date,
    extracted_expiration: date | None,
) -> tuple[date | None, bool]:
    """Prefer a valid extracted end date, otherwise compute it from the term.

    A model can incorrectly label the effective date as the expiration date.
    When that happens, a positive structured term is a safer source for the
    calendar event's end date.
    """

    term = extract_term(first_value(payload, "term"))
    if (
        extracted_expiration is not None
        and extracted_expiration > effective_date
    ):
        return extracted_expiration, False

    if term and term[0] > 0:
        return add_term(effective_date, term[0], term[1]), True

    return extracted_expiration, False


def extract_parties(payload: dict[str, Any]) -> list[str]:
    parties = payload.get("parties", [])
    if not isinstance(parties, list) or not parties:
        fields = payload.get("fields")
        if isinstance(fields, dict):
            parties = fields.get("parties", [])
    if not isinstance(parties, list):
        return []

    result: list[str] = []
    seen: set[str] = set()
    for party in parties:
        if isinstance(party, dict):
            name = str(party.get("name", "")).strip()
            role = str(party.get("role", "")).strip()
            display = f"{role}: {name}" if role and name else name or role
        else:
            display = str(party).strip()
        if display and display.casefold() not in seen:
            result.append(display)
            seen.add(display.casefold())
    return result


def ical_escape(value: str) -> str:
    """Escape text according to RFC 5545."""

    return (
        value.replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\r\n", "\\n")
        .replace("\n", "\\n")
        .replace("\r", "\\n")
    )


def fold_ical_line(line: str, width: int = 75) -> list[str]:
    """Fold long content lines as required by iCalendar."""

    if len(line) <= width:
        return [line]
    lines = [line[:width]]
    remaining = line[width:]
    while remaining:
        lines.append(" " + remaining[: width - 1])
        remaining = remaining[width - 1 :]
    return lines


def make_uid(input_path: str, start: date, end: date) -> str:
    seed = f"{Path(input_path).resolve()}|{start.isoformat()}|{end.isoformat()}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]
    return f"{digest}@contract-extractor"


def build_ics(
    payload: dict[str, Any],
    input_path: str,
    effective_date: date,
    expiration_date: date,
) -> str:
    if expiration_date < effective_date:
        raise ValueError("expiration_date cannot be earlier than effective_date")

    parties = extract_parties(payload)
    contract_name = str(payload.get("input_file") or input_path)
    summary = "Contract term"
    if parties:
        summary += " - " + " / ".join(parties)

    description_lines = [f"Contract file: {contract_name}"]
    if parties:
        description_lines.append("Parties: " + "; ".join(parties))
    description_lines.append(f"Effective date: {effective_date.isoformat()}")
    description_lines.append(f"Expiration date: {expiration_date.isoformat()}")

    # For an all-day event, DTEND is exclusive, so include the expiration date
    # by setting DTEND to the following day.
    event_end = expiration_date + timedelta(days=1)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    uid = make_uid(input_path, effective_date, expiration_date)

    raw_lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Contract Extractor//Contract Term//EN",
        "CALSCALE:GREGORIAN",
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{stamp}",
        f"DTSTART;VALUE=DATE:{effective_date.strftime('%Y%m%d')}",
        f"DTEND;VALUE=DATE:{event_end.strftime('%Y%m%d')}",
        f"SUMMARY:{ical_escape(summary)}",
        f"DESCRIPTION:{ical_escape(chr(10).join(description_lines))}",
        "END:VEVENT",
        "END:VCALENDAR",
    ]

    folded_lines: list[str] = []
    for line in raw_lines:
        folded_lines.extend(fold_ical_line(line))
    return "\r\n".join(folded_lines) + "\r\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert contract JSON into an all-day ICS term event."
    )
    parser.add_argument("--input_json", required=True, help="Contract JSON file")
    parser.add_argument(
        "--output_ics",
        default="contract_term.ics",
        help="Output ICS path (default: contract_term.ics)",
    )
    parser.add_argument(
        "--effective_date",
        help="Override the effective date, for example 2026-07-01",
    )
    parser.add_argument(
        "--expiration_date",
        help="Override the expiration date, for example 2029-07-01",
    )
    args = parser.parse_args()

    input_path = Path(args.input_json)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON file does not exist: {input_path}")

    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("The input JSON must contain an object at the top level")

    effective_date = parse_date(
        args.effective_date or first_value(payload, "effective_date")
    )
    extracted_expiration = parse_date(
        args.expiration_date or first_value(payload, "expiration_date")
    )
    if effective_date is None:
        raise ValueError(
            "An effective_date is required. Provide --effective_date or use "
            "JSON that contains an effective_date field."
        )

    expiration_date, was_computed = choose_expiration_date(
        payload, effective_date, extracted_expiration
    )
    if expiration_date is None:
        raise ValueError(
            "Could not determine expiration_date from the JSON or the term. "
            "Provide --expiration_date manually."
        )

    ics = build_ics(payload, str(input_path), effective_date, expiration_date)
    output_path = Path(args.output_ics)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(ics, encoding="utf-8", newline="")
    if was_computed:
        print(
            "Expiration date was computed from the contract term: "
            f"{expiration_date.isoformat()}"
        )
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
