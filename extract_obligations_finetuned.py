"""
extract_obligations_finetuned.py

Extracts obligation-bearing clauses from a contract text file using a
LEGAL-BERT classifier fine-tuned on the Atticus Open Contract Dataset
(see finetune_legal_bert_obligations.py). This replaces the zero-shot
similarity approach with a real trained classification head, and should
be noticeably more accurate.

`extract_obligations()` returns a list of (categories, clause, confidence)
tuples, restricted to four categories: payment obligations, exclusivity /
restrictive covenants, notice periods, and renewal mechanics. `categories`
is a list, since a clause can match more than one (multi-label). The
classifier only distinguishes obligation vs. non-obligation, so category
membership and labeling is determined by a keyword/phrase filter (see
CATEGORY_PATTERNS and get_clause_categories() below) applied on top of the
classifier's output.

USAGE:
    python extract_obligations_finetuned.py contract.txt --model ./obligation-classifier
    python extract_obligations_finetuned.py contract.txt --model ./obligation-classifier --threshold 0.6 --out results.json
"""

import argparse
import json
import re
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# The fine-tuned classifier only knows "obligation vs. not" -- it has no
# concept of *which kind* of obligation a clause is. To narrow results down
# to specific categories, we apply keyword/phrase matching on top of the
# classifier's output. A clause must pass the classifier's obligation
# threshold AND match at least one of these patterns to be returned.
#
# Adjust/extend these patterns if you find real clauses in your contracts
# that use phrasing not covered here.
CATEGORY_PATTERNS = {
    "payment_obligations": re.compile(
        r"\b(pay|payment|invoice|fee|compensation|reimburse|remit|"
        r"consideration|amount due|interest|penalty|royalt|installment)\w*\b",
        re.IGNORECASE,
    ),
    "exclusivity_and_restrictive_covenants": re.compile(
        r"\b(exclusiv|non-?compet|restrictive covenant|solicit|"
        r"covenant not to|shall not compete|non-?disparag)\w*\b",
        re.IGNORECASE,
    ),
    "notice_periods": re.compile(
        r"\b(notice|notify|notification)\w*\b",
        re.IGNORECASE,
    ),
    "renewal_mechanics": re.compile(
        r"\b(renew|automatically renew|evergreen|extend(ed|s)? the term|"
        r"term shall extend)\w*\b",
        re.IGNORECASE,
    ),
}

# For notice_periods specifically: only count a clause as a notice-period
# clause if it names a concrete duration/deadline (e.g. "30 days", "ten (10)
# business days", "6 months"). Matches a number (digit or spelled out,
# optionally followed by a parenthetical digit form like "thirty (30)")
# immediately followed by a time unit.
SPECIFIC_DURATION_PATTERN = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"fifteen|twenty|thirty|forty|forty-five|sixty|ninety)\s*"
    r"(\(\d+\)\s*)?"
    r"(calendar\s+|business\s+)?"
    r"(day|days|week|weeks|month|months|year|years)\b",
    re.IGNORECASE,
)

# Vague timing language that should NOT count as a specific deadline, even
# though it often appears right next to the word "notice". Kept mainly for
# documentation/clarity -- a clause using only these phrases will already
# fail SPECIFIC_DURATION_PATTERN above, so it's excluded either way.
VAGUE_NOTICE_TERMS = re.compile(
    r"\b(immediately|promptly|as soon as (reasonably )?possible|"
    r"reasonable (period|time|amount of time)|reasonably promptly|"
    r"without (undue )?delay|forthwith)\b",
    re.IGNORECASE,
)


def has_specific_notice_duration(clause):
    """
    True if the clause names a concrete time duration/deadline (e.g. "within
    30 days", "ten (10) business days"). Vague timing language like
    "immediately" or "within a reasonable amount of time" does not count,
    since it has no number+unit for SPECIFIC_DURATION_PATTERN to match.
    """
    return bool(SPECIFIC_DURATION_PATTERN.search(clause))


def get_clause_categories(clause):
    """
    Returns a list of every category this clause matches (multi-label),
    in the fixed order categories are defined in CATEGORY_PATTERNS above
    (payment -> exclusivity/restrictive covenants -> notice -> renewal).
    Returns an empty list if the clause doesn't match any target category.

    Special case for "notice_periods": a clause only gets this label if it
    also names a specific duration/deadline (see has_specific_notice_duration).
    A clause that mentions "notice" but only uses vague timing language
    ("immediately", "promptly", "within a reasonable amount of time") will
    NOT be labeled notice_periods, even though the word "notice" matched.
    """
    categories = []
    for category_name, pattern in CATEGORY_PATTERNS.items():
        if not pattern.search(clause):
            continue
        if category_name == "notice_periods" and not has_specific_notice_duration(clause):
            continue
        categories.append(category_name)
    return categories


def get_category(clause):
    """
    Returns the single best-matching category name for a clause, or None if
    no category matches. If a clause matches more than one category (e.g. a
    clause about both payment and renewal), the category with the most
    keyword hits wins; ties are broken by the order CATEGORY_PATTERNS is
    defined in (payment_obligations first, etc).
    """
    best_category, best_count = None, 0
    for category, pattern in CATEGORY_PATTERNS.items():
        count = len(pattern.findall(clause))
        if count > best_count:
            best_category, best_count = category, count
    return best_category


def split_into_clauses(text):
    """Same clause-splitting logic as the zero-shot version, for consistency."""
    text = re.sub(r"\s+", " ", text).strip()
    raw_sentences = re.split(r"(?<=[.;])\s+(?=[A-Z0-9(])", text)
    return [s.strip() for s in raw_sentences if len(s.strip().split()) >= 5]


def classify_clauses(clauses, tokenizer, model, batch_size=16):
    """
    Returns the model's predicted probability that each clause is an
    obligation (label 1), in the same order as `clauses`.
    """
    probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(clauses), batch_size):
            batch = clauses[i : i + batch_size]
            inputs = tokenizer(batch, truncation=True, padding=True, max_length=256, return_tensors="pt")
            logits = model(**inputs).logits
            batch_probs = torch.softmax(logits, dim=1)[:, 1]  # P(obligation)
            probs.extend(batch_probs.tolist())
    return probs


def extract_obligations(text, tokenizer, model, threshold=0):
    """
    Returns a list of (categories, clause, confidence) tuples for clauses that:
      1. The fine-tuned classifier flags as an obligation above `threshold`, AND
      2. Match at least one of the target categories: payment obligations,
         exclusivity/restrictive covenants, notice periods, or renewal mechanics.

    `categories` is a list containing every matching category name (a clause
    can appear under more than one, e.g. a renewal clause that also requires
    written notice will have categories == ["notice_periods", "renewal_mechanics"]
    -- see get_clause_categories() for the matching order/logic.
    """
    clauses = split_into_clauses(text)
    if not clauses:
        return []

    probs = classify_clauses(clauses, tokenizer, model)

    results = []
    for clause, prob in zip(clauses, probs):
        if prob < threshold:
            continue
        categories = get_clause_categories(clause)
        if categories:
            results.append((categories, clause, round(prob, 4)))

    results.sort(key=lambda triple: triple[2], reverse=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Extract obligation clauses using a fine-tuned LEGAL-BERT classifier.")
    parser.add_argument("input_file", help="Path to a .txt file containing the contract text.")
    parser.add_argument("--model", required=True, help="Path to the fine-tuned model directory (from finetune_legal_bert_obligations.py).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold (0-1) for flagging a clause. Default 0.5.")
    parser.add_argument("--out", help="Optional path to write results as JSON.")
    args = parser.parse_args()

    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading fine-tuned model from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)

    print("Extracting obligation clauses...")
    results = extract_obligations(text, tokenizer, model, threshold=args.threshold)

    if not results:
        print("No obligation clauses found above the confidence threshold.")
        return

    print(f"\nFound {len(results)} candidate obligation clause(s):\n")
    for i, (categories, clause, confidence) in enumerate(results, 1):
        print(f"[{i}] categories: {', '.join(categories)} | confidence: {confidence}")
        print(f"    {clause}\n")

    if args.out:
        # JSON has no native tuple type, so each triple is written as a
        # 3-element array: [["payment_obligations"], "clause text", 0.87].
        # Reload with `[tuple(t) for t in json.load(f)]` if you want tuples back.
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
