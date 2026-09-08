"""
generate_dataset.py — builds a synthetic, labeled dataset of contract clauses
rated Red / Amber / Green for risk, using templates with controlled risk
indicators (not an LLM). Labels are known-correct by construction, since each
template's risk level is fixed by its wording, not judged after the fact.

Output: data/clauses.jsonl, one {"text": ..., "rating": ..., "category": ...} per line.
Also writes data/train.jsonl / val.jsonl / test.jsonl (80/10/10 stratified split).
"""
import json
import random
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Filler values used to create surface variation within each template.
DAYS = [5, 7, 10, 14, 15, 20, 30, 45, 60, 90]
PERCENTS = [1, 2, 5, 10, 15, 20, 25]
AMOUNTS = ["10,000", "25,000", "50,000", "100,000", "250,000", "500,000", "1,000,000"]
CURRENCIES = ["USD", "GBP", "EUR"]
PARTIES_A = ["the Seller", "the Company", "the Contractor", "the Vendor", "the Licensor"]
PARTIES_B = ["the Buyer", "the Client", "the Counterparty", "the Customer", "the Licensee"]
JURISDICTIONS = ["England and Wales", "the State of New York", "Singapore", "Germany", "Delaware"]

# ---------------------------------------------------------------------------
# Each category has a Red, Amber, and Green template. {a}/{b} = party names,
# other placeholders are filled from the lists above for variety.
# ---------------------------------------------------------------------------
TEMPLATES = {
    "liability": {
        "Red": "{a} shall have no liability whatsoever to {b} for any loss, damage, "
               "or claim arising under or in connection with this Agreement, including "
               "losses caused by {a}'s own negligence, gross negligence, or wilful misconduct.",
        "Amber": "{a}'s total aggregate liability under this Agreement shall not exceed "
                 "{currency} {amount}, save that this limitation shall not apply in cases "
                 "of fraud.",
        "Green": "Neither party's liability under this Agreement shall be limited or "
                 "excluded in respect of death or personal injury caused by negligence, "
                 "fraud, or any other liability which cannot be excluded by law.",
    },
    "termination": {
        "Red": "{a} may terminate this Agreement at any time, for any reason or no reason, "
               "upon written notice to {b}, with no liability for compensation, refund, or "
               "damages of any kind arising from such termination.",
        "Amber": "Either party may terminate this Agreement upon {days} days' written notice "
                  "if the other party commits a material breach that remains uncured for "
                  "{days2} days following notice of such breach.",
        "Green": "Either party may terminate this Agreement upon {days} days' prior written "
                 "notice to the other party, without cause, provided that all outstanding "
                 "obligations accrued prior to termination shall survive.",
    },
    "force_majeure": {
        "Red": "{a} shall be excused from performance of any obligation under this Agreement "
               "for any reason {a} deems to constitute force majeure, in {a}'s sole and "
               "absolute discretion, for an unlimited period.",
        "Amber": "A party affected by force majeure shall notify the other party within "
                  "{days} days and shall use reasonable efforts to mitigate the impact; if "
                  "the event continues for more than {days2} days, either party may terminate.",
        "Green": "Force majeure means an event beyond a party's reasonable control, including "
                 "acts of God, war, or natural disaster, which could not have been prevented "
                 "by reasonable precautions; the affected party must notify the other within "
                 "{days} days and resume performance as soon as reasonably possible.",
    },
    "payment": {
        "Red": "{b} shall pay all invoiced amounts within {days} days of the invoice date; "
               "{a} reserves the right to change prices at any time without notice, and "
               "{b} shall have no right to dispute or withhold payment for any reason.",
        "Amber": "Invoices are due within {days} days of receipt. Amounts not paid by the "
                  "due date shall accrue interest at {percent}% per annum, and {b} may "
                  "withhold payment only for amounts genuinely disputed in good faith.",
        "Green": "{b} shall pay undisputed invoiced amounts within {days} days of receipt. "
                 "Either party may raise a good-faith dispute over any invoice within "
                 "{days2} days, during which the disputed portion only may be withheld "
                 "pending resolution.",
    },
    "confidentiality": {
        "Red": "{b} shall keep all information disclosed by {a} confidential in perpetuity, "
               "with no exceptions, including information that is independently developed, "
               "publicly available, or already known to {b}.",
        "Amber": "Each party shall keep the other's confidential information confidential "
                  "for a period of {days2} years following disclosure, except where "
                  "disclosure is required by law or regulatory authority.",
        "Green": "Confidential Information excludes information that is or becomes publicly "
                 "available through no fault of the receiving party, was already known to "
                 "the receiving party, or is independently developed without reference to "
                 "the disclosing party's information. Obligations survive for {days2} years.",
    },
    "indemnification": {
        "Red": "{b} shall indemnify, defend, and hold harmless {a} from and against any and "
               "all claims, losses, and damages of any kind whatsoever, including those "
               "arising from {a}'s own negligence, gross negligence, or breach of this "
               "Agreement.",
        "Amber": "Each party shall indemnify the other against third-party claims arising "
                  "from its own breach of this Agreement, except to the extent such claims "
                  "arise from the indemnified party's negligence or wilful misconduct.",
        "Green": "Each party shall indemnify the other against direct third-party claims "
                 "caused by its own breach of this Agreement, gross negligence, or wilful "
                 "misconduct, provided that the indemnifying party is promptly notified and "
                 "given control of the defense.",
    },
    "intellectual_property": {
        "Red": "All intellectual property created by {b} in connection with this Agreement, "
               "including pre-existing IP incorporated into any deliverable, shall "
               "automatically and irrevocably vest in {a} at no additional cost.",
        "Amber": "IP created specifically for {a} under this Agreement shall vest in {a} "
                  "upon full payment; {b} retains ownership of its pre-existing IP and any "
                  "general know-how used in performing the Agreement.",
        "Green": "Each party retains ownership of its pre-existing intellectual property. "
                 "IP created jointly under this Agreement shall be jointly owned, with each "
                 "party granted a royalty-free license to use it for its own business "
                 "purposes.",
    },
    "warranty": {
        "Red": "{a} makes no warranties of any kind, express or implied, including as to "
               "merchantability, fitness for purpose, or non-infringement, and disclaims "
               "all such warranties to the fullest extent permitted, including those that "
               "cannot ordinarily be excluded.",
        "Amber": "{a} warrants that the goods will conform to the agreed specification for "
                  "a period of {days2} months from delivery, and that this is the sole "
                  "warranty given in place of all other warranties, express or implied.",
        "Green": "{a} warrants that the goods will conform to the agreed specification and "
                 "be free from material defects for {days2} months from delivery. This "
                 "warranty is in addition to, and does not limit, any statutory rights "
                 "{b} may have.",
    },
    "dispute_resolution": {
        "Red": "Any dispute arising under this Agreement shall be resolved exclusively by "
               "{a}, whose decision shall be final, binding, and not subject to appeal, "
               "mediation, or any form of judicial review.",
        "Amber": "Any dispute shall first be escalated to senior management of both parties "
                  "for good-faith negotiation for {days} days, failing which either party "
                  "may commence binding arbitration under the rules of the ICC.",
        "Green": "Any dispute shall first be subject to good-faith negotiation between the "
                 "parties for {days} days. If unresolved, either party may refer the dispute "
                 "to mediation, and if mediation fails, to binding arbitration seated in "
                 "{jurisdiction}.",
    },
    "assignment": {
        "Red": "{a} may assign, novate, or transfer this Agreement or any of its rights and "
               "obligations to any third party at any time without notice to or consent "
               "from {b}, who shall remain bound to the assignee on the same terms.",
        "Amber": "Neither party may assign this Agreement without the prior written consent "
                  "of the other party, such consent not to be unreasonably withheld, except "
                  "that {a} may assign to an affiliate or in connection with a merger or "
                  "sale of substantially all its assets.",
        "Green": "Neither party may assign or transfer this Agreement, in whole or in part, "
                 "without the prior written consent of the other party, which shall not be "
                 "unreasonably withheld or delayed.",
    },
    "governing_law": {
        "Red": "This Agreement shall be governed by such laws and subject to the jurisdiction "
               "of such courts as {a} may unilaterally designate from time to time by notice "
               "to {b}, without {b}'s further consent.",
        "Amber": "This Agreement shall be governed by the laws of {jurisdiction}, and the "
                  "parties submit to the non-exclusive jurisdiction of the courts of that "
                  "jurisdiction.",
        "Green": "This Agreement shall be governed by the laws of {jurisdiction}. Any "
                 "proceedings shall be brought in the courts of {jurisdiction}, to whose "
                 "exclusive jurisdiction each party irrevocably submits, save for "
                 "applications for interim injunctive relief.",
    },
    "data_protection": {
        "Red": "{b} grants {a} an unrestricted, perpetual right to use, sell, and share any "
               "personal data obtained in connection with this Agreement for any purpose, "
               "without further consent or notice to the data subjects.",
        "Amber": "Each party shall comply with applicable data protection law when processing "
                  "personal data under this Agreement and shall implement reasonable "
                  "technical and organizational security measures.",
        "Green": "Each party shall comply with applicable data protection law, process "
                 "personal data only as necessary to perform this Agreement, implement "
                 "appropriate technical and organizational security measures, and notify "
                 "the other party without undue delay of any personal data breach.",
    },
}

CATEGORIES = list(TEMPLATES.keys())
RATINGS = ["Red", "Amber", "Green"]
VARIATIONS_PER_TEMPLATE = 30


def fill_template(template: str) -> str:
    a, b = random.sample(PARTIES_A, 1)[0], random.sample(PARTIES_B, 1)[0]
    days_choices = random.sample(DAYS, 2)
    return template.format(
        a=a,
        b=b,
        days=days_choices[0],
        days2=days_choices[1],
        percent=random.choice(PERCENTS),
        amount=random.choice(AMOUNTS),
        currency=random.choice(CURRENCIES),
        jurisdiction=random.choice(JURISDICTIONS),
    )


def build_dataset() -> list[dict]:
    rows = []
    for category, ratings in TEMPLATES.items():
        for rating, template in ratings.items():
            seen = set()
            attempts = 0
            while len(seen) < VARIATIONS_PER_TEMPLATE and attempts < VARIATIONS_PER_TEMPLATE * 4:
                text = fill_template(template)
                attempts += 1
                if text in seen:
                    continue
                seen.add(text)
                rows.append({"text": text, "rating": rating, "category": category})
    random.shuffle(rows)
    return rows


def stratified_split(rows: list[dict], train_frac=0.8, val_frac=0.1):
    by_key: dict[tuple, list[dict]] = {}
    for r in rows:
        by_key.setdefault((r["category"], r["rating"]), []).append(r)

    train, val, test = [], [], []
    for group in by_key.values():
        random.shuffle(group)
        n = len(group)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train += group[:n_train]
        val += group[n_train:n_train + n_val]
        test += group[n_train + n_val:]

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


def write_jsonl(path: Path, rows: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    rows = build_dataset()
    write_jsonl(DATA_DIR / "clauses.jsonl", rows)

    train, val, test = stratified_split(rows)
    write_jsonl(DATA_DIR / "train.jsonl", train)
    write_jsonl(DATA_DIR / "val.jsonl", val)
    write_jsonl(DATA_DIR / "test.jsonl", test)

    print(f"Total: {len(rows)}  |  Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
    from collections import Counter
    print("Rating distribution:", Counter(r["rating"] for r in rows))
    print("Category distribution:", Counter(r["category"] for r in rows))
