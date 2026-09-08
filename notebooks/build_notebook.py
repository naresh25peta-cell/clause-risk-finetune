"""
Builds notebooks/clause_risk_finetune.ipynb programmatically so the cell
source is easy to review/edit as plain Python strings instead of raw JSON.
Run this locally to (re)generate the .ipynb file; nothing here runs in Colab.
"""
import json
from pathlib import Path

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}

cells = []

cells.append(md("""\
# Clause Risk Fine-Tune: Does Fine-Tuning Beat Prompting?

This notebook fine-tunes a small open-source LLM (Qwen2.5-0.5B-Instruct) to classify
contract clauses as **Red / Amber / Green** risk, then compares it against the same
base model **zero-shot prompted** for the identical task.

Both approaches use the exact same model architecture and the exact same inference
cost, so the comparison isolates one variable: **does LoRA fine-tuning on ~450 labeled
examples improve accuracy over just prompting the base model?**

The task and label scheme come from a companion project, [clauseguard](https://github.com/naresh25peta-cell/clauseguard),
which prompts a much larger model to do this same rating. The dataset here is
synthetic and template-generated (see `generate_dataset.py`) so every label is
correct by construction — useful for isolating the fine-tuning effect without
label noise from an LLM-labeled dataset.

**Runtime:** Runtime > Change runtime type > T4 GPU (free tier is enough for a 0.5B model).
"""))

cells.append(md("## 1. Setup"))

cells.append(code("""\
!pip install -q transformers==4.46.2 peft==0.13.2 accelerate==1.1.1 datasets==3.1.0 bitsandbytes==0.43.1 scikit-learn
"""))

cells.append(code("""\
import json
import random
import time
import re
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

random.seed(42)
torch.manual_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)
if DEVICE == "cpu":
    print("WARNING: no GPU detected. Go to Runtime > Change runtime type > T4 GPU.")
"""))

cells.append(md("""\
## 2. Build the dataset

Same template-based generator as `generate_dataset.py` in the repo, inlined here so
this notebook runs standalone with no file uploads. Every clause's label is fixed by
which template produced it — not judged after the fact — so labels have zero noise.
"""))

cells.append(code("""\
DAYS = [5, 7, 10, 14, 15, 20, 30, 45, 60, 90]
PERCENTS = [1, 2, 5, 10, 15, 20, 25]
AMOUNTS = ["10,000", "25,000", "50,000", "100,000", "250,000", "500,000", "1,000,000"]
CURRENCIES = ["USD", "GBP", "EUR"]
PARTIES_A = ["the Seller", "the Company", "the Contractor", "the Vendor", "the Licensor"]
PARTIES_B = ["the Buyer", "the Client", "the Counterparty", "the Customer", "the Licensee"]
JURISDICTIONS = ["England and Wales", "the State of New York", "Singapore", "Germany", "Delaware"]

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

VARIATIONS_PER_TEMPLATE = 30

def fill_template(template):
    a, b = random.choice(PARTIES_A), random.choice(PARTIES_B)
    days_choices = random.sample(DAYS, 2)
    return template.format(
        a=a, b=b, days=days_choices[0], days2=days_choices[1],
        percent=random.choice(PERCENTS), amount=random.choice(AMOUNTS),
        currency=random.choice(CURRENCIES), jurisdiction=random.choice(JURISDICTIONS),
    )

def build_dataset():
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

def stratified_split(rows, train_frac=0.8, val_frac=0.1):
    by_key = {}
    for r in rows:
        by_key.setdefault((r["category"], r["rating"]), []).append(r)
    train, val, test = [], [], []
    for group in by_key.values():
        random.shuffle(group)
        n = len(group)
        n_train, n_val = int(n * train_frac), int(n * val_frac)
        train += group[:n_train]
        val += group[n_train:n_train + n_val]
        test += group[n_train + n_val:]
    random.shuffle(train); random.shuffle(val); random.shuffle(test)
    return train, val, test

rows = build_dataset()
train_rows, val_rows, test_rows = stratified_split(rows)
print(f"Total: {len(rows)}  Train: {len(train_rows)}  Val: {len(val_rows)}  Test: {len(test_rows)}")
"""))

cells.append(md("## 3. Load the base model"))

cells.append(code("""\
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
"""))

cells.append(md("""\
## 4. Baseline: zero-shot prompting

Ask the base model directly, no fine-tuning, using the same prompt style
`clauseguard`'s risk scanner uses. This is the "just prompt it" baseline.
"""))

cells.append(code("""\
PROMPT_TEMPLATE = (
    "You are a contract risk analyst. Rate the following clause as Red, Amber, or Green.\\n"
    "Red = significant risk, requires review. Amber = moderate risk or ambiguity. "
    "Green = standard, acceptable.\\n\\n"
    "Clause: {clause}\\n\\n"
    "Respond with exactly one word: Red, Amber, or Green."
)

def extract_rating(text):
    for rating in ("Red", "Amber", "Green"):
        if re.search(rating, text, re.IGNORECASE):
            return rating
    return "Amber"  # fallback if the model doesn't answer cleanly

def predict_zero_shot(model, clause, max_new_tokens=8):
    messages = [{"role": "user", "content": PROMPT_TEMPLATE.format(clause=clause)}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return extract_rating(generated)

def evaluate(model, rows, label=""):
    y_true, y_pred = [], []
    start = time.time()
    for row in rows:
        pred = predict_zero_shot(model, row["text"])
        y_true.append(row["rating"])
        y_pred.append(pred)
    elapsed = time.time() - start
    acc = accuracy_score(y_true, y_pred)
    print(f"[{label}] accuracy: {acc:.3f}  |  {elapsed:.1f}s for {len(rows)} examples "
          f"({elapsed/len(rows)*1000:.0f}ms/example)")
    print(classification_report(y_true, y_pred, labels=["Red", "Amber", "Green"], zero_division=0))
    return acc, elapsed, y_true, y_pred

baseline_acc, baseline_time, base_true, base_pred = evaluate(base_model, test_rows, label="Zero-shot baseline")
"""))

cells.append(md("## 5. LoRA fine-tune"))

cells.append(code("""\
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

ft_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)
ft_model = get_peft_model(ft_model, lora_config)
ft_model.print_trainable_parameters()
"""))

cells.append(code("""\
def format_training_example(row):
    messages = [
        {"role": "user", "content": PROMPT_TEMPLATE.format(clause=row["text"])},
        {"role": "assistant", "content": row["rating"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

def tokenize_fn(examples):
    texts = [format_training_example({"text": t, "rating": r}) for t, r in zip(examples["text"], examples["rating"])]
    tokenized = tokenizer(texts, truncation=True, max_length=512, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

train_ds = Dataset.from_list(train_rows).map(tokenize_fn, batched=True, remove_columns=["text", "rating", "category"])
val_ds = Dataset.from_list(val_rows).map(tokenize_fn, batched=True, remove_columns=["text", "rating", "category"])
"""))

cells.append(code("""\
training_args = TrainingArguments(
    output_dir="./clause-risk-lora",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=10,
    learning_rate=2e-4,
    fp16=(DEVICE == "cuda"),
    report_to="none",
)

trainer = Trainer(
    model=ft_model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
"""))

cells.append(md("## 6. Evaluate the fine-tuned model on the same test set"))

cells.append(code("""\
ft_model.eval()
ft_acc, ft_time, ft_true, ft_pred = evaluate(ft_model, test_rows, label="LoRA fine-tuned")

print(f"\\nZero-shot baseline accuracy: {baseline_acc:.3f}")
print(f"LoRA fine-tuned accuracy:    {ft_acc:.3f}")
print(f"Absolute improvement:       {(ft_acc - baseline_acc)*100:+.1f} percentage points")
"""))

cells.append(md("""\
## 7. Confusion matrices (side by side)
"""))

cells.append(code("""\
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

labels = ["Red", "Amber", "Green"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ConfusionMatrixDisplay.from_predictions(base_true, base_pred, labels=labels, ax=axes[0], colorbar=False)
axes[0].set_title(f"Zero-shot baseline (acc={baseline_acc:.2f})")

ConfusionMatrixDisplay.from_predictions(ft_true, ft_pred, labels=labels, ax=axes[1], colorbar=False)
axes[1].set_title(f"LoRA fine-tuned (acc={ft_acc:.2f})")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()
"""))

cells.append(md("""\
## 8. Save the adapter and results

Download `clause-risk-lora-adapter/` and `results.json` from the Colab file browser
(left sidebar) before the runtime disconnects — Colab does not persist files.
"""))

cells.append(code("""\
ft_model.save_pretrained("clause-risk-lora-adapter")
tokenizer.save_pretrained("clause-risk-lora-adapter")

results = {
    "model": MODEL_NAME,
    "test_set_size": len(test_rows),
    "zero_shot_baseline": {"accuracy": baseline_acc, "total_seconds": baseline_time},
    "lora_finetuned": {"accuracy": ft_acc, "total_seconds": ft_time},
    "improvement_pp": (ft_acc - baseline_acc) * 100,
}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
"""))

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = Path(__file__).parent / "clause_risk_finetune.ipynb"
out_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {out_path}")
