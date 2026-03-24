import os
import json
import torch
import re
from tqdm import tqdm
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# CONFIG (PAPER SETTINGS)
# =========================

MODEL_NAME = "Qwen/Qwen3-1.7B"
DEVICE = "cuda:0"

MAX_NEW_TOKENS = 16384   # paper AIME setting
NUM_SAMPLES = 8         # CRITICAL (paper)

TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20

OUTPUT_FILE = "outputs/aime_2025_qwen3_paper.jsonl"

os.makedirs("outputs", exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = True

# =========================
# ANSWER EXTRACTION
# =========================

def extract_answer(text):
    # boxed
    boxed = re.findall(r"\\boxed\{\s*(\d+)\s*\}", text)
    if boxed:
        return boxed[-1]

    # "answer is"
    match = re.search(r"[Tt]he answer is\s*(\d+)", text)
    if match:
        return match.group(1)

    # fallback (AIME safe)
    nums = re.findall(r"\b\d{1,3}\b", text)
    if nums:
        return nums[-1]

    return None


# =========================
# LOAD MODEL
# =========================

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map={"": DEVICE},
    trust_remote_code=True
)

model.eval()

# =========================
# LOAD DATASET
# =========================

print("Loading AIME 2025 dataset...")
dataset = load_dataset("MathArena/aime_2025", split="train")

print(f"Total problems: {len(dataset)}")

# =========================
# RESUME SUPPORT
# =========================

completed = 0

if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE) as f:
        completed = sum(1 for _ in f)

print(f"Resuming from {completed}")

# =========================
# GENERATION LOOP
# =========================

for idx, item in enumerate(tqdm(dataset)):

    if idx < completed:
        continue

    question = item["problem"]
    gt_answer = str(item["answer"]).strip()

    # PAPER-STYLE PROMPT (NO STOPPING INSTRUCTIONS)
    prompt = f"""Solve the following AIME problem.

Show full reasoning before giving the final answer.
Give the final answer in the form \\boxed{{integer}}.

Problem:
{question}

Solution:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    all_preds = []
    all_outputs = []

    for _ in range(NUM_SAMPLES):

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output[0]
        input_len = inputs["input_ids"].shape[-1]

        if len(generated_ids) > input_len:
            text = tokenizer.decode(
                generated_ids[input_len:],
                skip_special_tokens=True
            )
        else:
            text = ""

        pred = extract_answer(text)

        if pred is not None:
            all_preds.append(pred)

        all_outputs.append(text)

    # =========================
    # MAJORITY VOTE (CRITICAL)
    # =========================

    if all_preds:
        final_pred = Counter(all_preds).most_common(1)[0][0]
    else:
        final_pred = None

    result = {
        "id": idx,
        "question": question,
        "ground_truth": gt_answer,
        "predicted_answer": final_pred,
        "all_predictions": all_preds,
        "num_valid_samples": len(all_preds)
    }

    with open(OUTPUT_FILE, "a") as f:
        json.dump(result, f)
        f.write("\n")
        f.flush()

# =========================
# EVALUATION
# =========================

correct = 0
total = 0

with open(OUTPUT_FILE) as f:
    for line in f:
        item = json.loads(line)

        if item["predicted_answer"] == item["ground_truth"]:
            correct += 1

        total += 1

print("\n=======================")
print("Total:", total)
print("Correct:", correct)
print("Accuracy:", correct / total if total > 0 else 0)
print("=======================")