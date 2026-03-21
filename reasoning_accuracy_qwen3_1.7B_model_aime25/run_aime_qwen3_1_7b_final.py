import os
import json
import torch
import re
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# CONFIG
# =========================

MODEL_NAME = "Qwen/Qwen3-1.7B"
DEVICE = "cuda:0"

MAX_NEW_TOKENS = 16384   # full reasoning budget

OUTPUT_FILE = "outputs/aime_2025_qwen3_reasoning.jsonl"

os.makedirs("outputs", exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = True

# =========================
# ANSWER EXTRACTION
# =========================

def extract_answer(text):
    # boxed answer (strict)
    boxed = re.findall(r"\\boxed\{\s*(\d+)\s*\}", text)
    if boxed:
        return boxed[-1]

    # "answer is X"
    match = re.search(r"[Tt]he answer is\s*(\d+)", text)
    if match:
        return match.group(1)

    # last number fallback (AIME safe)
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
# RESUME
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

    # 🔥 PURE REASONING PROMPT (no early stop instructions)
    prompt = f"""Solve the following AIME problem.

You must reason step by step carefully and completely.
Do not stop early.
At the end, give the final answer in the format \\boxed{{integer}}.

Problem:
{question}

Solution:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # 🔥 IMPORTANT: deterministic reasoning
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output[0]
    input_len = inputs["input_ids"].shape[-1]

    # SAFE DECODE
    if len(generated_ids) > input_len:
        text = tokenizer.decode(
            generated_ids[input_len:],
            skip_special_tokens=True
        )
    else:
        text = ""

    pred = extract_answer(text)

    result = {
        "id": idx,
        "question": question,
        "ground_truth": gt_answer,
        "model_output": text,
        "predicted_answer": pred
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