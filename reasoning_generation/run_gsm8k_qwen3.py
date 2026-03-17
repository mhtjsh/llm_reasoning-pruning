import os
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# =========================
# CONFIG
# =========================

MODEL_NAME = "Qwen/Qwen3-4B"
DEVICE = "cuda:0"

MAX_NEW_TOKENS = 512

TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20

OUTPUT_FILE = "outputs/gsm8k_results.jsonl"

os.makedirs("outputs", exist_ok=True)

torch.backends.cuda.matmul.allow_tf32 = True

# =========================
# ANSWER EXTRACTION
# =========================

def extract_answer(text):

    match = re.search(r"The answer is (-?\d+)", text)

    if match:
        return match.group(1)

    nums = re.findall(r"-?\d+", text)

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

print("Loading GSM8K dataset...")

dataset = load_dataset("gsm8k", "main")
test_set = dataset["test"]

# =========================
# RESUME LOGIC
# =========================

completed = 0

if os.path.exists(OUTPUT_FILE):

    with open(OUTPUT_FILE) as f:
        completed = sum(1 for _ in f)

print(f"Resuming from sample {completed}")

# =========================
# MAIN LOOP
# =========================

for idx, item in enumerate(tqdm(test_set)):

    if idx < completed:
        continue

    question = item["question"]
    gt_answer = item["answer"].split("####")[-1].strip()

    prompt = f"""Question: {question}
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            eos_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    pred_answer = extract_answer(text)

    result = {
        "id": idx,
        "question": question,
        "ground_truth": gt_answer,
        "model_output": text,
        "predicted_answer": pred_answer
    }

    # =========================
    # SAVE IMMEDIATELY
    # =========================

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

accuracy = correct / total

print("\n==========================")
print("Total:", total)
print("Correct:", correct)
print("Accuracy:", accuracy)
print("==========================")