import os
import json
import torch
import re
from tqdm import tqdm
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# CONFIG (PAPER-ALIGNED)
# =========================

MODEL_NAME = "Qwen/Qwen3-1.7B"
DEVICE = "cuda:0"

MAX_NEW_TOKENS = 16384   # your constraint
NUM_SAMPLES = 8          # your requirement

TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20

OUTPUT_FILE = "outputs/aime_qwen3_think_template_think_agnos.jsonl"

os.makedirs("outputs", exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = True

# =========================
# EXTRACTION
# =========================

def extract_answer(text):
    boxed = re.findall(r"\\boxed\{\s*(\d+)\s*\}", text)
    if boxed:
        return boxed[-1]

    match = re.search(r"[Tt]he answer is\s*(\d+)", text)
    if match:
        return match.group(1)

    nums = re.findall(r"\b\d{1,3}\b", text)
    if nums:
        return nums[-1]

    return None


# =========================
# SPLIT THINK / ANSWER (POST ONLY)
# =========================

def split_think_answer(text):
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        think = match.group(1).strip()
        answer = text[match.end():].strip()
        return think, answer

    return "", text


# =========================
# LOAD MODEL
# =========================

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

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

dataset = load_dataset("MathArena/aime_2025", split="train")

# =========================
# LOOP
# =========================

for idx, item in enumerate(tqdm(dataset)):

    question = item["problem"]
    gt_answer = str(item["answer"]).strip()

    # 🔥 EXACT QWEN TEMPLATE (CRITICAL)
    prompt = f"""<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
<think>
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    all_preds = []
    samples = []

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
                generated_ids[input_len:], skip_special_tokens=True
            )
        else:
            text = ""

        full_output = "<think>\n" + text  # reconstruct full structure

        think, answer_text = split_think_answer(full_output)
        pred = extract_answer(full_output)

        if pred:
            all_preds.append(pred)

        samples.append({
            "full_output": full_output,
            "thinking": think,
            "answer_text": answer_text,
            "predicted_answer": pred
        })

    # =========================
    # MAJORITY VOTE
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
        "samples": samples
    }

    with open(OUTPUT_FILE, "a") as f:
        json.dump(result, f)
        f.write("\n")

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
print("Accuracy:", correct / total if total else 0)
print("=======================")