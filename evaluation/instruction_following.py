#!/usr/bin/env python3
"""
QWEN Instruction-Following Evaluation (TRUE DATA PARALLEL)

• One process per GPU
• One full model copy per GPU
• Thinking mode enabled
• Streaming logs
• Near-linear inference speedup
"""

import os
import json
import re
import argparse
import statistics
import time
import multiprocessing as mp
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =============================================================================
# CONFIG
# =============================================================================

MAX_NEW_TOKENS = 2048
LOG_RESPONSE_CHARS = 3000


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

INSTRUCTION_PROMPT_TEMPLATE = """You are an expert evaluator specializing in assessing instruction-following capabilities of audio language models, particularly in the domain of speaking style captioning. Your role is to rigorously evaluate whether generated captions accurately follow given instructions while maintaining factual accuracy about the audio content.

## Context
Audio language models are tasked with generating descriptive captions about speaking styles, vocal characteristics, and acoustic properties of speech recordings. These models receive specific instructions that guide what aspects to focus on, what style to adopt, or what format to use in their captions.

## Your Task
Evaluate how well the generated caption follows the provided instruction, considering both explicit requirements and implicit expectations.

---

## Input Materials

### Instruction Given to the Model:
<instruction>
{instruction}
</instruction>

### Generated Caption by the Model:
<generated_caption>
{caption}
</generated_caption>

---

## Evaluation Framework

### 1. **Instruction Compliance Categories**

Analyze the instruction for the following types of requirements:

**A. Content Requirements**
- Specific attributes to include (e.g., "describe the pitch," "mention the accent")
- Attributes to exclude or de-emphasize
- Level of detail requested (brief, detailed, comprehensive)
- Factual accuracy requirements

**B. Structural Requirements**
- Format specifications (paragraph, bullet points, structured description)
- Length constraints (word count, sentence count)
- Organizational pattern (chronological, importance-based, categorical)

**C. Stylistic Requirements**
- Tone (formal, casual, technical, creative)
- Perspective (objective observer, subjective interpreter, technical analyst)
- Language complexity (simple, sophisticated, jargon-heavy)
- Writing style (descriptive, analytical, narrative)

**D. Focus Requirements**
- Specific aspects to emphasize (prosody, emotion, acoustic features)
- Target audience considerations
- Use case or application context

**E. Constraint Requirements**
- Prohibitions (avoid certain terms, don't speculate)
- Boundaries (stick to observable features only)
- Scope limitations (focus on X, ignore Y)

### 2. **Evaluation Criteria**

For each requirement identified in the instruction, assess:

**Explicit Compliance (70% weight)**
- Does the caption directly address explicitly stated requirements?
- Are all mandatory elements present?
- Are any prohibited elements absent?
- Is the specified format/structure followed?

**Implicit Compliance (20% weight)**
- Does the caption follow reasonable interpretations of the instruction?
- Are contextual clues and implied expectations honored?
- Does it align with the spirit/intent of the instruction?

**Quality of Execution (10% weight)**
- How well are the requirements implemented (not just their presence)?
- Is the execution natural and coherent, or forced and awkward?
- Does following the instruction enhance or detract from caption quality?

### 3. **Common Instruction Types in Audio Captioning**

Be aware of these typical instruction patterns:

- **Attribute-focused**: "Describe the speaker's pitch and speaking rate"
- **Style-focused**: "Write in a technical, objective manner"
- **Audience-focused**: "Explain as if to a non-expert"
- **Format-focused**: "Provide a structured analysis with separate sections"
- **Perspective-focused**: "Focus on perceptual qualities rather than technical measurements"
- **Comparative**: "Compare this to typical conversational speech"
- **Holistic vs. Analytical**: "Provide an overall impression" vs. "Break down individual components"

---

## Scoring Guidelines

### Instruction-Following Success: Success / Partial Success / Failure

**Success**: 
- All critical requirements met (90%+ compliance)
- No significant violations of explicit constraints
- Natural execution that maintains caption quality
- Minor omissions only in non-essential elements

**Partial Success**:
- Most requirements met (60-89% compliance)
- Some explicit requirements missed or violated
- Execution may be somewhat awkward or forced
- Core intent of instruction honored despite gaps

**Failure**:
- Major requirements ignored (<60% compliance)
- Explicit constraints violated
- Instruction intent misunderstood or disregarded
- Caption appears written without considering the instruction

### Instruction-Following Accuracy: 1–10 Scale

**9-10 (Exceptional)**: Near-perfect instruction adherence. All explicit and implicit requirements met with natural, high-quality execution. Any deviations are trivial.

**7-8 (Strong)**: Excellent instruction following with minor gaps. All critical requirements met. May miss 1-2 minor elements or execute some aspects imperfectly, but overall very aligned.

**5-6 (Adequate)**: Satisfactory instruction following with notable gaps. Core requirements met but several secondary elements missed or poorly executed. Instruction intent generally honored.

**3-4 (Weak)**: Poor instruction following with major gaps. Many requirements missed or violated. Some alignment with instruction but significant divergence in execution.

**1-2 (Very Weak)**: Minimal instruction following. Most requirements ignored. Caption appears largely independent of the instruction. Major misunderstanding or disregard of requirements.

---

## Special Considerations

### Edge Cases to Watch For:

1. **Conflicting Requirements**: If instruction contains contradictory elements, evaluate based on how well the model navigates the conflict.

2. **Ambiguous Instructions**: If instruction is vague, give credit for reasonable interpretations that align with common sense.

3. **Impossible Requirements**: If instruction asks for information not determinable from audio alone, don't penalize the model for not hallucinating content.

4. **Over-compliance**: If model follows instruction so rigidly that caption quality suffers dramatically, note this in justification but still credit the instruction-following attempt.

5. **Creative Interpretation**: Distinguish between helpful creative interpretation (good) and ignoring instructions (bad).

### Common Pitfalls to Identify:

- ❌ **Instruction ignored entirely**: Caption is generic and could apply to any instruction
- ❌ **Partial attention**: Model addresses only the first part of a multi-part instruction
- ❌ **Format violation**: Instruction specifies structure but caption uses different format
- ❌ **Tone mismatch**: Formal instruction produces casual output, or vice versa
- ❌ **Scope creep**: Model includes content explicitly excluded by instruction
- ❌ **Misinterpretation**: Model misunderstands key terms or intent of instruction

---

## Output Format

Provide your evaluation in the following structure:

<analysis>
[Identify the key requirements in the instruction: What are the explicit demands? What are the implicit expectations? What constraints are specified? Categorize them as content, structural, stylistic, focus, or constraint requirements.]
</analysis>

<compliance_assessment>
[For each identified requirement, evaluate whether the caption meets it. Mark each as: ✓ Met, ⚠ Partially Met, or ✗ Not Met. Provide brief evidence from the caption.]
</compliance_assessment>

<strengths>
[Highlight what the model did well in following the instruction. What requirements were executed particularly effectively?]
</strengths>

<weaknesses>
[Identify gaps, violations, or missed opportunities. What requirements were ignored, poorly executed, or misinterpreted?]
</weaknesses>

<justification>
[Synthesize your analysis into a coherent evaluation. Explain your success determination and accuracy score. Reference specific examples from both the instruction and caption. Consider the severity of any violations and the overall alignment with instruction intent.]
</justification>

<ratings>
{{
  "instruction_following_success": "[Success/Partial Success/Failure]",
  "instruction_following_accuracy": [1-10]
}}
</ratings>

---

## Important Reminders

1. **Focus on instruction compliance, not caption quality**: A poorly written caption that follows instructions perfectly should score higher than a beautifully written caption that ignores them.

2. **Be precise in your reasoning**: Cite specific phrases from both the instruction and caption to support your evaluation.

3. **Distinguish between critical and minor requirements**: Not all instruction elements carry equal weight. Prioritize core requirements over secondary details.

4. **Consider context**: Audio captioning has domain-specific norms. Evaluate within the context of speaking style description conventions.

5. **Be fair but rigorous**: Give credit for good-faith attempts while maintaining high standards for what constitutes successful instruction-following.

6. **Avoid lenience bias**: Don't give full credit just because the caption is reasonable. It must specifically follow the given instruction.

Begin your evaluation now."""


# =============================================================================
# EVALUATION (ONE BATCH, ONE GPU)
# =============================================================================

def evaluate_batch(batch, tokenizer, model):
    device = torch.device("cuda:0")

    prompts = [
        INSTRUCTION_PROMPT_TEMPLATE.format(
            instruction=x["instruction"],
            caption=x["caption"]
        )
        for x in batch
    ]

    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for p in prompts
    ]

    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )

    input_len = inputs.input_ids.shape[1]
    outputs = generated[:, input_len:].cpu()

    results = []

    for meta, seq in zip(batch, outputs):
        text = tokenizer.decode(seq, skip_special_tokens=True).strip()

        print(
            f"\n=== GPU OUTPUT | index={meta['original_index']} | {meta['instruction_type']} ===\n"
            f"{text[:LOG_RESPONSE_CHARS]}\n"
            f"{'='*80}\n",
            flush=True
        )

        j = re.search(r"<justification>(.*?)</justification>", text, re.DOTALL)
        justification = j.group(1).strip() if j else "No justification found."

        try:
            r = re.search(r"<ratings>(.*?)</ratings>", text, re.DOTALL)
            parsed = json.loads(r.group(1).replace("'", '"'))
            success = parsed.get("instruction_following_success", "Failure")
            accuracy = float(parsed.get("instruction_following_accuracy", 0))
        except Exception:
            success, accuracy = "Failure", 0.0

        results.append({
            "index": meta["original_index"],
            "wav_path": meta["wav_path"],
            "instruction_type": meta["instruction_type"],
            "success": success,
            "accuracy": accuracy,
            "justification": justification
        })

    torch.cuda.empty_cache()
    return results


# =============================================================================
# WORKER PROCESS (ONE GPU)
# =============================================================================

def worker_process(gpu_id, chunk, args, out_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)

    print(f"[GPU {gpu_id}] Loading model...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16
    ).cuda()

    model.eval()
    print(f"[GPU {gpu_id}] Processing {len(chunk)} samples", flush=True)

    results = []

    for i in range(0, len(chunk), args.batch_size):
        batch = chunk[i:i + args.batch_size]
        results.extend(evaluate_batch(batch, tokenizer, model))

    out_queue.put(results)
    print(f"[GPU {gpu_id}] Done.", flush=True)


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(results):
    accuracies = [r["accuracy"] for r in results]
    successes = [r["success"] == "Success" for r in results]

    return {
        "instruction_following_score": {
            "mean": statistics.mean(accuracies),
            "median": statistics.median(accuracies),
            "std": statistics.pstdev(accuracies),
            "min": min(accuracies),
            "max": max(accuracies),
        },
        "instruction_following_accuracy": {
            "accuracy": sum(successes) / len(successes),
            "num_passing": sum(successes),
            "num_failing": len(successes) - sum(successes),
        }
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--model_name", default="Qwen/Qwen3-32B")
    p.add_argument("--batch_size", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    start = time.time()
    args = parse_arguments()

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available")

    print(f"Launching data-parallel evaluation on {num_gpus} GPUs", flush=True)

    data = json.load(open(args.input_file))

    TARGET_STYLES = {
        "speaker_idiosyncratic_style",
        "situational_contextual_style",
        "expressive_emotional_style",
        "linguistic_pragmatic_style",
        "perceptual_listener_centric_style",
        "holistic_creative_synthesis_style"
    }

    pairs = []
    for i, utt in enumerate(data):
        for key, info in utt.get("generated_captions", {}).items():
            # Extract style type from key (handles both "style_name" and "promptN_style_name" formats)
            # Remove "promptN_" prefix if present (e.g., "prompt1_speaker_idiosyncratic_style" -> "speaker_idiosyncratic_style")
            style_type = re.sub(r"^prompt\d+_", "", key)
            
            if style_type in TARGET_STYLES and "used_instruction" in info:
                pairs.append({
                    "original_index": i,
                    "wav_path": utt.get("wav_path", "unknown"),
                    "instruction_type": style_type,
                    "instruction": info["used_instruction"],
                    "caption": info["generated_caption"]
                })

    print(f"Total evaluation samples: {len(pairs)}", flush=True)

    chunks = [pairs[i::num_gpus] for i in range(num_gpus)]
    queue = mp.Queue()
    procs = []

    for gid in range(num_gpus):
        p = mp.Process(target=worker_process, args=(gid, chunks[gid], args, queue))
        p.start()
        procs.append(p)

    all_results = []
    for _ in range(num_gpus):
        all_results.extend(queue.get())

    for p in procs:
        p.join()

    all_results.sort(key=lambda x: x["index"])
    metrics = compute_metrics(all_results)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.output_file)

    json.dump({
        "aggregated_metrics": metrics,
        "individual_results": all_results
    }, open(out_path, "w"), indent=2)

    print(f"Saved results to {out_path}", flush=True)
    print(f"Total wall time: {time.time() - start:.2f}s for {num_gpus} GPUs and batch size {args.batch_size}", flush=True)
