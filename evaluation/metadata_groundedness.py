"""
QWEN LLM Judge Evaluation with Multi-GPU Support

This module evaluates the similarity between ground truth metadata and generated captions
using the QWEN model with multi-GPU processing capabilities.
"""

import json
import re
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import numpy as np
import argparse

PROMPT_TEMPLATE = """You are an expert linguist and speech analyst tasked with comparing the speaking styles between two sets of audio metadata. Your goal is to determine the similarity between a ground truth recording and a generated caption, focusing particularly on stylistic aspects of speech, intention, and prosody.

Please carefully review the following metadata:

Ground Truth Metadata (DATA1):
<ground_truth_metadata>
{data1}
</ground_truth_metadata>

Generated Caption Metadata (DATA2):
<generated_caption_metadata>
{data2}
</generated_caption_metadata>

Your task is to compare these two sets of metadata and rate their similarity on a scale of 1 to 10, where 1 indicates completely different speaking styles and 10 indicates identical speaking styles.

Instructions:
1. Analyze and compare the following attributes between DATA1 and DATA2:
   a) Speaker gender
   b) Accent
   c) Pitch
   d) Speaking rate
   e) Environment/noise level
   f) Emotion/expression
   g) Voice quality (e.g., guttural, silky)
   h) Prosody (including rhythm, intonation, and stress)
   i) Intention of speech

2. Inside your thinking block, wrap your attribute comparisons inside <attribute_comparison> tags. For each attribute:
   - Rate its similarity on a scale of 1-10
   - Summarize key similarities and differences
   - Consider how they contribute to the overall speaking style
   Pay special attention to stylistic aspects, intention, and prosody.

3. After analyzing all attributes, calculate an average similarity score based on your individual attribute ratings.

4. Don't penalize the generation if is more specific than the ground truth. For example, if the ground truth says "American accent" and the generation says "Boston accent", that's fine.

5. Provide a comprehensive justification for your similarity rating in <justification> tags. Explain how the similarities and differences in each attribute contribute to your overall assessment.

6. In the metadata, only using the following information:
    - speaker gender
    - accent
    - pitch
    - speaking rate
    - environment/noise level
    - emotion/expression

7. Finally, provide your similarity score in JSON format within <score> tags. The JSON should include a single key 'similarity_score' with a value between 1 and 10, based on your calculated average and overall assessment.

Example output structure (do not copy the content, only the structure):

<attribute_comparison>
a) Speaker gender:
   Similarity rating: X/10
   Key similarities: [...]
   Key differences: [...]
   Contribution to overall style: [...]

b) Accent:
   Similarity rating: X/10
   Key similarities: [...]
   Key differences: [...]
   Contribution to overall style: [...]

[... continue for all attributes ...]

Average similarity score: X/10
</attribute_comparison>

<justification>
[Your comprehensive justification for the similarity rating, synthesizing the analyses of individual attributes]
</justification>

<score>
{{'similarity_score': X}}
</score>

Remember to focus on the stylistic aspects of speech, paying particular attention to the intention behind the speech and the nuances of prosody in your analysis and justification.

Your final output should consist only of the justification and score, without duplicating the detailed attribute comparisons from your thinking block."""


def get_available_gpus():
    """Get the number of available GPUs."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def load_qwen_model_multi_gpu(model_name="Qwen/Qwen3-32B", device_map="auto"):
    """Loads the Qwen model and tokenizer with multi-GPU support."""
    print(f"Loading Qwen model: {model_name} ⚙️")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device_map
    )
    model.eval()
    
    print(f"Model loaded successfully on {get_available_gpus()} GPUs! ✅")
    return tokenizer, model


def compare_speech_metadata_qwen_batch_multi_gpu(batch_data, tokenizer, model):
    """Performs batch inference on a list of data pairs with multi-GPU support."""
    prompts = [PROMPT_TEMPLATE.format(data1=item['data1'], data2=item['data2']) for item in batch_data]

    messages_batch = [[{"role": "user", "content": p}] for p in prompts]
    texts = [tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    ) for messages in messages_batch]

    # model_inputs = tokenizer(texts, return_tensors="pt", padding=True)
    model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024,
        do_sample=False,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id
    )

    input_ids_len = model_inputs.input_ids.shape[1]
    output_ids_batch = generated_ids[:, input_ids_len:]

    batch_outputs = []
    for i, output_ids in enumerate(output_ids_batch):
        response_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        
        if i < 100:
            print(f"\n=== DEBUG: Response {i+1} ===")
            print(f"Response length: {len(response_text)}")
            print(f"Response preview: {response_text[:500]}...")
            print(f"Contains <score>: {'<score>' in response_text}")
            print(f"Contains <justification>: {'<justification>' in response_text}")
        
        similarity_score = 0
        try:
            score_match = re.search(r'<score>\s*(\{.*?\})\s*</score>', response_text, re.DOTALL)
            if score_match:
                score_json = json.loads(score_match.group(1))
                similarity_score = score_json.get('similarity_score', 0)
            else:
                json_match = re.search(r'\{[^}]*"similarity_score"[^}]*\}', response_text)
                if json_match:
                    score_json = json.loads(json_match.group(0))
                    similarity_score = score_json.get('similarity_score', 0)
                else:
                    number_match = re.search(r'"similarity_score":\s*(\d+(?:\.\d+)?)', response_text)
                    if number_match:
                        similarity_score = float(number_match.group(1))
                    else:
                        score_match = re.search(r'\b([1-9]|10)\b', response_text)
                        if score_match:
                            similarity_score = int(score_match.group(1))
                        else:
                            similarity_score = 0
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            if i < 2:
                print(f"JSON parsing error: {e}")
            similarity_score = 0

        justification = "No justification found"
        try:
            justification_match = re.search(r'<justification>(.*?)</justification>', response_text, re.DOTALL)
            if justification_match:
                justification = justification_match.group(1).strip()
            else:
                paragraphs = response_text.split('\n\n')
                if len(paragraphs) > 1:
                    justification = paragraphs[-1].strip()
        except Exception as e:
            if i < 2:
                print(f"Justification parsing error: {e}")

        batch_outputs.append({
            "similarity_score": similarity_score,
            "justification": justification,
            "full_response": response_text
        })

    del model_inputs
    del generated_ids
    del output_ids_batch
    torch.cuda.empty_cache()

    return batch_outputs


def process_chunk_qwen(chunk_data, tokenizer, model, chunk_id, batch_size=4):
    """Process a chunk of utterances on a specific GPU."""
    results = []
    failed_comparisons = []
    
    for i in range(0, len(chunk_data), batch_size):
        batch_items = chunk_data[i:i+batch_size]
        
        batch_input_data = [{
            'data1': item['data1'],
            'data2': item['data2']
        } for item in batch_items]

        try:
            batch_outputs = compare_speech_metadata_qwen_batch_multi_gpu(batch_input_data, tokenizer, model)

            for j, output in enumerate(batch_outputs):
                original_item = batch_items[j]
                result_item = {
                    "chunk_id": chunk_id,
                    "index": original_item['index'],
                    "wav_path": original_item['wav_path'],
                    "similarity_score": output["similarity_score"],
                    "justification": output["justification"],
                    "ground_truth_metadata": original_item['data1'],
                    "generated_caption_metadata": original_item['data2']
                }
                results.append(result_item)

        except Exception as e:
            print(f"\nError processing batch starting at index {i} in chunk {chunk_id}: {e}")
            for item in batch_items:
                failed_comparisons.append({
                    "index": item['index'],
                    "wav_path": item.get('wav_path', 'unknown'),
                    "reason": f"batch processing error: {str(e)}"
                })
        finally:
            torch.cuda.empty_cache()
    
    return results, failed_comparisons


def process_all_utterances_qwen_multi_gpu(input_file, tokenizer, model, output_dir=None, batch_size=4):
    """Processes all utterances from the input file using multi-GPU batch inference."""
    with open(input_file, 'r') as f:
        all_utterances = json.load(f)

    data_pairs = []
    for i, utterance in enumerate(all_utterances):
        if "generated_captions" in utterance and "metadata" in utterance:
            generated_caption = utterance["generated_captions"].get("holistic_creative_synthesis", {}).get("generated_caption", "")
            if not generated_caption:
                # Fallback: first available caption (e.g. single-task runs)
                for cap_info in utterance["generated_captions"].values():
                    if isinstance(cap_info, dict) and cap_info.get("generated_caption"):
                        generated_caption = cap_info["generated_caption"]
                        break
            generated_metadata = f"Generated caption: {generated_caption}"
            meta = dict(utterance["metadata"])
            # Gigaspeech rows may only carry ground_truth_caption at top level
            if "ground_truth_caption" not in meta and utterance.get("ground_truth_caption"):
                meta["ground_truth_caption"] = utterance["ground_truth_caption"]
            ground_truth_metadata = json.dumps(meta, indent=2)
            
            data_pairs.append({
                "index": i,
                "wav_path": utterance.get("wav_path", "unknown"),
                "data1": ground_truth_metadata,
                "data2": generated_metadata
            })

    print(f"Found {len(data_pairs)} utterances to process.")

    num_gpus = get_available_gpus()
    chunk_size = len(data_pairs) // num_gpus
    chunks = []
    
    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_gpus - 1 else len(data_pairs)
        chunks.append(data_pairs[start_idx:end_idx])

    print(f"Split {len(data_pairs)} utterances into {len(chunks)} chunks for {num_gpus} GPUs")
    
    all_results = []
    all_failed_comparisons = []
    
    for i, chunk in enumerate(tqdm(chunks, desc="Processing QWEN judge chunks")):
        chunk_results, chunk_failed = process_chunk_qwen(chunk, tokenizer, model, i, batch_size)
        all_results.extend(chunk_results)
        all_failed_comparisons.extend(chunk_failed)
        
        torch.cuda.empty_cache()

    all_results.sort(key=lambda x: x["index"])

    successful_comparisons = [r for r in all_results if isinstance(r['similarity_score'], (int, float))]
    similarity_scores = [r['similarity_score'] for r in successful_comparisons]

    aggregated_results = {
        "total_utterances": len(all_utterances),
        "successful_comparisons": len(successful_comparisons),
        "failed_comparisons": len(all_failed_comparisons),
        "aggregated_stats": {},
        "num_gpus_used": num_gpus
    }

    if similarity_scores:
        aggregated_results["aggregated_stats"] = {
            "mean_similarity": sum(similarity_scores) / len(similarity_scores),
            "min_similarity": min(similarity_scores),
            "max_similarity": max(similarity_scores),
            "median_similarity": sorted(similarity_scores)[len(similarity_scores) // 2],
            "scores_distribution": {
                "1-2": len([s for s in similarity_scores if 1 <= s <= 2]),
                "3-4": len([s for s in similarity_scores if 3 <= s <= 4]),
                "5-6": len([s for s in similarity_scores if 5 <= s <= 6]),
                "7-8": len([s for s in similarity_scores if 7 <= s <= 8]),
                "9-10": len([s for s in similarity_scores if 9 <= s <= 10])
            }
        }

    return {
        "aggregated_results": aggregated_results,
        "individual_results": all_results,
        "failed_log": all_failed_comparisons
    }


def run_full_evaluation_qwen_multi_gpu(input_file, output_dir, batch_size, tokenizer, model):
    """Main function to run the end-to-end evaluation with multi-GPU support."""
    print("Starting full evaluation with Qwen model using multi-GPU... 🚀")

    all_results = process_all_utterances_qwen_multi_gpu(
        input_file=input_file,
        tokenizer=tokenizer,
        model=model,
        output_dir=output_dir,
        batch_size=batch_size
    )

    print("\nEvaluation complete! ✨")
    return all_results


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="QWEN LLM Judge Evaluation with Multi-GPU Support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSON file containing utterances with metadata and generated captions"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the output results file will be saved"
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Name of the output results file"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for processing"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    num_gpus = get_available_gpus()
    if num_gpus == 0:
        print("No GPUs available. Please check your CUDA installation.")
        exit(1)
    
    print(f"Found {num_gpus} GPU(s) available")
    
    model_name = "Qwen/Qwen3-32B"

    tokenizer, model = load_qwen_model_multi_gpu(model_name, device_map="auto")

    results = run_full_evaluation_qwen_multi_gpu(
        input_file=args.input_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        tokenizer=tokenizer,
        model=model
    )

    print("\n--- 📊 Final Results Summary (Multi-GPU) ---")
    print(json.dumps(results['aggregated_results'], indent=2))

    summary_file = os.path.join(args.output_dir, args.output_file)
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {summary_file}")

    print("\nScript finished. Model and tokenizer are still in memory. 🧠")
