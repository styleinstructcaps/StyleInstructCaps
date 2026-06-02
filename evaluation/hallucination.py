# """
# QWEN LLM Judge Evaluation with Multi-GPU Support and Thinking Mode

# This module evaluates hallucinations in generated audio captions by comparing them
# against ground truth metadata using the QWEN model with multi-GPU processing capabilities
# and thinking mode enabled.
# """

# import json
# import re
# import os
# import argparse
# import torch
# import torch.nn as nn
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from tqdm import tqdm
# import multiprocessing as mp
# from functools import partial
# import numpy as np


# PROMPT_TEMPLATE = """You are an expert evaluator assessing the quality of automatically generated audio captions. Your task is to compare metadata from audio files with generated captions to identify and score hallucinations.

# ## Task Overview
# Compare the **metadata** (ground truth information about the audio) with a **generated caption** and evaluate the presence and severity of hallucinations.

# ## Input Format

# Ground Truth Metadata:
# <ground_truth_metadata>
# {data1}
# </ground_truth_metadata>

# Generated Caption:
# <generated_caption>
# {data2}
# </generated_caption>

# ## Evaluation Criteria

# ### What Counts as a Hallucination?
# A hallucination is information in the generated caption that:
# - Directly contradicts the metadata
# - Invents specific details not supported by any evidence
# - Misrepresents factual characteristics (e.g., wrong gender, wrong emotion, wrong accent)

# ### What Does NOT Count as a Hallucination?
# The following are **acceptable creative interpretations** and should NOT be penalized:
# - **Reasonable emotional inferences**: If metadata shows "laughing" audio, describing the speaker as "cheerful," "animated," or "enthusiastic" is acceptable
# - **Stylistic embellishments**: Descriptive language like "her voice carries warmth," "delivered with precision," or "words tumble out" that enhance the description without contradicting facts
# - **Contextual scenarios**: Adding reasonable context (e.g., "as if addressing an audience," "in a quiet environment") when supported by acoustic characteristics
# - **Prosodic interpretations**: Descriptions of rhythm, cadence, emphasis, or delivery style that are reasonable given the speech characteristics
# - **Minor elaborations**: Details that naturally extend from the metadata without contradicting it

# ### Hallucination Examples

# **SEVERE Hallucinations:**
# - Metadata indicates male speaker → Caption says "female speaker"
# - Metadata shows speaker p026 → Caption invents specific name or identity
# - Metadata indicates disgust emotion → Caption describes joy or happiness
# - Inventing specific events or quoted speech not in the audio

# **MODERATE Hallucinations:**
# - Claiming "noisy environment" when metadata suggests clean audio
# - Describing specific background sounds not mentioned in metadata
# - Asserting definitive context (e.g., "delivering a news broadcast") without support

# **MINOR Hallucinations:**
# - Slightly exaggerating a characteristic (e.g., "booming voice" for a moderately loud voice)
# - Adding plausible but unverified details that don't contradict metadata

# ## Scoring Instructions

# Provide two scores:

# ### 1. Hallucination Count (0-10 scale)
# - **0-2**: No hallucinations or only very minor embellishments
# - **3-4**: 1-2 minor hallucinations that don't significantly distort the audio
# - **5-6**: Multiple minor hallucinations OR 1 moderate hallucination
# - **7-8**: Multiple moderate hallucinations OR 1 severe hallucination
# - **9-10**: Multiple severe hallucinations that fundamentally misrepresent the audio

# ### 2. Severity Level
# - **Low**: Only minor embellishments; caption is largely accurate
# - **Medium**: Some factual errors that partially misrepresent the audio
# - **High**: Severe contradictions or fabrications that fundamentally mischaracterize the audio

# ## Instructions
# 1. In your thinking block, carefully analyze the metadata and generated caption step by step:
#    - List all factual claims made in the generated caption
#    - For each claim, check if it's supported by, contradicts, or extends beyond the metadata
#    - Categorize each issue as minor, moderate, or severe
#    - Count the total number of hallucinations
# 2. Be lenient with creative, descriptive language that doesn't contradict facts
# 3. Focus on factual accuracy regarding speaker identity, emotions, accent, and acoustic characteristics
# 4. Distinguish between reasonable inference and baseless fabrication

# ## Output Format

# After your thinking, provide your evaluation in the following structure:

# <analysis>
# [Brief explanation of what you observed, noting any hallucinations found and why they are/aren't problematic]
# </analysis>

# <justification>
# [Explain your scoring in detail, referencing specific examples from the caption and how they relate to the metadata]
# </justification>

# <score>
# {{"hallucination_count": X, "severity": "Low/Medium/High"}}
# </score>

# Remember: Context is important - "laughing" audio naturally suggests positive emotions. Be strict about factual contradictions but lenient about creative descriptions that align with the metadata.

# Your final output should consist only of the analysis, justification and score, without duplicating the detailed step-by-step analysis from your thinking block."""


# def get_available_gpus():
#     """Get the number of available GPUs."""
#     if torch.cuda.is_available():
#         return torch.cuda.device_count()
#     return 0


# def load_qwen_model_multi_gpu(model_name="Qwen/Qwen3-32B", device_map="auto"):
#     """Loads the Qwen model and tokenizer with multi-GPU support."""
#     print(f"Loading Qwen model: {model_name} ⚙️")
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     tokenizer.padding_side = "left"
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype="auto",
#         device_map=device_map
#     )
#     model.eval()
    
#     print(f"Model loaded successfully on {get_available_gpus()} GPUs! ✅")
#     return tokenizer, model


# def compare_speech_metadata_qwen_batch_multi_gpu(batch_data, tokenizer, model):
#     """Performs batch inference on a list of data pairs with multi-GPU support."""
#     prompts = [PROMPT_TEMPLATE.format(data1=item['data1'], data2=item['data2']) for item in batch_data]

#     messages_batch = [[{"role": "user", "content": p}] for p in prompts]
#     texts = [tokenizer.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
#     ) for messages in messages_batch]

#     model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)

#     generated_ids = model.generate(
#         **model_inputs,
#         max_new_tokens=2048,  # Increased for thinking mode
#         do_sample=False,
#         num_beams=1,
#         eos_token_id=tokenizer.eos_token_id
#     )

#     input_ids_len = model_inputs.input_ids.shape[1]
#     output_ids_batch = generated_ids[:, input_ids_len:]

#     batch_outputs = []
#     for i, output_ids in enumerate(output_ids_batch):
#         response_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        
#         if i < 5:  # Debug first 5 responses
#             print(f"\n=== DEBUG: Response {i+1} ===")
#             print(f"Response length: {len(response_text)}")
#             print(f"Response preview: {response_text[:500]}...")
#             print(f"Contains <score>: {'<score>' in response_text}")
#             print(f"Contains <analysis>: {'<analysis>' in response_text}")
#             print(f"Contains <justification>: {'<justification>' in response_text}")
        
#         # Parse hallucination count and severity
#         hallucination_count = 0
#         severity = "Low"
#         try:
#             score_match = re.search(r'<score>\s*(\{.*?\})\s*</score>', response_text, re.DOTALL)
#             if score_match:
#                 score_json = json.loads(score_match.group(1))
#                 hallucination_count = score_json.get('hallucination_count', 0)
#                 severity = score_json.get('severity', 'Low')
#             else:
#                 # Fallback: try to find JSON anywhere in the response
#                 json_match = re.search(r'\{[^}]*"hallucination_count"[^}]*"severity"[^}]*\}', response_text, re.DOTALL)
#                 if json_match:
#                     score_json = json.loads(json_match.group(0))
#                     hallucination_count = score_json.get('hallucination_count', 0)
#                     severity = score_json.get('severity', 'Low')
#                 else:
#                     # Try to extract individually
#                     count_match = re.search(r'"hallucination_count":\s*(\d+)', response_text)
#                     severity_match = re.search(r'"severity":\s*"(Low|Medium|High)"', response_text, re.IGNORECASE)
#                     if count_match:
#                         hallucination_count = int(count_match.group(1))
#                     if severity_match:
#                         severity = severity_match.group(1).capitalize()
#                     else:
#                         # Last resort: look for standalone numbers in context
#                         context_match = re.search(r'hallucination[_ ]count[^\d]*(\d+)', response_text, re.IGNORECASE)
#                         if context_match:
#                             hallucination_count = int(context_match.group(1))
#         except (json.JSONDecodeError, TypeError, ValueError) as e:
#             if i < 2:
#                 print(f"JSON parsing error: {e}")
#             hallucination_count = 0
#             severity = "Low"

#         # Parse justification
#         justification = "No justification found"
#         analysis = ""
#         try:
#             justification_match = re.search(r'<justification>(.*?)</justification>', response_text, re.DOTALL)
#             if justification_match:
#                 justification = justification_match.group(1).strip()
            
#             # Also extract analysis if available
#             analysis_match = re.search(r'<analysis>(.*?)</analysis>', response_text, re.DOTALL)
#             if analysis_match:
#                 analysis = analysis_match.group(1).strip()
#                 # If no justification found, use analysis
#                 if justification == "No justification found":
#                     justification = analysis
            
#             # If neither found, try to extract from the visible part (after thinking)
#             if justification == "No justification found":
#                 # Look for text between common markers
#                 paragraphs = response_text.split('\n\n')
#                 if len(paragraphs) > 1:
#                     # Find the last substantial paragraph before the score
#                     for para in reversed(paragraphs):
#                         if len(para) > 50 and '<score>' not in para:
#                             justification = para.strip()
#                             break
#         except Exception as e:
#             if i < 2:
#                 print(f"Justification parsing error: {e}")

#         batch_outputs.append({
#             "hallucination_count": hallucination_count,
#             "severity": severity,
#             "justification": justification,
#             "analysis": analysis,
#             "full_response": response_text
#         })

#     del model_inputs
#     del generated_ids
#     del output_ids_batch
#     torch.cuda.empty_cache()

#     return batch_outputs


# def process_chunk_qwen(chunk_data, tokenizer, model, chunk_id, batch_size=4):
#     """Process a chunk of utterances on a specific GPU."""
#     results = []
#     failed_comparisons = []
    
#     for i in range(0, len(chunk_data), batch_size):
#         batch_items = chunk_data[i:i+batch_size]
        
#         batch_input_data = [{
#             'data1': item['data1'],
#             'data2': item['data2']
#         } for item in batch_items]

#         try:
#             batch_outputs = compare_speech_metadata_qwen_batch_multi_gpu(batch_input_data, tokenizer, model)

#             for j, output in enumerate(batch_outputs):
#                 original_item = batch_items[j]
#                 result_item = {
#                     "chunk_id": chunk_id,
#                     "index": original_item['index'],
#                     "wav_path": original_item['wav_path'],
#                     "hallucination_count": output["hallucination_count"],
#                     "severity": output["severity"],
#                     "justification": output["justification"],
#                     "analysis": output.get("analysis", ""),
#                     "ground_truth_metadata": original_item['data1'],
#                     "generated_caption_metadata": original_item['data2'],
#                     "full_response": output.get("full_response", "")
#                 }
#                 results.append(result_item)

#         except Exception as e:
#             print(f"\nError processing batch starting at index {i} in chunk {chunk_id}: {e}")
#             for item in batch_items:
#                 failed_comparisons.append({
#                     "index": item['index'],
#                     "wav_path": item.get('wav_path', 'unknown'),
#                     "reason": f"batch processing error: {str(e)}"
#                 })
#         finally:
#             torch.cuda.empty_cache()
    
#     return results, failed_comparisons


# def process_all_utterances_qwen_multi_gpu(input_file, tokenizer, model, output_dir=None, batch_size=4):
#     """Processes all utterances from the input file using multi-GPU batch inference."""
#     with open(input_file, 'r') as f:
#         all_utterances = json.load(f)

#     data_pairs = []
#     for i, utterance in enumerate(all_utterances):
#         if "generated_captions" in utterance and "metadata" in utterance:
#             generated_caption = utterance["generated_captions"].get("holistic_creative_synthesis", {}).get("generated_caption", "")
#             generated_metadata = f"{generated_caption}"
#             ground_truth_metadata = json.dumps(utterance["metadata"], indent=2)
            
#             data_pairs.append({
#                 "index": i,
#                 "wav_path": utterance.get("wav_path", "unknown"),
#                 "data1": ground_truth_metadata,
#                 "data2": generated_metadata
#             })

#     print(f"Found {len(data_pairs)} utterances to process.")

#     num_gpus = get_available_gpus()
#     chunk_size = len(data_pairs) // num_gpus
#     chunks = []
    
#     for i in range(num_gpus):
#         start_idx = i * chunk_size
#         end_idx = start_idx + chunk_size if i < num_gpus - 1 else len(data_pairs)
#         chunks.append(data_pairs[start_idx:end_idx])

#     print(f"Split {len(data_pairs)} utterances into {len(chunks)} chunks for {num_gpus} GPUs")
    
#     all_results = []
#     all_failed_comparisons = []
    
#     for i, chunk in enumerate(tqdm(chunks, desc="Processing QWEN hallucination evaluation (thinking mode)")):
#         chunk_results, chunk_failed = process_chunk_qwen(chunk, tokenizer, model, i, batch_size)
#         all_results.extend(chunk_results)
#         all_failed_comparisons.extend(chunk_failed)
        
#         torch.cuda.empty_cache()

#     all_results.sort(key=lambda x: x["index"])

#     successful_comparisons = [r for r in all_results if isinstance(r['hallucination_count'], (int, float))]
#     hallucination_counts = [r['hallucination_count'] for r in successful_comparisons]
#     severity_counts = {"Low": 0, "Medium": 0, "High": 0}
#     for r in successful_comparisons:
#         severity_counts[r['severity']] = severity_counts.get(r['severity'], 0) + 1

#     aggregated_results = {
#         "total_utterances": len(all_utterances),
#         "successful_comparisons": len(successful_comparisons),
#         "failed_comparisons": len(all_failed_comparisons),
#         "aggregated_stats": {},
#         "num_gpus_used": num_gpus
#     }

#     if hallucination_counts:
#         aggregated_results["aggregated_stats"] = {
#             "mean_hallucination_count": sum(hallucination_counts) / len(hallucination_counts),
#             "min_hallucination_count": min(hallucination_counts),
#             "max_hallucination_count": max(hallucination_counts),
#             "median_hallucination_count": sorted(hallucination_counts)[len(hallucination_counts) // 2],
#             "severity_distribution": severity_counts,
#             "hallucination_distribution": {
#                 "0-2 (Minimal)": len([c for c in hallucination_counts if 0 <= c <= 2]),
#                 "3-4 (Minor)": len([c for c in hallucination_counts if 3 <= c <= 4]),
#                 "5-6 (Moderate)": len([c for c in hallucination_counts if 5 <= c <= 6]),
#                 "7-8 (Severe)": len([c for c in hallucination_counts if 7 <= c <= 8]),
#                 "9-10 (Critical)": len([c for c in hallucination_counts if 9 <= c <= 10])
#             }
#         }

#     return {
#         "aggregated_results": aggregated_results,
#         "individual_results": all_results,
#         "failed_log": all_failed_comparisons
#     }


# def run_full_evaluation_qwen_multi_gpu(input_file, output_dir, batch_size, tokenizer, model):
#     """Main function to run the end-to-end hallucination evaluation with multi-GPU support."""
#     print("Starting hallucination evaluation with Qwen model using multi-GPU and thinking mode... 🚀")

#     all_results = process_all_utterances_qwen_multi_gpu(
#         input_file=input_file,
#         tokenizer=tokenizer,
#         model=model,
#         output_dir=output_dir,
#         batch_size=batch_size
#     )

#     print("\nEvaluation complete! ✨")
#     return all_results


# def parse_arguments():
#     """Parse command-line arguments."""
#     parser = argparse.ArgumentParser(
#         description="QWEN LLM Judge Evaluation with Multi-GPU Support and Thinking Mode",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
    
#     parser.add_argument(
#         "--input_file",
#         type=str,
#         default="/path/to/project/results/qwen2audio_trial_list_results.json",
#         help="Path to the input JSON file containing utterances with metadata and generated captions"
#     )
    
#     parser.add_argument(
#         "--output_dir",
#         type=str,
#         default="/path/to/project/results/",
#         help="Directory where the output results file will be saved"
#     )
    
#     parser.add_argument(
#         "--output_file",
#         type=str,
#         default="hallucination_evaluation_results_thinking.json",
#         help="Name of the output JSON file to save evaluation results"
#     )
    
#     parser.add_argument(
#         "--batch_size",
#         type=int,
#         default=4,
#         help="Batch size for processing (may want to reduce for thinking mode due to longer generations)"
#     )
    
#     parser.add_argument(
#         "--model_name",
#         type=str,
#         default="Qwen/Qwen3-32B",
#         help="Name or path of the Qwen model to use for evaluation"
#     )
    
#     parser.add_argument(
#         "--device_map",
#         type=str,
#         default="auto",
#         help="Device map strategy for multi-GPU distribution (e.g., 'auto', 'balanced', 'balanced_low_0')"
#     )
    
#     return parser.parse_args()


# if __name__ == "__main__":
#     args = parse_arguments()
    
#     num_gpus = get_available_gpus()
#     if num_gpus == 0:
#         print("No GPUs available. Please check your CUDA installation.")
#         exit(1)
    
#     print(f"Found {num_gpus} GPU(s) available")
    
#     tokenizer, model = load_qwen_model_multi_gpu(args.model_name, device_map=args.device_map)

#     results = run_full_evaluation_qwen_multi_gpu(
#         input_file=args.input_file,
#         output_dir=args.output_dir,
#         batch_size=args.batch_size,
#         tokenizer=tokenizer,
#         model=model
#     )

#     print("\n--- 📊 Final Hallucination Evaluation Summary (Multi-GPU, Thinking Mode) ---")
#     print(json.dumps(results['aggregated_results'], indent=2))

#     summary_file = os.path.join(args.output_dir, args.output_file)
#     with open(summary_file, 'w') as f:
#         json.dump(results, f, indent=2)
#     print(f"\nFull results saved to {summary_file}")

#     print("\nScript finished. Model and tokenizer are still in memory. 🧠")

# """
# QWEN LLM Judge Evaluation with Multi-GPU Support and Thinking Mode

# This module evaluates hallucinations in generated audio captions by comparing them
# against ground truth metadata using the QWEN model with multi-GPU processing capabilities
# and thinking mode enabled.
# """

# import json
# import re
# import os
# import argparse
# import torch
# import torch.nn as nn
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from tqdm import tqdm
# import multiprocessing as mp
# from functools import partial
# import numpy as np


# PROMPT_TEMPLATE = """You are an expert evaluator assessing the quality of automatically generated audio captions. Your task is to compare metadata from audio files with generated captions to identify and score hallucinations.

# ## Task Overview
# Compare the **metadata** (ground truth information about the audio) with a **generated caption** and evaluate the presence and severity of hallucinations.

# ## Input Format

# Ground Truth Metadata:
# <ground_truth_metadata>
# {data1}
# </ground_truth_metadata>

# Generated Caption:
# <generated_caption>
# {data2}
# </generated_caption>

# ## Evaluation Criteria

# ### What Counts as a Hallucination?
# A hallucination is information in the generated caption that:
# - Directly contradicts the metadata
# - Invents specific details not supported by any evidence
# - Misrepresents factual characteristics (e.g., wrong gender, wrong emotion, wrong accent)

# ### What Does NOT Count as a Hallucination?
# The following are **acceptable creative interpretations** and should NOT be penalized:
# - **Reasonable emotional inferences**: If metadata shows "laughing" audio, describing the speaker as "cheerful," "animated," or "enthusiastic" is acceptable
# - **Stylistic embellishments**: Descriptive language like "her voice carries warmth," "delivered with precision," or "words tumble out" that enhance the description without contradicting facts
# - **Contextual scenarios**: Adding reasonable context (e.g., "as if addressing an audience," "in a quiet environment") when supported by acoustic characteristics
# - **Prosodic interpretations**: Descriptions of rhythm, cadence, emphasis, or delivery style that are reasonable given the speech characteristics
# - **Minor elaborations**: Details that naturally extend from the metadata without contradicting it

# ### Hallucination Examples

# **SEVERE Hallucinations:**
# - Metadata indicates male speaker → Caption says "female speaker"
# - Metadata shows speaker p026 → Caption invents specific name or identity
# - Metadata indicates disgust emotion → Caption describes joy or happiness
# - Inventing specific events or quoted speech not in the audio

# **MODERATE Hallucinations:**
# - Claiming "noisy environment" when metadata suggests clean audio
# - Describing specific background sounds not mentioned in metadata
# - Asserting definitive context (e.g., "delivering a news broadcast") without support

# **MINOR Hallucinations:**
# - Slightly exaggerating a characteristic (e.g., "booming voice" for a moderately loud voice)
# - Adding plausible but unverified details that don't contradict metadata

# ## Scoring Instructions

# Provide two scores:

# ### 1. Hallucination Count (0-10 scale)
# - **0-2**: No hallucinations or only very minor embellishments
# - **3-4**: 1-2 minor hallucinations that don't significantly distort the audio
# - **5-6**: Multiple minor hallucinations OR 1 moderate hallucination
# - **7-8**: Multiple moderate hallucinations OR 1 severe hallucination
# - **9-10**: Multiple severe hallucinations that fundamentally misrepresent the audio

# ### 2. Severity Level
# - **Low**: Only minor embellishments; caption is largely accurate
# - **Medium**: Some factual errors that partially misrepresent the audio
# - **High**: Severe contradictions or fabrications that fundamentally mischaracterize the audio

# ## Instructions
# 1. In your thinking block, carefully analyze the metadata and generated caption step by step:
#    - List all factual claims made in the generated caption
#    - For each claim, check if it's supported by, contradicts, or extends beyond the metadata
#    - Categorize each issue as minor, moderate, or severe
#    - Count the total number of hallucinations
# 2. Be lenient with creative, descriptive language that doesn't contradict facts
# 3. Focus on factual accuracy regarding speaker identity, emotions, accent, and acoustic characteristics
# 4. Distinguish between reasonable inference and baseless fabrication

# ## Output Format

# After your thinking, provide your evaluation in the following structure:

# <analysis>
# [Brief explanation of what you observed, noting any hallucinations found and why they are/aren't problematic]
# </analysis>

# <justification>
# [Explain your scoring in detail, referencing specific examples from the caption and how they relate to the metadata]
# </justification>

# <score>
# {{"hallucination_count": X, "severity": "Low/Medium/High"}}
# </score>

# Remember: Context is important - "laughing" audio naturally suggests positive emotions. Be strict about factual contradictions but lenient about creative descriptions that align with the metadata.

# Your final output should consist only of the analysis, justification and score, without duplicating the detailed step-by-step analysis from your thinking block."""


# def get_available_gpus():
#     """Get the number of available GPUs."""
#     if torch.cuda.is_available():
#         return torch.cuda.device_count()
#     return 0


# def load_qwen_model_on_gpu(model_name, gpu_id):
#     """Loads the Qwen model and tokenizer on a specific GPU."""
#     print(f"Loading Qwen model on GPU {gpu_id}: {model_name} ⚙️")
    
#     # Set the device for this process
#     device = f"cuda:{gpu_id}"
#     torch.cuda.set_device(gpu_id)
    
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     tokenizer.padding_side = "left"
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype=torch.bfloat16,
#         device_map={"": gpu_id}  # Force model to specific GPU
#     )
#     model.eval()
    
#     print(f"Model loaded successfully on GPU {gpu_id}! ✅")
#     return tokenizer, model, device


# def compare_speech_metadata_qwen_batch(batch_data, tokenizer, model, device, debug=False):
#     """Performs batch inference on a list of data pairs."""
#     prompts = [PROMPT_TEMPLATE.format(data1=item['data1'], data2=item['data2']) for item in batch_data]

#     messages_batch = [[{"role": "user", "content": p}] for p in prompts]
#     texts = [tokenizer.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
#     ) for messages in messages_batch]

#     model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)

#     generated_ids = model.generate(
#         **model_inputs,
#         max_new_tokens=2048,
#         do_sample=False,
#         num_beams=1,
#         eos_token_id=tokenizer.eos_token_id
#     )

#     input_ids_len = model_inputs.input_ids.shape[1]
#     output_ids_batch = generated_ids[:, input_ids_len:]

#     batch_outputs = []
#     for i, output_ids in enumerate(output_ids_batch):
#         response_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        
#         if debug and i < 5:  # Debug first 5 responses
#             print(f"\n=== DEBUG: Response {i+1} ===")
#             print(f"Response length: {len(response_text)}")
#             print(f"Response preview: {response_text[:500]}...")
#             print(f"Contains <score>: {'<score>' in response_text}")
#             print(f"Contains <analysis>: {'<analysis>' in response_text}")
#             print(f"Contains <justification>: {'<justification>' in response_text}")
        
#         # Parse hallucination count and severity
#         hallucination_count = 0
#         severity = "Low"
#         try:
#             score_match = re.search(r'<score>\s*(\{.*?\})\s*</score>', response_text, re.DOTALL)
#             if score_match:
#                 score_json = json.loads(score_match.group(1))
#                 hallucination_count = score_json.get('hallucination_count', 0)
#                 severity = score_json.get('severity', 'Low')
#             else:
#                 # Fallback: try to find JSON anywhere in the response
#                 json_match = re.search(r'\{[^}]*"hallucination_count"[^}]*"severity"[^}]*\}', response_text, re.DOTALL)
#                 if json_match:
#                     score_json = json.loads(json_match.group(0))
#                     hallucination_count = score_json.get('hallucination_count', 0)
#                     severity = score_json.get('severity', 'Low')
#                 else:
#                     # Try to extract individually
#                     count_match = re.search(r'"hallucination_count":\s*(\d+)', response_text)
#                     severity_match = re.search(r'"severity":\s*"(Low|Medium|High)"', response_text, re.IGNORECASE)
#                     if count_match:
#                         hallucination_count = int(count_match.group(1))
#                     if severity_match:
#                         severity = severity_match.group(1).capitalize()
#         except (json.JSONDecodeError, TypeError, ValueError) as e:
#             if debug:
#                 print(f"JSON parsing error: {e}")
#             hallucination_count = 0
#             severity = "Low"

#         # Parse justification
#         justification = "No justification found"
#         analysis = ""
#         try:
#             justification_match = re.search(r'<justification>(.*?)</justification>', response_text, re.DOTALL)
#             if justification_match:
#                 justification = justification_match.group(1).strip()
            
#             # Also extract analysis if available
#             analysis_match = re.search(r'<analysis>(.*?)</analysis>', response_text, re.DOTALL)
#             if analysis_match:
#                 analysis = analysis_match.group(1).strip()
#                 # If no justification found, use analysis
#                 if justification == "No justification found":
#                     justification = analysis
#         except Exception as e:
#             if debug:
#                 print(f"Justification parsing error: {e}")

#         batch_outputs.append({
#             "hallucination_count": hallucination_count,
#             "severity": severity,
#             "justification": justification,
#             "analysis": analysis,
#             "full_response": response_text
#         })

#     del model_inputs
#     del generated_ids
#     del output_ids_batch
#     torch.cuda.empty_cache()

#     return batch_outputs


# def process_chunk_on_gpu(gpu_id, chunk_data, model_name, batch_size, chunk_id):
#     """Process a chunk of utterances on a specific GPU."""
#     try:
#         # Load model on this GPU
#         tokenizer, model, device = load_qwen_model_on_gpu(model_name, gpu_id)
        
#         results = []
#         failed_comparisons = []
        
#         # Process in batches
#         for i in range(0, len(chunk_data), batch_size):
#             batch_items = chunk_data[i:i+batch_size]
            
#             batch_input_data = [{
#                 'data1': item['data1'],
#                 'data2': item['data2']
#             } for item in batch_items]

#             try:
#                 batch_outputs = compare_speech_metadata_qwen_batch(
#                     batch_input_data, tokenizer, model, device, debug=(i == 0)
#                 )

#                 for j, output in enumerate(batch_outputs):
#                     original_item = batch_items[j]
#                     result_item = {
#                         "chunk_id": chunk_id,
#                         "gpu_id": gpu_id,
#                         "index": original_item['index'],
#                         "wav_path": original_item['wav_path'],
#                         "hallucination_count": output["hallucination_count"],
#                         "severity": output["severity"],
#                         "justification": output["justification"],
#                         "analysis": output.get("analysis", ""),
#                         "ground_truth_metadata": original_item['data1'],
#                         "generated_caption_metadata": original_item['data2'],
#                         "full_response": output.get("full_response", "")
#                     }
#                     results.append(result_item)

#             except Exception as e:
#                 print(f"\nError processing batch starting at index {i} on GPU {gpu_id}: {e}")
#                 for item in batch_items:
#                     failed_comparisons.append({
#                         "index": item['index'],
#                         "wav_path": item.get('wav_path', 'unknown'),
#                         "reason": f"batch processing error: {str(e)}"
#                     })
#             finally:
#                 torch.cuda.empty_cache()
        
#         # Cleanup
#         del model
#         del tokenizer
#         torch.cuda.empty_cache()
        
#         return results, failed_comparisons
        
#     except Exception as e:
#         print(f"Error on GPU {gpu_id}: {e}")
#         return [], [{"gpu_id": gpu_id, "reason": str(e)}]


# def process_all_utterances_multi_gpu(input_file, model_name, batch_size, num_gpus):
#     """Processes all utterances using multiple GPUs in parallel."""
#     with open(input_file, 'r') as f:
#         all_utterances = json.load(f)

#     data_pairs = []
#     for i, utterance in enumerate(all_utterances):
#         if "generated_captions" in utterance and "metadata" in utterance:
#             generated_caption = utterance["generated_captions"].get("holistic_creative_synthesis", {}).get("generated_caption", "")
#             generated_metadata = f"{generated_caption}"
#             ground_truth_metadata = json.dumps(utterance["metadata"], indent=2)
            
#             data_pairs.append({
#                 "index": i,
#                 "wav_path": utterance.get("wav_path", "unknown"),
#                 "data1": ground_truth_metadata,
#                 "data2": generated_metadata
#             })

#     print(f"Found {len(data_pairs)} utterances to process across {num_gpus} GPUs.")

#     # Split data across GPUs
#     chunk_size = len(data_pairs) // num_gpus
#     chunks = []
    
#     for i in range(num_gpus):
#         start_idx = i * chunk_size
#         end_idx = start_idx + chunk_size if i < num_gpus - 1 else len(data_pairs)
#         chunks.append(data_pairs[start_idx:end_idx])
#         print(f"GPU {i}: Processing {len(chunks[i])} utterances (indices {start_idx}-{end_idx-1})")

#     # Use multiprocessing to process chunks in parallel
#     print("\nStarting parallel processing across GPUs...")
    
#     with mp.Pool(processes=num_gpus) as pool:
#         # Create partial function with fixed arguments
#         process_func = partial(
#             process_chunk_on_gpu,
#             model_name=model_name,
#             batch_size=batch_size
#         )
        
#         # Map GPU IDs and chunks to the function
#         args = [(i, chunks[i], i) for i in range(num_gpus)]
#         results_list = pool.starmap(process_func, args)
    
#     # Combine results from all GPUs
#     all_results = []
#     all_failed_comparisons = []
    
#     for results, failed in results_list:
#         all_results.extend(results)
#         all_failed_comparisons.extend(failed)
    
#     # Sort by original index
#     all_results.sort(key=lambda x: x["index"])

#     # Calculate statistics
#     successful_comparisons = [r for r in all_results if isinstance(r['hallucination_count'], (int, float))]
#     hallucination_counts = [r['hallucination_count'] for r in successful_comparisons]
#     severity_counts = {"Low": 0, "Medium": 0, "High": 0}
#     for r in successful_comparisons:
#         severity_counts[r['severity']] = severity_counts.get(r['severity'], 0) + 1

#     aggregated_results = {
#         "total_utterances": len(all_utterances),
#         "successful_comparisons": len(successful_comparisons),
#         "failed_comparisons": len(all_failed_comparisons),
#         "aggregated_stats": {},
#         "num_gpus_used": num_gpus
#     }

#     if hallucination_counts:
#         aggregated_results["aggregated_stats"] = {
#             "mean_hallucination_count": sum(hallucination_counts) / len(hallucination_counts),
#             "min_hallucination_count": min(hallucination_counts),
#             "max_hallucination_count": max(hallucination_counts),
#             "median_hallucination_count": sorted(hallucination_counts)[len(hallucination_counts) // 2],
#             "severity_distribution": severity_counts,
#             "hallucination_distribution": {
#                 "0-2 (Minimal)": len([c for c in hallucination_counts if 0 <= c <= 2]),
#                 "3-4 (Minor)": len([c for c in hallucination_counts if 3 <= c <= 4]),
#                 "5-6 (Moderate)": len([c for c in hallucination_counts if 5 <= c <= 6]),
#                 "7-8 (Severe)": len([c for c in hallucination_counts if 7 <= c <= 8]),
#                 "9-10 (Critical)": len([c for c in hallucination_counts if 9 <= c <= 10])
#             }
#         }

#     return {
#         "aggregated_results": aggregated_results,
#         "individual_results": all_results,
#         "failed_log": all_failed_comparisons
#     }


# def parse_arguments():
#     """Parse command-line arguments."""
#     parser = argparse.ArgumentParser(
#         description="QWEN LLM Judge Evaluation with Multi-GPU Support and Thinking Mode",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
    
#     parser.add_argument(
#         "--input_file",
#         type=str,
#         default="/path/to/project/results/qwen2audio_trial_list_results.json",
#         help="Path to the input JSON file containing utterances with metadata and generated captions"
#     )
    
#     parser.add_argument(
#         "--output_dir",
#         type=str,
#         default="/path/to/project/results/",
#         help="Directory where the output results file will be saved"
#     )
    
#     parser.add_argument(
#         "--output_file",
#         type=str,
#         default="hallucination_evaluation_results_thinking.json",
#         help="Name of the output JSON file to save evaluation results"
#     )
    
#     parser.add_argument(
#         "--batch_size",
#         type=int,
#         default=4,
#         help="Batch size for processing (may want to reduce for thinking mode due to longer generations)"
#     )
    
#     parser.add_argument(
#         "--model_name",
#         type=str,
#         default="Qwen/Qwen3-32B",
#         help="Name or path of the Qwen model to use for evaluation"
#     )
    
#     parser.add_argument(
#         "--num_gpus",
#         type=int,
#         default=None,
#         help="Number of GPUs to use (default: use all available GPUs)"
#     )
    
#     return parser.parse_args()


# if __name__ == "__main__":
#     # Required for multiprocessing with CUDA
#     mp.set_start_method('spawn', force=True)
    
#     args = parse_arguments()
    
#     available_gpus = get_available_gpus()
#     if available_gpus == 0:
#         print("No GPUs available. Please check your CUDA installation.")
#         exit(1)
    
#     # Use specified number of GPUs or all available
#     num_gpus = args.num_gpus if args.num_gpus else available_gpus
#     num_gpus = min(num_gpus, available_gpus)  # Don't exceed available GPUs
    
#     print(f"Found {available_gpus} GPU(s) available, using {num_gpus} GPU(s)")
    
#     print("Starting hallucination evaluation with Qwen model using multi-GPU and thinking mode... 🚀")
    
#     results = process_all_utterances_multi_gpu(
#         input_file=args.input_file,
#         model_name=args.model_name,
#         batch_size=args.batch_size,
#         num_gpus=num_gpus
#     )

#     print("\n--- 📊 Final Hallucination Evaluation Summary (Multi-GPU, Thinking Mode) ---")
#     print(json.dumps(results['aggregated_results'], indent=2))

#     summary_file = os.path.join(args.output_dir, args.output_file)
#     with open(summary_file, 'w') as f:
#         json.dump(results, f, indent=2)
#     print(f"\nFull results saved to {summary_file}")

#     print("\nEvaluation complete! ✨")



#!/usr/bin/env python3
"""
Qwen Hallucination Judge — Multi-GPU Parallel Implementation
Clean, stable, production-ready version.

Each GPU loads its own full Qwen model and processes its assigned samples.
No model sharding. True parallelism with multiprocessing.
"""
import sys
sys.stdout.flush()

import json, re, os, argparse
import torch
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
#                    GLOBAL TEMPLATE
# ============================================================


PROMPT_TEMPLATE = """You are an expert evaluator assessing the quality of automatically generated audio captions. Your task is to compare metadata from audio files with generated captions to identify and score hallucinations.

## Task Overview
Compare the **metadata** (ground truth information about the audio) with a **generated caption** and evaluate the presence and severity of hallucinations.

## Input Format

Ground Truth Metadata:
<ground_truth_metadata>
{data1}
</ground_truth_metadata>

Generated Caption:
<generated_caption>
{data2}
</generated_caption>

## Evaluation Criteria

### What Counts as a Hallucination?
A hallucination is information in the generated caption that:
- Directly contradicts the metadata
- Invents specific details not supported by any evidence
- Misrepresents factual characteristics (e.g., wrong gender, wrong emotion, wrong accent)

### What Does NOT Count as a Hallucination?
The following are **acceptable creative interpretations** and should NOT be penalized:
- **Reasonable emotional inferences**: If metadata shows "laughing" audio, describing the speaker as "cheerful," "animated," or "enthusiastic" is acceptable
- **Stylistic embellishments**: Descriptive language like "her voice carries warmth," "delivered with precision," or "words tumble out" that enhance the description without contradicting facts
- **Contextual scenarios**: Adding reasonable context (e.g., "as if addressing an audience," "in a quiet environment") when supported by acoustic characteristics
- **Prosodic interpretations**: Descriptions of rhythm, cadence, emphasis, or delivery style that are reasonable given the speech characteristics
- **Minor elaborations**: Details that naturally extend from the metadata without contradicting it

### Hallucination Examples

**SEVERE Hallucinations:**
- Metadata indicates male speaker → Caption says "female speaker"
- Metadata shows speaker p026 → Caption invents specific name or identity
- Metadata indicates disgust emotion → Caption describes joy or happiness
- Inventing specific events or quoted speech not in the audio

**MODERATE Hallucinations:**
- Claiming "noisy environment" when metadata suggests clean audio
- Describing specific background sounds not mentioned in metadata
- Asserting definitive context (e.g., "delivering a news broadcast") without support

**MINOR Hallucinations:**
- Slightly exaggerating a characteristic (e.g., "booming voice" for a moderately loud voice)
- Adding plausible but unverified details that don't contradict metadata

## Scoring Instructions

Provide two scores:

### 1. Hallucination Count (0-10 scale)
- **0-2**: No hallucinations or only very minor embellishments
- **3-4**: 1-2 minor hallucinations that don't significantly distort the audio
- **5-6**: Multiple minor hallucinations OR 1 moderate hallucination
- **7-8**: Multiple moderate hallucinations OR 1 severe hallucination
- **9-10**: Multiple severe hallucinations that fundamentally misrepresent the audio

### 2. Severity Level
- **Low**: Only minor embellishments; caption is largely accurate
- **Medium**: Some factual errors that partially misrepresent the audio
- **High**: Severe contradictions or fabrications that fundamentally mischaracterize the audio

## Instructions
1. In your thinking block, carefully analyze the metadata and generated caption step by step:
   - List all factual claims made in the generated caption
   - For each claim, check if it's supported by, contradicts, or extends beyond the metadata
   - Categorize each issue as minor, moderate, or severe
   - Count the total number of hallucinations
2. Be lenient with creative, descriptive language that doesn't contradict facts
3. Focus on factual accuracy regarding speaker identity, emotions, accent, and acoustic characteristics
4. Distinguish between reasonable inference and baseless fabrication

## Output Format

After your thinking, provide your evaluation in the following structure:

<analysis>
[Brief explanation of what you observed, noting any hallucinations found and why they are/aren't problematic]
</analysis>

<justification>
[Explain your scoring in detail, referencing specific examples from the caption and how they relate to the metadata]
</justification>

<score>
{{"hallucination_count": X, "severity": "Low/Medium/High"}}
</score>

Remember: Context is important - "laughing" audio naturally suggests positive emotions. Be strict about factual contradictions but lenient about creative descriptions that align with the metadata.

Your final output should consist only of the analysis, justification and score, without duplicating the detailed step-by-step analysis from your thinking block."""



# ============================================================
#                    UTILITY FUNCTIONS
# ============================================================

def get_available_gpus():
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def safe_json(obj):
    try:
        return json.dumps(obj, indent=2)
    except:
        return str(obj)


# ============================================================
#       LOAD QWEN MODEL ON A SPECIFIC GPU (WORKER SIDE)
# ============================================================

def load_qwen_on_gpu(model_name, gpu_id):
    """Load tokenizer + full model on a specific GPU."""
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"

    print(f"[GPU {gpu_id}] Loading Qwen model on {device} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": gpu_id},      # Force FULL model onto this GPU
        torch_dtype=torch.bfloat16
    )
    model.eval()

    print(f"[GPU {gpu_id}] Model loaded!")
    return tokenizer, model, device


# ============================================================
#                       INFERENCE
# ============================================================

def run_batch(items, tokenizer, model, device):
    """Run hallucination evaluation on a batch of examples."""
    prompts = [
        PROMPT_TEMPLATE.format(data1=x["data1"], data2=x["data2"])
        for x in items
    ]

    # Build chat-style text
    messages = [
        [{"role": "user", "content": p}]
        for p in prompts
    ]

    formatted = [
        tokenizer.apply_chat_template(m, tokenize=False, enable_thinking=True, add_generation_prompt=True)
        for m in messages
    ]

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True
    ).to(device)

    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=False,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
    )

    cut = inputs.input_ids.shape[1]
    outputs = outputs[:, cut:]

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch_results = []
    for resp in decoded:
        # ---------------- PARSE SCORE ----------------
        halluc_count = 0
        severity = "Low"

        score_match = re.search(r"<score>\s*(\{.*?\})\s*</score>", resp, re.DOTALL)
        if score_match:
            try:
                js = json.loads(score_match.group(1))
                halluc_count = js.get("hallucination_count", 0)
                severity = js.get("severity", "Low")
            except:
                pass

        # ---------------- PARSE JUSTIFICATION ----------------
        just_match = re.search(r"<justification>(.*?)</justification>", resp, re.DOTALL)
        analysis_match = re.search(r"<analysis>(.*?)</analysis>", resp, re.DOTALL)

        justification = just_match.group(1).strip() if just_match else ""
        analysis = analysis_match.group(1).strip() if analysis_match else ""

        batch_results.append({
            "hallucination_count": halluc_count,
            "severity": severity,
            "justification": justification,
            "analysis": analysis,
            "full_response": resp
        })

    torch.cuda.empty_cache()
    return batch_results


# ============================================================
#              WORKER FUNCTION (RUNS ON GPU K)
# ============================================================

def worker_run_chunk(gpu_id, chunk_data, chunk_id, model_name, batch_size):
    """Executed inside each GPU worker process."""
    try:
        tokenizer, model, device = load_qwen_on_gpu(model_name, gpu_id)
    except Exception as e:
        print(f"[GPU {gpu_id}] ERROR loading model: {e}")
        return [], [{"gpu": gpu_id, "reason": str(e)}]

    results, failed = [], []

    for i in range(0, len(chunk_data), batch_size):
        batch = chunk_data[i:i + batch_size]

        try:
            outputs = run_batch(batch, tokenizer, model, device)

            for j, out in enumerate(outputs):
                meta = batch[j]
                results.append({
                    "index": meta["index"],
                    "gpu_id": gpu_id,
                    "chunk_id": chunk_id,
                    "wav_path": meta["wav_path"],
                    "hallucination_count": out["hallucination_count"],
                    "severity": out["severity"],
                    "justification": out["justification"],
                    "analysis": out.get("analysis", ""),
                    "ground_truth_metadata": meta["data1"],
                    "generated_caption_metadata": meta["data2"],
                    "full_response": out["full_response"]
                })

        except Exception as e:
            for meta in batch:
                failed.append({
                    "index": meta["index"],
                    "gpu_id": gpu_id,
                    "reason": f"Batch error: {e}"
                })

    del model
    torch.cuda.empty_cache()

    return results, failed


# ============================================================
#         MAIN MULTI-GPU EVALUATION COORDINATION
# ============================================================

def run_multi_gpu_eval(input_file, model_name, batch_size, num_gpus):
    """Split data and run hallucination evaluation across GPUs."""

    with open(input_file, "r") as f:
        raw = json.load(f)

    # Build evaluation pairs
    data = []
    for i, u in enumerate(raw):
        if "metadata" not in u or "generated_captions" not in u:
            continue

        cap = u["generated_captions"].get("holistic_creative_synthesis", {}).get("generated_caption", "")
        if not cap:
            for cap_info in u["generated_captions"].values():
                if isinstance(cap_info, dict) and cap_info.get("generated_caption"):
                    cap = cap_info["generated_caption"]
                    break
        meta = dict(u["metadata"])
        if "ground_truth_caption" not in meta and u.get("ground_truth_caption"):
            meta["ground_truth_caption"] = u["ground_truth_caption"]
        data.append({
            "index": i,
            "wav_path": u.get("wav_path", ""),
            "data1": json.dumps(meta, indent=2),
            "data2": cap
        })

    total = len(data)
    print(f"\nTotal examples: {total}\n")

    # Split across GPUs
    chunks = []
    size = total // num_gpus

    for g in range(num_gpus):
        start = g * size
        end = start + size if g < num_gpus - 1 else total
        chunk = data[start:end]
        chunks.append(chunk)
        print(f"GPU {g}: {len(chunk)} items")

    # If only 1 GPU → run directly (no multiprocessing)
    if num_gpus == 1:
        print("\nRunning in single-GPU direct mode (no multiprocessing).")
        results, failed = worker_run_chunk(
            0, chunks[0], 0, model_name, batch_size
        )
        return summarize_results(results, failed, total), results, failed

    # MULTI-GPU PARALLEL MODE
    print("\nLaunching multiprocessing across GPUs...\n")
    mp.set_start_method("spawn", force=True)

    with mp.Pool(processes=num_gpus) as pool:
        func = partial(worker_run_chunk,
                       model_name=model_name,
                       batch_size=batch_size)

        args = [(g, chunks[g], g) for g in range(num_gpus)]
        outputs = pool.starmap(func, args)

    all_results, all_failed = [], []
    for r, f in outputs:
        all_results.extend(r)
        all_failed.extend(f)

    all_results.sort(key=lambda x: x["index"])

    return summarize_results(all_results, all_failed, total), all_results, all_failed


# ============================================================
#                       SUMMARY BUILDER
# ============================================================

def summarize_results(results, failed, total):
    halluc_counts = [r["hallucination_count"] for r in results]

    severity_dist = {"Low": 0, "Medium": 0, "High": 0}
    for r in results:
        severity_dist[r["severity"]] += 1

    summary = {
        "total_samples": total,
        "successful": len(results),
        "failed": len(failed),
        "mean_hallucination_count": sum(halluc_counts)/len(halluc_counts) if halluc_counts else 0,
        "severity_distribution": severity_dist,
    }
    return summary


# ============================================================
#                       COMMAND LINE
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen3-32B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--num_gpus", type=int, default=None)
    return parser.parse_args()


# ============================================================
#                           MAIN
# ============================================================

if __name__ == "__main__":
    args = parse_args()

    available = get_available_gpus()
    if available == 0:
        raise RuntimeError("❌ No CUDA GPUs available")

    num = args.num_gpus or available
    num = min(num, available)

    print(f"Using {num}/{available} GPUs")

    summary, results, failed = run_multi_gpu_eval(
        args.input_file,
        args.model_name,
        args.batch_size,
        num
    )

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    os.makedirs(args.output_dir, exist_ok=True)
    out = {
        "summary": summary,
        "results": results,
        "failed": failed
    }
    dest = os.path.join(args.output_dir, args.output_file)
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved results → {dest}")
