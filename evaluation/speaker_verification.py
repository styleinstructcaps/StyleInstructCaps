#!/usr/bin/env python3
"""
Speaker Verification System for Captions

This script implements a complete speaker verification pipeline:
1. Loads caption data from JSON file
2. Creates verification trials (positive: same speaker different utterances, negative: different speakers)
3. Computes text embeddings using e5-base-v2
4. Calculates cosine similarities for all trials
5. Evaluates using EER, minDCF, DCF0.8, DCF1.0, and AUC metrics

Supports loading/saving trials and embeddings for efficiency.
Includes sanity check mode for quick testing (320 positive, 320 negative trials).
"""

import json
import os
import argparse
import warnings
import random
from typing import List, Tuple, Dict, Optional
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_auc_score, roc_curve

from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# =============================================================================
# Data Loading and Processing
# =============================================================================

def load_json_data(input_json: str) -> List[Dict]:
    """Load JSON file and return data list."""
    print(f"Loading JSON file: {input_json}")
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle both formats: dict with "annotation" key or direct list
    if isinstance(data, dict) and "annotation" in data:
        data = data["annotation"]
    
    print(f"Loaded {len(data)} entries")
    return data


def extract_captions_per_utterance(data: List[Dict]) -> Dict[str, Dict[str, str]]:
    """
    Extract first caption per utterance, grouped by speaker.
    
    Returns:
        Dictionary: {speaker_id: {utterance_id: caption}}
    """
    speaker_to_utterances = {}
    
    for entry in data:
        # Extract speaker ID
        speaker_id = entry.get("metadata", {}).get("speakerid")
        if not speaker_id:
            speaker_id = entry.get("speakerid")
        
        if not speaker_id:
            continue
        
        # Extract utterance ID
        utterance_id = (entry.get("wav_path") or 
                      entry.get("relative_audio_path") or 
                      entry.get("audio_path"))
        
        if not utterance_id:
            continue
        
        # Extract first caption from generated_captions
        generated_captions = entry.get("generated_captions", {})
        if not isinstance(generated_captions, dict):
            continue
        
        # Get first caption (sorted by key to ensure consistency)
        first_caption = None
        for prompt_key in sorted(generated_captions.keys()):
            caption_info = generated_captions[prompt_key]
            if isinstance(caption_info, dict):
                caption = caption_info.get("generated_caption", "").strip()
                if caption:
                    # Clean caption
                    caption = caption.replace("<s>", "").replace("</s>", "").strip()
                    first_caption = caption
                    break
        
        if first_caption:
            if speaker_id not in speaker_to_utterances:
                speaker_to_utterances[speaker_id] = {}
            speaker_to_utterances[speaker_id][utterance_id] = first_caption
    
    print(f"Extracted data for {len(speaker_to_utterances)} speakers")
    total_utterances = sum(len(utts) for utts in speaker_to_utterances.values())
    print(f"Total utterances: {total_utterances}")
    
    return speaker_to_utterances


# =============================================================================
# Trial Generation
# =============================================================================

def create_positive_trials(
    speaker_to_utterances: Dict[str, Dict[str, str]],
    num_positive: int,
    equal_per_speaker: bool = True
) -> List[Dict]:
    """
    Create positive trials: same speaker, different utterances.
    Each utterance is used as utterance_id1, paired with a different utterance 
    from the same speaker as utterance_id2. If more trials are needed than 
    available utterances, utterances will be reused.
    
    Args:
        speaker_to_utterances: Dictionary mapping speaker_id to {utterance_id: caption}
        num_positive: Total number of positive trials to create
        equal_per_speaker: If True, distribute trials equally across speakers. 
                          If False, sample randomly from all possible pairs.
    
    Returns:
        List of trial dictionaries with keys: speaker_id, utterance_id1, utterance_id2, caption1, caption2, label
    """
    positive_trials = []
    
    # Collect all speakers with at least 2 utterances
    valid_speakers = {
        spk: utts for spk, utts in speaker_to_utterances.items() 
        if len(utts) >= 2
    }
    
    if not valid_speakers:
        raise ValueError("No speakers with at least 2 utterances found")
    
    num_speakers = len(valid_speakers)
    print(f"Creating {num_positive} positive trials from {num_speakers} speakers")
    
    if equal_per_speaker:
        # Ensure equal distribution across speakers
        trials_per_speaker = num_positive // num_speakers
        remainder = num_positive % num_speakers
        
        print(f"  Distributing {trials_per_speaker} trials per speaker (with {remainder} extra)")
        
        # Collect all utterances with their speakers for selection
        all_utterances = []  # List of (speaker_id, utterance_id)
        for speaker_id, utterances in valid_speakers.items():
            for utt_id in utterances.keys():
                all_utterances.append((speaker_id, utt_id))
        
        # Shuffle to randomize selection
        random.shuffle(all_utterances)
        
        # Select utterances for utterance_id1, ensuring equal distribution per speaker
        selected_utt1_list = []
        speaker_list = list(valid_speakers.keys())
        random.shuffle(speaker_list)
        
        # Distribute trials per speaker
        for idx, speaker_id in enumerate(speaker_list):
            num_trials_for_speaker = trials_per_speaker
            if idx < remainder:
                num_trials_for_speaker += 1
            
            # Get utterances for this speaker
            speaker_utts = [(spk, utt) for spk, utt in all_utterances if spk == speaker_id]
            random.shuffle(speaker_utts)
            
            # Select exactly num_trials_for_speaker utterances for this speaker
            # If we need more trials than available utterances, allow reuse
            if len(speaker_utts) >= num_trials_for_speaker:
                selected_utt1_list.extend(speaker_utts[:num_trials_for_speaker])
            else:
                # Reuse utterances if needed
                selected_utt1_list.extend(speaker_utts)
                # Fill remaining with random selection (with replacement)
                remaining = num_trials_for_speaker - len(speaker_utts)
                selected_utt1_list.extend(random.choices(speaker_utts, k=remaining))
        
        # Shuffle the final list
        random.shuffle(selected_utt1_list)
        
        # For each selected utterance as utterance_id1, pair with a different utterance from same speaker
        for speaker_id, utt1 in selected_utt1_list:
            # Get all other utterances from the same speaker
            other_utts = [utt for utt in valid_speakers[speaker_id].keys() if utt != utt1]
            if not other_utts:
                raise ValueError(f"Speaker {speaker_id} has no other utterances to pair with {utt1}")
            
            # Randomly select one as utterance_id2
            utt2 = random.choice(other_utts)
            
            caption1 = valid_speakers[speaker_id][utt1]
            caption2 = valid_speakers[speaker_id][utt2]
            
            trial = {
                "speaker_id": speaker_id,
                "utterance_id1": utt1,
                "utterance_id2": utt2,
                "caption1": caption1,
                "caption2": caption2,
                "label": 1
            }
            positive_trials.append(trial)
        
        # Verify distribution
        speaker_counts = {}
        utterance_id1_counts = {}
        for trial in positive_trials:
            spk = trial["speaker_id"]
            speaker_counts[spk] = speaker_counts.get(spk, 0) + 1
            utt1 = trial["utterance_id1"]
            utterance_id1_counts[utt1] = utterance_id1_counts.get(utt1, 0) + 1
        
        print(f"  Trial distribution per speaker:")
        for spk, count in sorted(speaker_counts.items()):
            print(f"    {spk}: {count} trials")
        
        # Show utterance_id1 usage statistics
        max_utt1_usage = max(utterance_id1_counts.values()) if utterance_id1_counts else 0
        min_utt1_usage = min(utterance_id1_counts.values()) if utterance_id1_counts else 0
        mean_utt1_usage = sum(utterance_id1_counts.values()) / len(utterance_id1_counts) if utterance_id1_counts else 0
        print(f"  Utterance_id1 usage: min={min_utt1_usage}, max={max_utt1_usage}, mean={mean_utt1_usage:.2f}")
    
    else:
        # Original behavior: sample randomly from all pairs
        # Generate all possible positive pairs
        all_positive_pairs = []
        for speaker_id, utterances in valid_speakers.items():
            utterance_ids = list(utterances.keys())
            # Generate all pairs of different utterances for this speaker
            for utt1, utt2 in combinations(utterance_ids, 2):
                all_positive_pairs.append((speaker_id, utt1, utt2))
        
        # Sample to get exactly num_positive trials
        if len(all_positive_pairs) >= num_positive:
            sampled_pairs = random.sample(all_positive_pairs, num_positive)
        else:
            # If not enough pairs, sample with replacement
            sampled_pairs = random.choices(all_positive_pairs, k=num_positive)
        
        # Create trial dictionaries
        for speaker_id, utt1, utt2 in sampled_pairs:
            caption1 = speaker_to_utterances[speaker_id][utt1]
            caption2 = speaker_to_utterances[speaker_id][utt2]
            
            trial = {
                "speaker_id": speaker_id,
                "utterance_id1": utt1,
                "utterance_id2": utt2,
                "caption1": caption1,
                "caption2": caption2,
                "label": 1
            }
            positive_trials.append(trial)
    
    print(f"Created {len(positive_trials)} positive trials")
    return positive_trials


def create_negative_trials(
    speaker_to_utterances: Dict[str, Dict[str, str]],
    num_negative: int,
    equal_per_speaker: bool = True
) -> List[Dict]:
    """
    Create negative trials: different speakers.
    Ensures equal distribution per speaker and balanced utterance usage.
    
    Args:
        speaker_to_utterances: Dictionary mapping speaker_id to {utterance_id: caption}
        num_negative: Total number of negative trials to create
        equal_per_speaker: If True, distribute trials equally across speakers.
                          If False, sample randomly from all possible pairs.
    
    Returns:
        List of trial dictionaries with keys: speaker_id1, speaker_id2, utterance_id1, utterance_id2, caption1, caption2, label
    """
    negative_trials = []
    
    speakers = list(speaker_to_utterances.keys())
    if len(speakers) < 2:
        raise ValueError("Need at least 2 speakers for negative trials")
    
    num_speakers = len(speakers)
    print(f"Creating {num_negative} negative trials from {num_speakers} speakers")
    
    if equal_per_speaker:
        # Ensure equal distribution across speakers
        # Each speaker should appear in roughly equal number of negative trials
        # Since each trial involves 2 speakers, we need to think about this differently
        # We'll ensure each speaker contributes equally as speaker_id1
        
        trials_per_speaker = num_negative // num_speakers
        remainder = num_negative % num_speakers
        
        print(f"  Distributing {trials_per_speaker} trials per speaker as primary speaker (with {remainder} extra)")
        
        # Track utterance usage for negative trials
        utterance_usage = {spk: {utt: 0 for utt in speaker_to_utterances[spk].keys()} 
                          for spk in speakers}
        
        speaker_list = list(speakers)
        random.shuffle(speaker_list)
        
        # Create negative trials ensuring each speaker appears equally
        seen_pairs = set()
        
        for idx, spk1 in enumerate(speaker_list):
            # Calculate how many trials this speaker should be involved in as primary
            num_trials_for_speaker = trials_per_speaker
            if idx < remainder:
                num_trials_for_speaker += 1
            
            # Get utterances for this speaker
            utts1 = list(speaker_to_utterances[spk1].keys())
            random.shuffle(utts1)
            
            # Get other speakers to pair with
            other_speakers = [spk for spk in speakers if spk != spk1]
            random.shuffle(other_speakers)
            
            trials_created = 0
            utt_idx = 0
            other_spk_idx = 0
            max_attempts_per_speaker = num_trials_for_speaker * 100
            attempts = 0
            
            while trials_created < num_trials_for_speaker and attempts < max_attempts_per_speaker:
                attempts += 1
                
                # Cycle through utterances of primary speaker
                if utt_idx >= len(utts1):
                    utt_idx = 0
                    random.shuffle(utts1)  # Reshuffle for next cycle
                
                utt1 = utts1[utt_idx]
                
                # Cycle through other speakers
                if other_spk_idx >= len(other_speakers):
                    other_spk_idx = 0
                    random.shuffle(other_speakers)
                
                spk2 = other_speakers[other_spk_idx]
                utts2 = list(speaker_to_utterances[spk2].keys())
                
                # Select utterance from spk2 with lowest usage
                utts2_sorted = sorted(utts2, key=lambda u: utterance_usage[spk2][u])
                utt2 = utts2_sorted[0]
                
                # Create ordered pair to avoid duplicates
                pair_key = tuple(sorted([(spk1, utt1), (spk2, utt2)]))
                
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    caption1 = speaker_to_utterances[spk1][utt1]
                    caption2 = speaker_to_utterances[spk2][utt2]
                    
                    trial = {
                        "speaker_id1": spk1,
                        "speaker_id2": spk2,
                        "utterance_id1": utt1,
                        "utterance_id2": utt2,
                        "caption1": caption1,
                        "caption2": caption2,
                        "label": 0
                    }
                    negative_trials.append(trial)
                    
                    utterance_usage[spk1][utt1] += 1
                    utterance_usage[spk2][utt2] += 1
                    trials_created += 1
                    utt_idx += 1
                    other_spk_idx += 1
                else:
                    # Try next utterance from spk2
                    found = False
                    for next_utt2 in utts2_sorted[1:]:
                        pair_key = tuple(sorted([(spk1, utt1), (spk2, next_utt2)]))
                        if pair_key not in seen_pairs:
                            seen_pairs.add(pair_key)
                            caption1 = speaker_to_utterances[spk1][utt1]
                            caption2 = speaker_to_utterances[spk2][next_utt2]
                            
                            trial = {
                                "speaker_id1": spk1,
                                "speaker_id2": spk2,
                                "utterance_id1": utt1,
                                "utterance_id2": next_utt2,
                                "caption1": caption1,
                                "caption2": caption2,
                                "label": 0
                            }
                            negative_trials.append(trial)
                            
                            utterance_usage[spk1][utt1] += 1
                            utterance_usage[spk2][next_utt2] += 1
                            trials_created += 1
                            utt_idx += 1
                            other_spk_idx += 1
                            found = True
                            break
                    
                    if not found:
                        # Try next utterance from primary speaker
                        utt_idx += 1
                        if utt_idx >= len(utts1):
                            utt_idx = 0
                            other_spk_idx += 1
                            if other_spk_idx >= len(other_speakers):
                                other_spk_idx = 0
                                # If we've cycled through everything, allow duplicates
                                utt2 = random.choice(utts2)
                                pair_key = tuple(sorted([(spk1, utt1), (spk2, utt2)]))
                                seen_pairs.add(pair_key)
                                caption1 = speaker_to_utterances[spk1][utt1]
                                caption2 = speaker_to_utterances[spk2][utt2]
                                
                                trial = {
                                    "speaker_id1": spk1,
                                    "speaker_id2": spk2,
                                    "utterance_id1": utt1,
                                    "utterance_id2": utt2,
                                    "caption1": caption1,
                                    "caption2": caption2,
                                    "label": 0
                                }
                                negative_trials.append(trial)
                                
                                utterance_usage[spk1][utt1] += 1
                                utterance_usage[spk2][utt2] += 1
                                trials_created += 1
                                utt_idx += 1
                                other_spk_idx += 1
            
            if trials_created < num_trials_for_speaker:
                print(f"  Warning: Only created {trials_created}/{num_trials_for_speaker} trials for speaker {spk1}")
        
        # Verify distribution
        speaker_counts = {}
        for trial in negative_trials:
            spk1 = trial["speaker_id1"]
            speaker_counts[spk1] = speaker_counts.get(spk1, 0) + 1
        
        print(f"  Trial distribution per speaker (as primary):")
        for spk, count in sorted(speaker_counts.items()):
            print(f"    {spk}: {count} trials")
        
        # Print utterance usage statistics
        print(f"  Utterance usage per speaker in negative trials (min, max, mean):")
        for spk in sorted(utterance_usage.keys()):
            usages = list(utterance_usage[spk].values())
            if usages:
                print(f"    {spk}: min={min(usages)}, max={max(usages)}, mean={sum(usages)/len(usages):.2f}")
    
    else:
        # Original behavior: sample randomly from all pairs
        seen_pairs = set()
        attempts = 0
        max_attempts = num_negative * 100
        
        while len(negative_trials) < num_negative and attempts < max_attempts:
            attempts += 1
            
            # Randomly select two different speakers
            spk1, spk2 = random.sample(speakers, 2)
            
            # Randomly select one utterance from each speaker
            utt1 = random.choice(list(speaker_to_utterances[spk1].keys()))
            utt2 = random.choice(list(speaker_to_utterances[spk2].keys()))
            
            # Create ordered pair to avoid duplicates
            pair_key = tuple(sorted([(spk1, utt1), (spk2, utt2)]))
            
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                caption1 = speaker_to_utterances[spk1][utt1]
                caption2 = speaker_to_utterances[spk2][utt2]
                
                trial = {
                    "speaker_id1": spk1,
                    "speaker_id2": spk2,
                    "utterance_id1": utt1,
                    "utterance_id2": utt2,
                    "caption1": caption1,
                    "caption2": caption2,
                    "label": 0
                }
                negative_trials.append(trial)
        
        if len(negative_trials) < num_negative:
            print(f"Warning: Only generated {len(negative_trials)} negative trials (requested {num_negative})")
    
    print(f"Created {len(negative_trials)} negative trials")
    return negative_trials


def save_trials_to_json(trials: List[Dict], output_file: str):
    """Save trials to JSON file."""
    print(f"Saving {len(trials)} trials to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(trials, f, indent=2, ensure_ascii=False)
    print("Trials saved successfully")


def load_trials_from_json(trials_file: str) -> List[Dict]:
    """Load trials from JSON file."""
    print(f"Loading trials from {trials_file}")
    with open(trials_file, "r", encoding="utf-8") as f:
        trials = json.load(f)
    print(f"Loaded {len(trials)} trials")
    return trials


# =============================================================================
# Embedding Computation
# =============================================================================

def average_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average pooling with attention mask."""
    masked = hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)


def compute_text_embeddings(
    captions: List[str],
    batch_size: int = 32,
    device: Optional[str] = None
) -> np.ndarray:
    """
    Compute text embeddings using e5-base-v2 model.
    
    Args:
        captions: List of caption strings
        batch_size: Batch size for processing
        device: Device to use (cuda/cpu), auto-detect if None
    
    Returns:
        numpy array of embeddings (num_captions, embedding_dim)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "intfloat/e5-base-v2"
    
    print(f"Loading model {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    # Add query prefix for e5-base-v2
    captions = [f"query: {c}" for c in captions]
    
    all_embeddings = []
    
    print(f"Computing embeddings for {len(captions)} captions in batches of {batch_size}")
    
    for i in range(0, len(captions), batch_size):
        batch = captions[i:i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            emb = average_pool(outputs.last_hidden_state, inputs["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)
        
        all_embeddings.append(emb.cpu().numpy())
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {i + len(batch)}/{len(captions)} captions")
    
    embeddings = np.vstack(all_embeddings)
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def save_embeddings(
    embeddings: np.ndarray,
    caption_to_idx: Dict[Tuple[str, str], int],
    output_file: str,
    speaker_ids: Optional[List[str]] = None,
    utterance_ids: Optional[List[str]] = None
):
    """
    Save embeddings to NPZ file.
    
    Args:
        embeddings: numpy array of embeddings
        caption_to_idx: mapping from (speaker_id, utterance_id) to embedding index
        output_file: path to save NPZ file
        speaker_ids: optional list of speaker IDs
        utterance_ids: optional list of utterance IDs
    """
    print(f"Saving embeddings to {output_file}")
    
    save_dict = {
        "embeddings": embeddings,
        "caption_to_idx": caption_to_idx
    }
    
    if speaker_ids is not None:
        save_dict["speaker_ids"] = np.array(speaker_ids, dtype=object)
    if utterance_ids is not None:
        save_dict["utterance_ids"] = np.array(utterance_ids, dtype=object)
    
    np.savez(output_file, **save_dict, allow_pickle=True)
    print("Embeddings saved successfully")


def load_embeddings(embeddings_file: str) -> Tuple[np.ndarray, Dict[Tuple[str, str], int]]:
    """
    Load embeddings from NPZ file.
    
    Returns:
        (embeddings, caption_to_idx)
    """
    print(f"Loading embeddings from {embeddings_file}")
    data = np.load(embeddings_file, allow_pickle=True)
    
    embeddings = data["embeddings"]
    caption_to_idx = data["caption_to_idx"].item() if isinstance(data["caption_to_idx"].item(), dict) else {}
    
    print(f"Loaded embeddings shape: {embeddings.shape}")
    print(f"Loaded {len(caption_to_idx)} caption mappings")
    
    return embeddings, caption_to_idx


# =============================================================================
# Similarity Computation
# =============================================================================

def compute_cosine_similarities(
    trials: List[Dict],
    embeddings: np.ndarray,
    caption_to_idx: Dict[Tuple[str, str], int]
) -> np.ndarray:
    """
    Compute cosine similarity for all trials.
    
    Args:
        trials: List of trial dictionaries
        embeddings: numpy array of embeddings
        caption_to_idx: mapping from (speaker_id, utterance_id) to embedding index
    
    Returns:
        numpy array of similarity scores
    """
    similarities = []
    
    print(f"Computing cosine similarities for {len(trials)} trials")
    
    for i, trial in enumerate(trials):
        # Get embedding indices
        if trial["label"] == 1:
            # Positive trial: same speaker
            speaker_id = trial["speaker_id"]
            utt1 = trial["utterance_id1"]
            utt2 = trial["utterance_id2"]
            idx1 = caption_to_idx.get((speaker_id, utt1))
            idx2 = caption_to_idx.get((speaker_id, utt2))
        else:
            # Negative trial: different speakers
            spk1 = trial["speaker_id1"]
            spk2 = trial["speaker_id2"]
            utt1 = trial["utterance_id1"]
            utt2 = trial["utterance_id2"]
            idx1 = caption_to_idx.get((spk1, utt1))
            idx2 = caption_to_idx.get((spk2, utt2))
        
        if idx1 is None or idx2 is None:
            raise ValueError(f"Could not find embedding indices for trial {i}")
        
        # Compute cosine similarity
        emb1 = embeddings[idx1]
        emb2 = embeddings[idx2]
        sim = 1 - cosine(emb1, emb2)
        similarities.append(sim)
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(trials)} trials")
    
    similarities = np.array(similarities)
    print(f"Similarity scores range: [{similarities.min():.4f}, {similarities.max():.4f}]")
    print(f"Mean similarity: {similarities.mean():.4f}")
    
    return similarities


# =============================================================================
# Verification Metrics
# =============================================================================

def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER).
    
    Returns:
        (EER, threshold)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    
    # Find threshold where FPR = FNR
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_idx]
    eer_threshold = thresholds[eer_idx]
    
    return eer, eer_threshold


def compute_dcf(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
    p_target: float = 0.01
) -> float:
    """
    Compute Detection Cost Function (DCF).
    
    Args:
        y_true: True labels (1 for positive, 0 for negative)
        y_scores: Similarity scores
        c_miss: Cost of miss (false negative)
        c_fa: Cost of false alarm (false positive)
        p_target: Prior probability of target
    
    Returns:
        Minimum DCF value
    """
    # Sort scores in descending order
    sorted_indices = np.argsort(y_scores)[::-1]
    sorted_scores = y_scores[sorted_indices]
    sorted_labels = y_true[sorted_indices]
    
    # Compute cumulative statistics
    n_targets = np.sum(y_true == 1)
    n_nontargets = np.sum(y_true == 0)
    
    if n_targets == 0 or n_nontargets == 0:
        return float('inf')
    
    # Compute DCF for each threshold
    min_dcf = float('inf')
    
    for threshold in sorted_scores:
        # Predictions: 1 if score >= threshold, 0 otherwise
        predictions = (sorted_scores >= threshold).astype(int)
        
        # Miss (false negative): target but predicted as non-target
        n_miss = np.sum((sorted_labels == 1) & (predictions == 0))
        p_miss = n_miss / n_targets if n_targets > 0 else 0.0
        
        # False alarm (false positive): non-target but predicted as target
        n_fa = np.sum((sorted_labels == 0) & (predictions == 1))
        p_fa = n_fa / n_nontargets if n_nontargets > 0 else 0.0
        
        # DCF = C_miss * P_miss * P_target + C_fa * P_fa * (1 - P_target)
        dcf = c_miss * p_miss * p_target + c_fa * p_fa * (1 - p_target)
        min_dcf = min(min_dcf, dcf)
    
    return min_dcf


def compute_all_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
    """
    Compute all verification metrics.
    
    Returns:
        Dictionary with metrics: AUC, EER, minDCF, DCF08, DCF10
    """
    metrics = {}
    
    # AUC
    try:
        metrics["AUC"] = roc_auc_score(y_true, y_scores)
    except ValueError:
        metrics["AUC"] = 0.0
    
    # EER
    eer, eer_threshold = compute_eer(y_true, y_scores)
    metrics["EER"] = eer
    metrics["EER_threshold"] = eer_threshold
    
    # minDCF (C_miss=1, C_fa=1, P_target=0.01)
    metrics["minDCF"] = compute_dcf(y_true, y_scores, c_miss=1.0, c_fa=1.0, p_target=0.01)
    
    # DCF08 (NIST SRE 2008 style: C_miss=1, C_fa=1, P_target=0.01)
    # Assumes 1 in 100 trials is a target speaker
    metrics["DCF08"] = compute_dcf(y_true, y_scores, c_miss=1.0, c_fa=1.0, p_target=0.01)
    
    # DCF10 (NIST SRE 2010 style: C_miss=1, C_fa=1, P_target=0.001)
    # Assumes 1 in 1000 trials is a target speaker (stricter operating point)
    metrics["DCF10"] = compute_dcf(y_true, y_scores, c_miss=1.0, c_fa=1.0, p_target=0.001)
    
    return metrics


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Speaker Verification System for Captions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Input JSON file with captions"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--trials_file",
        type=str,
        default=None,
        help="Path to trials JSON file (optional, for loading existing trials)"
    )
    parser.add_argument(
        "--embeddings_file",
        type=str,
        default=None,
        help="Path to embeddings NPZ file (optional, for loading existing embeddings)"
    )
    parser.add_argument(
        "--num_positive",
        type=int,
        default=4800,
        help="Number of positive trials"
    )
    parser.add_argument(
        "--num_negative",
        type=int,
        default=4800,
        help="Number of negative trials"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding computation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Default: auto-detect"
    )
    parser.add_argument(
        "--sanity_check",
        action="store_true",
        help="Enable sanity check mode (320 positive, 320 negative trials)"
    )
    parser.add_argument(
        "--equal_per_speaker",
        action="store_true",
        default=True,
        help="Distribute positive trials equally across speakers (default: True)"
    )
    parser.add_argument(
        "--no_equal_per_speaker",
        dest="equal_per_speaker",
        action="store_false",
        help="Disable equal distribution - sample randomly from all pairs"
    )
    
    args = parser.parse_args()
    
    # Handle sanity check mode
    if args.sanity_check:
        args.num_positive = 1280
        args.num_negative = 1280
        print("=" * 60)
        print("SANITY CHECK MODE: Using 320 positive and 320 negative trials")
        print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine output file names
    base_name = os.path.splitext(os.path.basename(args.input_json))[0]
    trials_output = args.trials_file or os.path.join(args.output_dir, f"{base_name}_trials.json")
    embeddings_output = args.embeddings_file or os.path.join(args.output_dir, f"{base_name}_embeddings.npz")
    similarities_output = os.path.join(args.output_dir, f"{base_name}_similarities.npy")
    metrics_output = os.path.join(args.output_dir, f"{base_name}_metrics.txt")
    
    # =====================================================================
    # Step 1: Load or Create Trials
    # =====================================================================
    print("\n" + "=" * 60)
    print("STEP 1: Loading/Creating Verification Trials")
    print("=" * 60)
    
    if args.trials_file and os.path.exists(args.trials_file):
        print(f"Loading trials from existing file: {args.trials_file}")
        all_trials = load_trials_from_json(args.trials_file)
    else:
        # Load data
        data = load_json_data(args.input_json)
        
        # Extract captions per utterance
        speaker_to_utterances = extract_captions_per_utterance(data)
        
        # Create trials
        positive_trials = create_positive_trials(
            speaker_to_utterances, 
            args.num_positive,
            equal_per_speaker=args.equal_per_speaker
        )
        negative_trials = create_negative_trials(
            speaker_to_utterances, 
            args.num_negative,
            equal_per_speaker=args.equal_per_speaker
        )
        
        all_trials = positive_trials + negative_trials
        random.shuffle(all_trials)
        
        # Save trials
        save_trials_to_json(all_trials, trials_output)
    
    print(f"\nTotal trials: {len(all_trials)}")
    num_pos = sum(1 for t in all_trials if t["label"] == 1)
    num_neg = sum(1 for t in all_trials if t["label"] == 0)
    print(f"  Positive: {num_pos}")
    print(f"  Negative: {num_neg}")
    
    # =====================================================================
    # Step 2: Compute or Load Embeddings
    # =====================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Computing/Loading Embeddings")
    print("=" * 60)
    
    if args.embeddings_file and os.path.exists(args.embeddings_file):
        print(f"Loading embeddings from existing file: {args.embeddings_file}")
        embeddings, caption_to_idx = load_embeddings(args.embeddings_file)
    else:
        # Collect all unique captions from trials
        unique_captions = {}
        caption_to_idx = {}
        
        for trial in all_trials:
            if trial["label"] == 1:
                speaker_id = trial["speaker_id"]
                utt1 = trial["utterance_id1"]
                utt2 = trial["utterance_id2"]
                
                key1 = (speaker_id, utt1)
                key2 = (speaker_id, utt2)
                
                if key1 not in unique_captions:
                    unique_captions[key1] = trial["caption1"]
                if key2 not in unique_captions:
                    unique_captions[key2] = trial["caption2"]
            else:
                spk1 = trial["speaker_id1"]
                spk2 = trial["speaker_id2"]
                utt1 = trial["utterance_id1"]
                utt2 = trial["utterance_id2"]
                
                key1 = (spk1, utt1)
                key2 = (spk2, utt2)
                
                if key1 not in unique_captions:
                    unique_captions[key1] = trial["caption1"]
                if key2 not in unique_captions:
                    unique_captions[key2] = trial["caption2"]
        
        # Create mapping and caption list
        captions_list = []
        for idx, (key, caption) in enumerate(sorted(unique_captions.items())):
            caption_to_idx[key] = idx
            captions_list.append(caption)
        
        print(f"Computing embeddings for {len(captions_list)} unique captions")
        
        # Compute embeddings
        embeddings = compute_text_embeddings(
            captions_list,
            batch_size=args.batch_size,
            device=args.device
        )
        
        # Save embeddings
        save_embeddings(
            embeddings,
            caption_to_idx,
            embeddings_output
        )
    
    # =====================================================================
    # Step 3: Compute Cosine Similarities
    # =====================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Computing Cosine Similarities")
    print("=" * 60)
    
    similarities = compute_cosine_similarities(all_trials, embeddings, caption_to_idx)
    
    # Save similarities
    np.save(similarities_output, similarities)
    print(f"Similarities saved to {similarities_output}")
    
    # =====================================================================
    # Step 4: Compute Metrics
    # =====================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Computing Verification Metrics")
    print("=" * 60)
    
    labels = np.array([trial["label"] for trial in all_trials])
    metrics = compute_all_metrics(labels, similarities)
    
    # Print metrics
    print("\n" + "=" * 60)
    print("VERIFICATION METRICS")
    print("=" * 60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    print("=" * 60)
    
    # Save metrics
    with open(metrics_output, "w") as f:
        f.write("Speaker Verification Metrics\n")
        f.write("=" * 60 + "\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.6f}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    print(f"\nMetrics saved to {metrics_output}")
    print("\nDone!")


if __name__ == "__main__":
    main()
