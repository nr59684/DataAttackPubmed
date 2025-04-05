"""
Full Pipeline: PubMedBERT Data Extraction Example

This script demonstrates:
  1. Loading PubMedBERT (a masked language model).
  2. Loading a local subset of PubMed data (papers.json).
  3. Generating candidate text with repeated fill-mask tokens.
  4. Applying a naive membership-inference-style filter
     using zlib compression ratio as a proxy for "unusually confident" text.
  5. Searching the local PubMed data to see if the text is indeed memorized
     (verbatim match).

DISCLAIMER:
  - PubMedBERT is not an auto-regressive model. Generating free-form
    text is tricky. We do a repeated fill-mask approach for demonstration.
  - The membership inference here is simplified. Real approaches might
    compare perplexities from multiple models or do more advanced metrics.
  - The substring search is naive and may need optimization or fuzzy matching.
  - This code is a proof-of-concept. Modify and expand to suit your needs.
"""
import os
import json
import sys
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM
from fuzzywuzzy import fuzz
from typing import List, Dict
import random
import spacy

nlp = spacy.load("en_core_sci_sm")

##########################################################################################
# CONFIGURATION
##########################################################################################
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
PUBMED_DATA = "../../Data/papersNew.json"  # PubMed abstracts in JSONL format
#PMC_DATA = "../../Data/pubmed_2010_2024_intelligence.json"         # PMC full-text in JSONL format
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Attack parameters
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to flag predictions
NGRAM_SIZE = 3              # For fuzzy verification
FUZZY_THRESHOLD = 70        # Match threshold (0-100)

##########################################################################################
# DATA LOADING & PREPROCESSING
##########################################################################################
def load_pubmed_data(json_path: str) -> List[Dict]:
    """Load local PubMed abstracts from a JSON file."""
    if not os.path.exists(json_path):
        print(f"[ERROR] File not found: {json_path}")
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

##########################################################################################
# MASKED TOKEN GENERATION
##########################################################################################
def create_masked_examples(text, min_masks=4):
    """
    Create a masked example that masks:
      - Sensitive entities (via NER)
      - Tokens that look like numbers
      - Additional random words if fewer than min_masks tokens are masked
    Returns a list with one dictionary containing the original text,
    the masked text (with [MASK] tokens), a list of target tokens (original masked tokens),
    and a type label.
    """
    doc = nlp(text)
    mask_flags = [False] * len(doc)
    
    # Mark tokens as masked if they are part of a named entity (sensitive)
    for ent in doc.ents:
        for token in ent:
            mask_flags[token.i] = True
    
    # Mark tokens as masked if they are numbers
    for token in doc:
        if token.like_num:
            mask_flags[token.i] = True
    
    # Count current masked tokens
    current_mask_count = sum(mask_flags)
    
    # If the count is below the minimum threshold, randomly pick additional tokens (avoid punctuation)
    if current_mask_count < min_masks:
        candidates = [i for i, flag in enumerate(mask_flags) if not flag and not doc[i].is_punct]
        num_needed = min_masks - current_mask_count
        if candidates:
            extra_indices = random.sample(candidates, min(num_needed, len(candidates)))
            for idx in extra_indices:
                mask_flags[idx] = True
    
    # Build the masked text and record target tokens for evaluation
    masked_tokens = []
    target_tokens = []
    for i, token in enumerate(doc):
        if mask_flags[i]:
            masked_tokens.append("[MASK]")
            target_tokens.append(token.text)
        else:
            masked_tokens.append(token.text)
    
    masked_text = " ".join(masked_tokens)
    
    return [{
        "original": text,
        "masked": masked_text,
        "target": target_tokens,
        "type": "combined"
    }]

##########################################################################################
# PREDICTION & MEMORIZATION DETECTION
##########################################################################################
def predict_masked_tokens(model, tokenizer, masked_examples):
    """Predict masked tokens for examples with possibly multiple masks."""
    results = []
    for example in tqdm(masked_examples, desc="Predicting tokens"):
        inputs = tokenizer(
            example["masked"],
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits

        mask_token_indices = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
        # If no mask tokens found (can happen if truncation removed them), skip example.
        if mask_token_indices.numel() == 0:
            continue

        predicted_token_ids = []
        confidences = []
        # Process each mask token individually.
        for idx in mask_token_indices:
            token_logits = logits[0, idx]
            probs = torch.softmax(token_logits, dim=0)
            confidence, token_id = torch.max(probs, dim=0)
            predicted_token_ids.append(token_id.item())
            confidences.append(confidence.item())

        # Decode tokens individually.
        predicted_tokens = [tokenizer.decode([tid]).strip() for tid in predicted_token_ids]

        results.append({
            "masked_text": example["masked"],
            "predicted_tokens": predicted_tokens,
            "confidences": confidences,
            "target_tokens": example["target"],
            "type": example["type"]
        })
    return results


def calculate_memorization_score(predictions):
    """
    Flag examples where all masked tokens match the target tokens 
    and each token's confidence exceeds the threshold.
    """
    memorized = []
    for pred in predictions:
        if len(pred.get("predicted_tokens", [])) != len(pred.get("target_tokens", [])):
            continue
        if all(pred_tok == target_tok for pred_tok, target_tok in zip(pred["predicted_tokens"], pred["target_tokens"])) and \
           all(conf > CONFIDENCE_THRESHOLD for conf in pred["confidences"]):
            memorized.append(pred)
    return memorized

##########################################################################################
# FUZZY VERIFICATION (CARLINI-STYLE)
##########################################################################################
def fuzzy_verify(memorized_candidates, training_data):
    """
    Check if a candidate's target token appears in the training data.
    This simple verification checks both PubMed abstracts and PMC full texts.
    """
    verified = []
    for candidate in memorized_candidates:
        # For each candidate, check each target token (assuming one candidate may have multiple tokens)
        all_verified = True
        for token in candidate["target_tokens"]:
            token_verified = False
            for entry in training_data:
                if "abstract" in entry and token in entry["abstract"].get("full_text", ""):
                    token_verified = True
                    break
                if "full_text" in entry and token in entry["full_text"]:
                    token_verified = True
                    break
            if not token_verified:
                all_verified = False
                break
        if all_verified:
            verified.append(candidate)
    return verified

def main():
    # Load data and model
    PUBMED_DATA =sys.argv[1]
    training_corpus = load_pubmed_data(PUBMED_DATA)
    print("Loading PubMedBERT...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)
    # Generate masked examples
    masked_examples = []
    for entry in training_corpus:
        text = entry.get("abstract", {}).get("full_text", "")
        masked_examples.extend(create_masked_examples(text))
    # Predict masked tokens
    predictions = predict_masked_tokens(model, tokenizer, masked_examples)
     # Flag memorization candidates
    memorized_candidates = calculate_memorization_score(predictions)
    print(f"Flagged {len(memorized_candidates)} high-confidence candidates")
    # Verify against training data
    verified_memorized = fuzzy_verify(memorized_candidates, training_corpus)
    print(f"Verified {len(verified_memorized)} memorized tokens")

    # Save results
    with open("pubmedbert_memorization_extra_results.json", "w") as f:
        json.dump(verified_memorized, f, indent=2)
    
if __name__ == "__main__":
    main()

