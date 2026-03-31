import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate  # pip install evaluate rouge_score absl-py

# 1. Initialize Metrics and Models
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "gpt2" # Using a small model for demonstration
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

# Load evaluation suites
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

def calculate_perplexity(text):
    """
    Calculates how 'surprised' the model is. 
    Lower = Better understanding of patterns.
    """
    encodings = tokenizer(text, return_tensors="pt").to(device)
    input_ids = encodings.input_ids
    
    with torch.no_grad():
        # The model calculates Cross-Entropy Loss automatically when labels are provided
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        
    # Perplexity is the exponent of the loss: e^loss
    return math.exp(loss.item())

def calculate_linguistic_scores(prediction, reference):
    """
    Compares LLM output to a 'Ground Truth' human reference.
    """
    # ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
    # Good for summarization (measures how much of the reference was captured)
    rouge_results = rouge.compute(predictions=[prediction], references=[reference])
    
    # BLEU (Bilingual Evaluation Understudy)
    # Good for translation/precision (measures how accurate the snippets are)
    bleu_results = bleu.compute(predictions=[prediction], references=[reference])
    
    return rouge_results, bleu_results

# --- EXECUTION ---

# Example Data
reference_text = "The quick brown fox jumps over the lazy dog."
llm_output = "A fast brown fox leaps across a tired dog."

# 1. Measure Perplexity (Intrinsic Metric)
ppl = calculate_perplexity(llm_output)
print(f"--- Intrinsic Metric ---")
print(f"Perplexity: {ppl:.2f}\n")

# 2. Measure Similarity (Benchmark Metrics)
rouge_score, bleu_score = calculate_linguistic_scores(llm_output, reference_text)
print(f"--- Similarity Metrics ---")
print(f"ROUGE-L: {rouge_score['rougeL']:.4f}")
print(f"BLEU: {bleu_score['bleu']:.4f}")

# 3. LLM-as-a-Judge (Conceptual Scripting)
def llm_judge(output):
    """
    In 2026, we use a stronger model to 'reason' about the quality.
    """
    print("\n--- LLM-as-a-Judge (Mock) ---")
    # In a real scenario, you would prompt a model like GPT-4o or Gemini 1.5 Pro:
    # "Rate the following text on a scale of 1-10 for 'Helpfulness'..."
    print("Judge Rating: 8/10 (Reason: High synonym usage, accurate meaning).")

llm_judge(llm_output)