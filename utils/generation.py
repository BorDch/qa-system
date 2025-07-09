# === generation.py ===
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence

BEST_PARAMS = {
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
    "max_new_tokens": 100
}

def load_generation_model(model_name="sberbank-ai/rugpt3large_based_on_gpt2"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    return tokenizer, model, device

def generate_answer(prompts, tokenizer, model, device, params=BEST_PARAMS, num_return_sequences=1):
    encoded = [tokenizer(p, return_tensors="pt", truncation=True, max_length=1024)["input_ids"][0] for p in prompts]
    outputs = []
    
    for i in range(0, len(encoded), 8):  # Batch size = 8
        batch = encoded[i:i+8]
        batch_inputs = pad_sequence(batch, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
        attention_mask = (batch_inputs != tokenizer.pad_token_id).long()

        with torch.no_grad():
            generated = model.generate(
                input_ids=batch_inputs,
                attention_mask=attention_mask,
                max_new_tokens=params["max_new_tokens"],
                do_sample=True,
                temperature=params["temperature"],
                top_k=params["top_k"],
                top_p=params["top_p"],
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=num_return_sequences,
            )
            outputs.extend(generated)

    # Группировка по prompt-ам
    answers = []
    for i in range(len(prompts)):
        group = outputs[i * num_return_sequences : (i + 1) * num_return_sequences]
        decoded = [tokenizer.decode(ans, skip_special_tokens=True).replace(prompts[i], "").strip() for ans in group]
        answers.append(decoded)

    return answers
