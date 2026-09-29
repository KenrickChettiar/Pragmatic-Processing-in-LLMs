import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# Document paths and load files
eval_data_path = "/home/ppllm26_team002/Seminar/data/eval_data.json"
responses_path = "/home/ppllm26_team002/Seminar/gemma_ft_responses.json"
output_path = "/home/ppllm26_team002/Seminar/gemma_scores_by_llama.json"

with open(eval_data_path, "r", encoding="utf-8") as f:
    eval_data = pd.DataFrame(json.load(f))

with open(responses_path, "r", encoding="utf-8") as f:
    model_responses = json.load(f)

eval_subset = eval_data.iloc[:200]
all_scoring_prompts = []

# Construct the 10-Point scoring prompts
for i in range(len(eval_subset)):
    golden_intention = eval_subset.iloc[i]["candidate_sentence_1_intention"]
    
    response_text = (
        model_responses[i].get("model_output", "")
        if isinstance(model_responses[i], dict)
        else model_responses[i]
    )
    
    prompt = f"""You are an expert evaluator of language model responses. Your task is to score how well a model’s response captures the idea of a response to a given sentence.
The tested model will see a conversation structure like this:
"context": ...
"initial turn": ...
"response_1": ...
"response_2": ...
And they are asked to evaluate: What is the intention behind ‘response_1’? Why or when might someone prefer ‘response_1’ over ‘response_2’?
We have the reference golden intention for you: {golden_intention}, this is the core idea of the intention of response_1.
Model’s response: {response_text}
You should understand that the tested models are base models, thus not deliberately instructed to format their responses and may output random strings and sentences. You just need to check the content of their responses instead of the format.
Score the model’s response on a scale of 1 to 10, or Invalid, where:
- 10: The model’s response perfectly captures the intention described in the reference
- 1: The model’s response poorly captures the intention described in the reference
- Invalid: The model’s response is nonsense or invalid
Return ONLY a JSON object with the following format:
{{
    "score": <number between 1 and 10 or Invalid>,
    "reason": "<brief explanation of your score, no more than 25 words>"
}}
Do not include any other text, just the JSON object."""
    
    all_scoring_prompts.append(prompt)


# Load judge base model (Llama)
model_id = "unsloth/Llama-3.2-3B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# Generate scores
llama_judgments = []
print("Starting Llama judging loop")

for i, scoring_prompt in enumerate(all_scoring_prompts):
    messages = [{"role": "user", "content": scoring_prompt}]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.5,
            do_sample=True,
            top_p=1.0,
            top_k=50,
            pad_token_id=tokenizer.pad_token_id
        )
        
    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    resp = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    llama_judgments.append({
        "sample_id": i + 1,
        "judgment": resp
    })


# Save final document with all scores
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(llama_judgments, f, ensure_ascii=False, indent=4)

print("Llama judging loop complete")