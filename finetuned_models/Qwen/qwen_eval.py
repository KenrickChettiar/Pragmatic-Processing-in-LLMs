import json
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Document paths
base_model_id = "Qwen/Qwen2.5-3B-Instruct"
adapter_path = "/home/ppllm26_team002/Seminar/qwen_cluster_lora"
eval_data_path = "/home/ppllm26_team002/Seminar/data/eval_data.json"
output_path = "/home/ppllm26_team002/Seminar/qwen_ft_responses.json"

# Use of URIAL prompt template to define the evaluation samples (as in the paper)
# and run inference with the finetuned model (producing pragmatic inferences based on the samples)
urial_prompt = (
    "# Instruction\nBelow is a list of conversations between a human and an AI assistant (you). "
    "Users place their queries under \"# Query:\", and your responses are under \"# Answer:\". "
    "You are a helpful, respectful, and honest assistant. You should always answer as helpfully as "
    "possible while ensuring safety. Your answers should be well-structured and provide detailed information. "
    "They should also have an engaging tone. -> Your responses must not contain any fake, harmful, unethical, "
    "racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful. Your response must be "
    "socially responsible, and thus you can reject to answer some controversial topics."
)

# Loading and saving evaluation data (only first 200 samples)
with open(eval_data_path, "r", encoding="utf-8") as f:
    eval_data = pd.DataFrame(json.load(f))[:200]

all_prompts = []
for i in range(len(eval_data)):
    query = (
        f"context: {eval_data.iloc[i]['context']}\n"
        f"root: {eval_data.iloc[i]['root']}\n"
        f"response_sentence_1: {eval_data.iloc[i]['candidate_sentence_1']}\n"
        f"response_sentence_2: {eval_data.iloc[i]['candidate_sentence_2']}\n"
    )
    instruction = (
        "\nYour task is to examine the following short\n"
        "conversation and assess:- What is the pragmatic intention behind\n"
        "‘response_1’?- Whyor when might someone prefer\n"
        "‘response_1’ over ‘response_2’ pragmatically?\n"
        "Please answer in 1 paragraph.\n"
    )
    
    
    full_prompt = f"{urial_prompt}\n\n# Query:\n{query}{instruction}\n# Answer:\n"
    all_prompts.append(full_prompt)


# Loading base model 
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Attaching the fine-tuned LoRA addapter to the base model
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# Response generation
model_responses = []
print("Inference is starting")

for i, prompt_text in enumerate(all_prompts):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    # Decoding parameters for response generation match the configurations specified in Appendix C of the original Yu et al. paper
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.5,
            top_k=50,
            top_p=1.0,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode only the newly generated tokens
    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    model_responses.append(response)


# Save final document with the list of responses
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(model_responses, f, ensure_ascii=False, indent=4)

print("Inference complete")