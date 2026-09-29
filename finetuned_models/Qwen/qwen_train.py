import json
import torch
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

# Load fine-tuning data and save it
data_path = "/home/ppllm26_team002/Seminar/data/ft_data.json"
with open(data_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

df = pd.DataFrame(raw_data)

# Use of URIAL prompt template to define the training samples (as in the paper)
urial_prompt = (
    "# Instruction\nBelow is a list of conversations between a human and an AI assistant (you). "
    "Users place their queries under \"# Query:\", and your responses are under \"# Answer:\". "
    "You are a helpful, respectful, and honest assistant. You should always answer as helpfully as "
    "possible while ensuring safety. Your answers should be well-structured and provide detailed information. "
    "They should also have an engaging tone. -> Your responses must not contain any fake, harmful, unethical, "
    "racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful. Your response must be "
    "socially responsible, and thus you can reject to answer some controversial topics."
)

formatted_samples = []
for _, row in df.iterrows():
    query = (
        f"context: {row['context']}\n"
        f"root: {row['root']}\n"
        f"response_sentence_1: {row['candidate_sentence_1']}\n"
        f"response_sentence_2: {row['candidate_sentence_2']}\n"
    )
    instruction = (
        "\nYour task is to examine the following short\n"
        "conversation and assess:- What is the pragmatic intention behind\n"
        "‘response_1’?- Whyor when might someone prefer\n"
        "‘response_1’ over ‘response_2’ pragmatically?\n"
        "Please answer in 1 paragraph.\n"
    )
    target = row.get("candidate_sentence_1_intention", "")
    if not target:
        continue

    
    full_text = f"{urial_prompt}\n\n# Query:\n{query}{instruction}\n# Answer:\n{target}"
    formatted_samples.append({"text": full_text})

dataset = Dataset.from_list(formatted_samples)

# Initialization of the base model
model_id = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# Data collator for Completion-only loss (ensures the model only learns to generate the answer, not to predict the URIAL prompt)
response_template = "\n# Answer:\n"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template, 
    tokenizer=tokenizer
)

# Configure LoRA adaptation
peft_config = LoraConfig(
    r=16,
    lora_alpha=8,                       
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,                  
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Configure Training Loop
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    data_collator=collator,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=2048,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,               
        lr_scheduler_type="cosine",     
        max_steps=60,
        learning_rate=3e-5,             
        fp16=True,
        logging_steps=1,
        output_dir="/home/ppllm26_team002/Seminar/cluster_qwen_lora_outputs",
        report_to="none"
    ),
)

print("Training is starting")
trainer.train()

# 6. Save the LoRA Adapter
model.save_pretrained("qwen_cluster_lora")
tokenizer.save_pretrained("qwen_cluster_lora")