Run all {gemma,llama,qwen}-pipeline files.
Results would be saved in {Gemma,Llama,Qwen} pipeline output.

## Fine-tuning
The result generation is carried out in three steps, starting with the train phase (LoRA fine-tuning), response generation phase, and finally, the LLM-as-a-Judge evaluation phase.

### 1. Training Scripts 

* **Files:**
    - `llama_train.py`
    - `qwen_train.py`
    - `gemma_train.py`

These scripts execute the LoRA fine-tuning for each respective model.

### 2. Evaluation Scripts 

* **Files:**
    - `llama_eval.py`
    - `qwen_eval.py`
    - `gemma_eval.py`

These scripts handle the generation of pragmatic inferences from the fine-tuned models.

### 3. Scoring Scripts 

* **Files:**
    - Scoring the Llama model:
        - `llama_gemma_as_judge.py`
        - `llama_qwen_as_judge.py`
    - Scoring the Gemma model:
        - `gemma_llama_as_judge.py`
        - `gemma_qwen_as_judge.py`
    - Scoring the Qwen model:
        - `qwen_llama_as_judge.py`
        - `qwen_gemma_as_judge.py`

Each model has two corresponding scoring scripts to execute the LLM-as-a-Judge evaluation.





Save the finetuned models outputs in a folder named finetuned_models.
Run 10pointscoring.py and generate the plots.
