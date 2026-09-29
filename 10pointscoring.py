import matplotlib.pyplot as plt
import json

#Read all the outputs
gemma_avg, llama_avg, qwen_avg = 0, 0, 0

with open("Gemma pipeline output/GemmatoJudgeLlama.json", "r") as f:
    gemmatojudgellamavalues = json.load(f)
with open("Gemma pipeline output/GemmatoJudgeQwen.json", "r") as f:
    gemmatojudgeQwen = json.load(f)

gemmatojudgeQwenvalues = [list(i.values())[0] for i in gemmatojudgeQwen]
gemma_avg = (sum(gemmatojudgellamavalues)/len(gemmatojudgellamavalues)+ sum(gemmatojudgeQwenvalues)/len(gemmatojudgeQwenvalues))/2


with open("Llama pipeline output/LlamatoJudgeGemma.json", "r") as f:
    llamatojudgegemma = json.load(f)
with open("Llama pipeline output/LlamatoJudgeQwen.json", "r") as f:
    llamatojudgeqwen = json.load(f)

llamatojudgegemmavalues = [list(i.values())[0] for i in llamatojudgegemma]
llamatojudgeqwenvalues = [list(i.values())[0] for i in llamatojudgeqwen]

llama_avg = (sum(llamatojudgegemmavalues)/len(llamatojudgegemmavalues)+ sum(llamatojudgeqwenvalues)/len(llamatojudgeqwenvalues))/2


with open("Qwen pipeline output/QwentoJudgeGemma.json", "r") as f:
    qwentojudgegemma = json.load(f)
with open("Qwen pipeline output/QwentoJudgeLlama.json", "r") as f:
    qwentojudgellamavalues = json.load(f)

qwentojudgegemmavalues = [list(i.values())[0] for i in qwentojudgegemma]
qwen_avg = (sum(qwentojudgellamavalues)/len(qwentojudgellamavalues)+ sum(qwentojudgegemmavalues)/len(qwentojudgegemmavalues))/2

#_--------------------------------------------------

#read all finetuned model outputs
with open("finetuned_models/llama_scores_by_gemma(V2).json", "r") as f:
    ft_llamatojudgegemma = json.load(f)
with open("finetuned_models/llama_scores_by_qwen(V2).json", "r") as f:
    ft_llamatojudgeqwen = json.load(f)

ft_llamatojudgegemmavalues = [i['score'] for i in ft_llamatojudgegemma]
ft_llamatojudgeqwenvalues = [i['score'] for i in ft_llamatojudgeqwen]


with open("finetuned_models/qwen_scores_by_gemma(V2).json", "r") as f:
    ft_qwentojudgegemma = json.load(f)
with open("finetuned_models/qwen_scores_by_llama(V2).json", "r") as f:
    ft_qwentojudgellama = json.load(f)

ft_qwentojudgegemmavalues = [i['score'] for i in ft_qwentojudgegemma]
ft_qwentojudgellamavalues = [i['score'] for i in ft_qwentojudgellama]


with open("finetuned_models/gemma_scores_by_llama(V2).json", "r") as f:
    ft_gemmatojudgellama = json.load(f)
with open("finetuned_models/gemma_scores_by_qwen(V2).json", "r") as f:
    ft_gemmatojudgeqwen = json.load(f)

ft_gemmatojudgellamavalues = [i['score'] for i in ft_gemmatojudgellama]
ft_gemmatojudgeqwenvalues = [i['score'] for i in ft_gemmatojudgeqwen]




#-------------------------

#Normal model averges
normal_llama_avg = (
    sum(llamatojudgegemmavalues) / len(llamatojudgegemmavalues)
  + sum(llamatojudgeqwenvalues) / len(llamatojudgeqwenvalues)
) / 2

normal_gemma_avg = (
    sum(gemmatojudgellamavalues) / len(gemmatojudgellamavalues)
  + sum(gemmatojudgeQwenvalues) / len(gemmatojudgeQwenvalues)
) / 2

normal_qwen_avg = (
    sum(qwentojudgellamavalues) / len(qwentojudgellamavalues)
  + sum(qwentojudgegemmavalues) / len(qwentojudgegemmavalues)
) / 2


#Finetuned model averages
ft_llama_avg = (
    sum(ft_llamatojudgegemmavalues)/ len(ft_llamatojudgegemmavalues)
  + sum(ft_llamatojudgeqwenvalues)/ len(ft_llamatojudgeqwenvalues))/ 2

ft_gemma_avg = (
    sum(ft_gemmatojudgellamavalues)/ len(ft_gemmatojudgellamavalues)
  + sum(ft_gemmatojudgeqwenvalues)/ len(ft_gemmatojudgeqwenvalues))/ 2

ft_qwen_avg = (
    sum(ft_qwentojudgegemmavalues)/ len(ft_qwentojudgegemmavalues)
  + sum(ft_qwentojudgellamavalues)/len(ft_qwentojudgellamavalues))/ 2


# Plotting the results
import numpy as np
models = ["Llama", "Gemma", "Qwen"]

normal_scores = [normal_llama_avg,normal_gemma_avg,normal_qwen_avg]

finetuned_scores = [ft_llama_avg,ft_gemma_avg,ft_qwen_avg]
x = np.arange(len(models))
width = 0.35
plt.figure(figsize=(9, 5))
b1 = plt.bar(x-width/2,normal_scores,width,label="Normal")
b2 = plt.bar(x+ width/2,finetuned_scores,width,label="Fine tuned")

plt.xlabel("Model Names")
plt.ylabel("Average Judge Score")
plt.title("Base vs Fine tuned Model Performance")
plt.xticks(x, models)
plt.ylim(0,10)
plt.legend()
for bars in [b1,b2]:
    for bar in bars:
        value = bar.get_height()
        plt.text(bar.get_x()+ bar.get_width()/2, value+ 0.1, f"{value:.2f}",ha="center",va="bottom")
plt.tight_layout()
plt.savefig("Scoring")
plt.show()

#calculating the metric for judge agreement
def agreement_within_1(judge1, judge2):
    judge1 = np.array(judge1)
    judge2 = np.array(judge2)
    agreement = np.abs(judge1-judge2) <= 1
    return np.mean(agreement) * 100

normal_agreement = [
    agreement_within_1(llamatojudgegemmavalues,llamatojudgeqwenvalues),
    agreement_within_1(gemmatojudgellamavalues,gemmatojudgeQwenvalues),
    agreement_within_1(qwentojudgellamavalues,qwentojudgegemmavalues)]

ft_agreement = [
    agreement_within_1(ft_llamatojudgegemmavalues,ft_llamatojudgeqwenvalues),
    agreement_within_1(ft_gemmatojudgellamavalues,ft_gemmatojudgeqwenvalues),
    agreement_within_1(ft_qwentojudgellamavalues,ft_qwentojudgegemmavalues)]
models = ["Llama", "Gemma", "Qwen"]
print("Agreement within ±1 point")

#plotting the agreements
x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(9, 5))

b1 = plt.bar(x-width/2,normal_agreement,width,label="Normal")

b2 = plt.bar(x+ width/2,ft_agreement,width,label="Fine-tuned")

plt.xlabel("Model Names")
plt.ylabel("Agreement within ±1 point (%)")
plt.title("Inter-Judge Agreement within ±1 Point")
plt.xticks(x, models)
plt.ylim(0, 100)
plt.legend()

for bars in [b1, b2]:
    for bar in bars:
        value = bar.get_height()
        plt.text(bar.get_x()+ bar.get_width() / 2,value+ 1,f"{value:.1f}%",ha="center",va="bottom")

plt.tight_layout()
plt.savefig("Inter-Judge agreement")
plt.show()
