from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16
).to("cuda")
with open('etc/transform_prompt', 'r', encoding='utf-8') as f:
    prompt_content = f.read()

messages = [
    {"role": "system", "content": "You translate robot instructions into a strict action DSL."},
    {"role": "user", "content": prompt_content},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(text, return_tensors="pt").to("cuda")

with torch.inference_mode():
    outputs = model.generate(
        **inputs, max_new_tokens=80,
        do_sample=False
        )

generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
print(tokenizer.decode(generated_ids, skip_special_tokens=True))

