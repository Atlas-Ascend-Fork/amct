import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from model import *
from datasets import load_dataset

model_path = "/data/groups/QY_LLM_Other/wanghaoyu/exp/tvq/MiniCPM4-1B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
config._attn_implementation = 'sdpa'
model = AutoModelForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=True).to("cuda")

torch.set_grad_enabled(False)

dataset_path = "/data/groups/QY_LLM_Other/wanghaoyu/resources/datasets/Fineweb/"

dataset = load_dataset(dataset_path)["train"]

for i, data in enumerate(dataset):
    inputs = tokenizer([data['content']], return_tensors='pt').to("cuda")
    res = model(**inputs, labels=inputs['input_ids'])
    print(res[0])
    if i > 5:
        break

# while True:
#     prompt = input("input prompt (quit to exit): ")
#     if prompt.strip() == "quit":
#         break
#     prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
#     inputs = tokenizer(prompt, return_tensors='pt')
#     outputs = model.generate(
#         input_ids=inputs['input_ids'].cuda(),
#         attention_mask=inputs['attention_mask'].cuda(),
#         max_new_tokens=64,
#         return_dict_in_generate=True
#     )
#     token = outputs.sequences[0, :]
#     print(token.tolist())
#     output_str = tokenizer.decode(token, skip_special_tokens=False)
#     print(output_str)

    