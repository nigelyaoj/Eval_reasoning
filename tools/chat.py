import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams


model_name_or_path = "hkust-nlp/Qwen-2.5-Math-7B-SimpleRL-Zoo"
temperature = 0
top_p = 1

llm = LLM(
    model=model_name_or_path,
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    trust_remote_code=True,
    gpu_memory_utilization=0.8,
    max_model_len=2048,
)

def print_message(message):
    for mess in message:
        if mess["role"] == "user":
            print("="*20, "User", "="*20)
            print(mess["content"])
        elif mess["role"] == "assistant":
            print("="*20, "Assistant", "="*20)
            print(mess["content"])
        elif mess["role"] == "system":
            print("="*20, "System", "="*20)
            print(mess["content"])
        else:
            raise ValueError("role must be one of [system, user, assistant]")

def add_message(message, text, role="user"):

    if role == "system":
        mess = {"role": "system", "content": text.strip()}
    elif role == "user":
        mess = {"role": "user", "content": text.strip()}
    elif role == "assistant":
        mess = {"role": "assistant", "content": text.strip()}
    else:
        raise ValueError("role must be one of [system, user, assistant]")
    
    message.append(mess)


def chat(message):

    tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, trust_remote_code=True
            )

    prompt = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )

    outputs = llm.generate(
        prompt,
        SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=2048,
            n=1,
            stop_token_ids=(
                [151645, 151643]
                if "qwen2" in model_name_or_path.lower()
                else None
            ),
        ),
    )
    outputs = [output.outputs[0].text for output in outputs]
    outputs = outputs[0]
    
    return prompt, outputs


def extract_json(json_str):
    try:
        if json_str.find("```") >-1:
            json_str = json_str[json_str.find("```")+3:].split("```")[0]
        if json_str.find("json") >-1:
            json_str = json_str[json_str.find("json")+5:]
        json_str = eval(json_str)
    except Exception as e:
        print(f"Error: {e}")
        return None

    return json_str

system_prompt = "Please reason step by step, and put your final answer within \\boxed{{}}."
problem = "Compute: $1-2+3-4+5- \dots +99-100$."

message = []
add_message(message, system_prompt, role="system")
add_message(message, problem, role="user")

response = chat(message)
print(response)
import pdb;pdb.set_trace()
add_message(message, response, role="assistant")
print_message(message)

import pdb;pdb.set_trace()