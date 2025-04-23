from datasets import Dataset, load_dataset
import numpy as np
import json

# Load the saved JSON file
with open("unsloth_data.json", "r") as f:
    unsloth_data = json.load(f)

# Convert to Hugging Face Dataset
dataset = Dataset.from_list(unsloth_data)

# Define the formatting function
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    #print(instructions)
    #print(inputs)
    #print(outputs)
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

# Apply formatting function
dataset = dataset.map(formatting_prompts_func, batched=True)

# Example: Check first entry
print(dataset[0]["text"])
