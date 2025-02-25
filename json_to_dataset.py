from datasets import Dataset, load_dataset

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

EOS_TOKEN = "eos" #tokenizer.eos_token  # Ensure the model knows when to stop generating

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = [
        alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN
        for instruction, input_text, output in zip(instructions, inputs, outputs)
    ]
    return {"text": texts}

# Apply formatting function
dataset = dataset.map(formatting_prompts_func, batched=True)

# Example: Check first entry
print(dataset[0]["text"])
