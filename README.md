# dataset-dolly-to-the-rescue

01GangaPutraBheeshma/databricks-facebook-opt2-ft-dolly-UT is an open-source language model, a fine-tuned version of facebook/opt-350m, and Supervised Finetuning was used to retrain and finetune the model - a strategy inspired by offline transfer reinforcement learning. This version of Model learn from mixed-quality data without preference labels, delivering exceptional performance. Despite the simple approach, my commitment is to develop a high-performance, commercially viable, open-source large language model, and I continue to make significant strides toward this vision.

## Model Details

### Model Description

The data on which this model was trained is databricks/databricks-dolly-15k. Within this dataset, you'll discover a compilation of entries featuring a category, an instruction, a context, and a response corresponding to that instruction. The project's objective is to enhance the quality of instructions, inputs, and responses, ensuring they align seamlessly with their designated task category. All textual components should be articulate, providing genuine information. Additionally, responses should strive for completeness while maintaining conciseness.

- **Developed by:** Uttasarga Singh
- **Funded by [optional]:** Self
- **Shared by [optional]:** Self
- **Model type:** Decoder based Model
- **Language(s) (NLP):** English
- **License:** Meta
- **Finetuned from model [optional]:** facebook/opt-350m

### Model Sources [optional]

- **Repository:** https://github.com/uttasarga9067/dataset-dolly-to-the-rescue
- **Paper [optional]:** In Development

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### How to get started with this Model
```
import torch
from peft import PeftModel, PeftConfig

model_name = "01GangaPutraBheeshma/facebook_opt2"
trained_model = AutoModelForCausalLM.from_pretrained(model_name)
trained_tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = """ Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
if one gets corona and you are self-isolating and it is not severe, is there any meds that one can take?

### Response: """
input_ids = trained_tokenizer(prompt, return_tensors="pt", truncation=True).input_ids

print(f"After Training Response :")
outputs = trained_model.generate(input_ids=input_ids, max_new_tokens=100, do_sample=True, top_p=0.9,temperature=1.0)
print(f"-------------------------\n\n")
print(f"Generated instruction:\n{trained_tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")
print(f"-------------------------\n\n")
```

### Fine-tuning this Model on your own Dataset(Preprocessing the Input Data)

If you would like to fine-tune this model for other datasets, please try to develop a function, that can make our datasets to be in the same format as our function desires, thus using this below script.
```
INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
INSTRUCTION_KEY = "### Instruction:"
INPUT_KEY = "Input:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"

PROMPT_NO_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{response_key}
{response}

{end_key}""".format(
  intro=INTRO_BLURB,
  instruction_key=INSTRUCTION_KEY,
  instruction="{instruction}",
  response_key=RESPONSE_KEY,
  response="{response}",
  end_key=END_KEY
)

PROMPT_WITH_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{input_key}
{input}

{response_key}
{response}

{end_key}""".format(
  intro=INTRO_BLURB,
  instruction_key=INSTRUCTION_KEY,
  instruction="{instruction}",
  input_key=INPUT_KEY,
  input="{input}",
  response_key=RESPONSE_KEY,
  response="{response}",
  end_key=END_KEY
)

def apply_prompt_template(examples):
  instruction = examples["instruction"]
  response = examples["response"]
  context = examples.get("context")

  if context:
    full_prompt = PROMPT_WITH_INPUT_FORMAT.format(instruction=instruction, response=response, input=context)
  else:
    full_prompt = PROMPT_NO_INPUT_FORMAT.format(instruction=instruction, response=response)
  return { "text": full_prompt }

dataset = dataset.map(apply_prompt_template)
```
## Training Details and Procedure
```
from transformers import TrainingArguments
from trl import SFTTrainer

output_dir = "./facebook_opt2"
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 500
logging_steps = 100
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 1000
warmup_ratio = 0.03
lr_scheduler_type = "constant"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    ddp_find_unused_parameters=False,
    push_to_hub=True
)

max_seq_length = 512

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)
```

| Parameter                     | Description                                                      |
|-------------------------------|------------------------------------------------------------------|
| `output_dir`                  | Directory to save the trained model and logs.                    |
| `per_device_train_batch_size` | Number of training samples per GPU.                               |
| `gradient_accumulation_steps` | Number of steps to accumulate gradients before updating the model.|
| `optim`                       | Optimizer for training (e.g., "paged_adamw_32bit").               |
| `save_steps`                  | Save model checkpoints every N steps.                            |
| `logging_steps`               | Log training information every N steps.                          |
| `learning_rate`               | Initial learning rate for training.                               |
| `max_grad_norm`               | Maximum gradient norm for gradient clipping.                      |
| `max_steps`                   | Maximum number of training steps.                                 |
| `warmup_ratio`                | Ratio of warmup steps during learning rate warmup.               |
| `lr_scheduler_type`          | Type of learning rate scheduler (e.g., "constant").              |
| `fp16`                        | Enable mixed-precision training.                                  |
| `group_by_length`             | Group training samples by length for efficiency.                 |
| `ddp_find_unused_parameters` | Enable distributed training parameter setting.                   |
| `push_to_hub`                 | Push the trained model to the Hugging Face Model Hub.            |


### Training Data

[More Information Needed]

#### Metrics

| Step  | Training Loss |
|-------|---------------|
| 100   | 2.189900      |
| 200   | 2.014100      |
| 300   | 1.957200      |
| 400   | 1.990000      |
| 500   | 1.985200      |
| 600   | 1.986500      |
| 700   | 1.964300      |
| 800   | 1.951900      |
| 900   | 1.936900      |
| 1000  | 2.011200      |


### Results

[More Information Needed]

#### Summary



## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

[More Information Needed]

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** [More Information Needed]
- **Hours used:** [More Information Needed]
- **Cloud Provider:** [More Information Needed]
- **Compute Region:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]

## Technical Specifications [optional]

### Model Architecture and Objective

[More Information Needed]

### Compute Infrastructure

[More Information Needed]

#### Hardware

[More Information Needed]

#### Software

[More Information Needed]

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

[More Information Needed]

**APA:**

[More Information Needed]

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

[More Information Needed]

## More Information [optional]

[More Information Needed]

## Model Card Authors [optional]

[More Information Needed]

## Model Card Contact

[More Information Needed]
