import torch
from transformers import (
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback
)
from datasets import DatasetDict
import os

# Custom callback to print sample text
class SampleTextCallback(TrainerCallback):
    def __init__(self, tokenizer, model, prompt, interval_steps=5000, max_length=50):
        self.tokenizer = tokenizer
        self.model = model
        self.prompt = prompt
        self.interval_steps = interval_steps
        self.max_length = max_length

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.interval_steps == 0 and state.global_step > 0:
            input_ids = self.tokenizer.encode(self.prompt, return_tensors='pt').to(self.model.device)
            output = self.model.generate(input_ids, max_length=self.max_length, do_sample=True)
            decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"\n=== Sample Output at step {state.global_step} ===\n{decoded}\n")
        return control

def train_poetry_model(tokenized_dataset, tokenizer, model_name="distilgpt2", output_dir="./poetry_model"):
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Ensure tokenizer has pad_token (GPT models don’t by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))  # Resize embedding layer

    #print("Training on device:", next(model.parameters()).device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Training on device:", device)

    # Split the actual train set
    train_valid = tokenized_dataset["train"].train_test_split(test_size=0.1)

    # Repackage as a DatasetDict with both train and eval
    tokenized_dataset = DatasetDict({
        "train": train_valid["train"],
        "eval": train_valid["test"]
    })

    training_args = TrainingArguments(
        output_dir="./gpt2-finetuned",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_strategy="steps",
        eval_steps=5000,
        save_strategy="steps",
        save_steps=5000,
        save_total_limit=2,  # keep only last 2 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # use this if using early stopping
        greater_is_better=False,
        logging_steps=500,
        num_train_epochs=1,  # or set max_steps
        report_to="tensorboard",  # or "none"
        # output_dir=output_dir,
        # overwrite_output_dir=True,
        # num_train_epochs=5,
        # per_device_train_batch_size=2,
        # learning_rate=5e-5,
        # save_steps=500,
        # save_total_limit=2,
        # logging_steps=100,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal Language Modeling for GPT
    )

    sample_callback = SampleTextCallback(
        tokenizer=tokenizer,
        model=model,
        prompt="In the twilight of the city,",
        interval_steps=5000,
        max_length=60
    )

    early_stop = EarlyStoppingCallback(early_stopping_patience=3)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["eval"],  # ✅ add this line
        data_collator=data_collator,
        callbacks=[sample_callback, early_stop]  # 👈 this line adds your callbacks
    )

    checkpoint_dir = "./gpt2-finetuned/checkpoint-5000"  # or just "./gpt2-finetuned"
    if os.path.isdir(checkpoint_dir):
        trainer.train(resume_from_checkpoint=checkpoint_dir)
    else:
        trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


# from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
#
# def train_poetry_model(tokenized_dataset, tokenizer, model_name="distilgpt2", output_dir="./poetry_model"):
#     model = AutoModelForCausalLM.from_pretrained(model_name)
#     print(next(model.parameters()).device)
#     training_args = TrainingArguments(
#         output_dir=output_dir,
#         overwrite_output_dir=True,
#         num_train_epochs=3,
#         per_device_train_batch_size=2,
#         save_steps=500,
#         save_total_limit=2,
#         logging_steps=100,
#     )
#
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_dataset["train"],
#     )
#
#     trainer.train()
#     model.save_pretrained(output_dir)
#     tokenizer.save_pretrained(output_dir)