from transformers import Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling, GPTJForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "EleutherAI/gpt-j-6B"
model = GPTJForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load your dataset
def load_dataset(file_path):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128
    )

train_dataset = load_dataset("../data/training_data.txt")
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Fine-tuning
trainer.train()
