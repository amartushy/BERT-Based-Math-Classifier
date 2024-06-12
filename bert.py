import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict, Dataset
import logging

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path):
    import pandas as pd
    logger.info(f"Loading data from {file_path}")
    data = pd.read_json(file_path)
    return Dataset.from_pandas(data)

def tokenize_data(example):
    inputs = tokenizer(example['problem'], padding='max_length', truncation=True, max_length=512)
    inputs['labels'] = example['labels']
    return inputs

def save_model(model, tokenizer, model_path):
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    logger.info(f"Model and tokenizer saved to {model_path}")

train_data = load_data('combined_train.json')
test_data = load_data('combined_test.json')

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=7)

def transform_labels(example):
    label_dict = {'Algebra': 0, 'Counting & Probability': 1, 'Geometry': 2, 'Intermediate Algebra': 3, 'Number Theory': 4, 'Prealgebra': 5, 'Precalculus': 6}
    example['labels'] = label_dict[example['type']]
    return example

train_data = train_data.map(transform_labels)
test_data = test_data.map(transform_labels)

tokenized_train = train_data.map(tokenize_data, batched=True)
tokenized_test = test_data.map(tokenize_data, batched=True)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",
    load_best_model_at_end=True,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test
)

logger.info("Starting training")
trainer.train()
logger.info("Evaluating model")
trainer.evaluate()

# Save the model and tokenizer
save_model(model, tokenizer, './model')

logger.info("Training and evaluation complete")
