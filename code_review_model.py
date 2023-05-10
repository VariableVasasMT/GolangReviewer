from transformers import (
    BertForSequenceClassification, BertTokenizer,
    GPT2ForSequenceClassification, GPT2Tokenizer,
    T5ForConditionalGeneration, T5Tokenizer
)
from transformers import Trainer, TrainingArguments, DataCollatorForTokenClassification, DataCollatorWithPadding
import datasets
from merge_request_fetcher import GitHubGitLabFetcher
import pickle5 as pickle

import os

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

dataset_lock = threading.Lock()

# from lock import use_custom_tqdm

# datasets.disable_progress_bar()

load_dataset_base = datasets.load_dataset 
DatasetDict = datasets.DatasetDict

class CodeReviewModel:
    def __init__(self, model_name, model_class, tokenizer_class):
        self.model_name = model_name
        self.model = model_class.from_pretrained(model_name)
        self.tokenizer = tokenizer_class.from_pretrained(model_name)
        self.trainer = None

        # Set padding token if it's not defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize(self, batch):
        return self.tokenizer(batch["text"], padding="max_length", truncation=True, return_tensors="pt")
    
    def tokenize_dataset(self, dataset, output_dir):
        tokenized_dataset = {}
        for split in dataset.keys():
            tokenized_dataset[split] = dataset[split].map(self.tokenize, batched=True, num_proc=5)

        print("tokenizing dataset to output_dir :=====================> ", output_dir)

        os.makedirs(output_dir, exist_ok=True)
        for split in tokenized_dataset.keys():
            with open(os.path.join(output_dir, f"{split}.pkl"), "wb") as f:
                pickle.dump(tokenized_dataset[split], f)

        
        train_size = int(0.8 * len(tokenized_dataset['train']))
        train_dataset = tokenized_dataset['train'].select(range(train_size))
        validation_dataset = tokenized_dataset['train'].select(range(train_size, len(tokenized_dataset['train'])))
        prepared_dataset = DatasetDict({"train": train_dataset, "validation": validation_dataset})
        return prepared_dataset

    def load_tokenized_dataset(self, input_dir):
        tokenized_dataset = {}
        print("loading tokenized dataset for input_dir :=====================> ", input_dir)
        
        for split in ["train", "validation"]:
            with open(os.path.join(input_dir, f"prepared_{self.model_name}/{split}.pkl"), "rb") as f:
                tokenized_dataset[split] = pickle.load(f)

        return tokenized_dataset
    
    def prepare_dataset(self, dataset_path, force_prepare):
        output_dir = f"prepared_{self.model_name}"
        if not os.path.exists(output_dir) or force_prepare:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            dataset = load_dataset_base("text", data_files=dataset_path)
            prepared_dataset = self.tokenize_dataset(dataset, output_dir)

            for split in ["train", "validation"]:
                with open(f"{output_dir}/{split}.pkl", "wb") as f:
                    pickle.dump(prepared_dataset[split], f)
        else:
            for split in ["train", "validation"]:
                with open(f"{output_dir}/{split}.pkl", "wb") as f:
                    prepared_dataset = pickle.load(f)

        return prepared_dataset

    def fetch_reviews(self, github_token, gitlab_token, gitlab_organization_id):
        fetcher = GitHubGitLabFetcher(github_token, gitlab_token, gitlab_organization_id)
        return fetcher.fetch_reviews()
    
    def get_collator(self):
        return DataCollatorWithPadding(tokenizer=self.tokenizer)
    
    def train(self, train_dataset, eval_dataset, output_dir="output", num_train_epochs=3, train_batch_size=8, eval_batch_size=8, learning_rate=2e-5):
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            learning_rate=learning_rate,
            evaluation_strategy="epoch",
            logging_dir="logs",
        )

        data_collator = self.get_collator()

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        self.trainer.train()

    def evaluate(self):
        return self.trainer.evaluate()


class BERTCodeReviewModel(CodeReviewModel):
    def __init__(self):
        super().__init__("bert-base-uncased", BertForSequenceClassification, BertTokenizer)


class GPT2CodeReviewModel(CodeReviewModel):
    def __init__(self):
        super().__init__("gpt2", GPT2ForSequenceClassification, GPT2Tokenizer)


class T5CodeReviewModel(CodeReviewModel):
    def __init__(self):
        super().__init__("t5-small", T5ForConditionalGeneration, T5Tokenizer)

    def tokenize(self, batch):
        tokenized_inputs = self.tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt")
        tokenized_outputs = self.tokenizer(batch["labels"], padding=True, truncation=True, return_tensors="pt")
        tokenized = {**tokenized_inputs, "decoder_input_ids": tokenized_outputs["input_ids"]}
        return tokenized
        
    def get_collator(self):
        return DataCollatorForTokenClassification(tokenizer=self.tokenizer)