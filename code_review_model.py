import os
import threading
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

import datasets
import pickle5 as pickle
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from transformers import (BertForSequenceClassification, BertTokenizer,
                          DataCollatorForTokenClassification,
                          DataCollatorWithPadding,
                          GPT2ForSequenceClassification, GPT2Tokenizer,
                          PreTrainedModel, T5ForConditionalGeneration,
                          T5Tokenizer, Trainer, TrainingArguments)

from merge_request_fetcher import GitHubGitLabFetcher
from typing import Dict, List

dataset_lock = threading.Lock()

# from lock import use_custom_tqdm

# datasets.disable_progress_bar()

load_dataset_base = datasets.load_dataset 
DatasetDict = datasets.DatasetDict

class CodeReviewDataset(Dataset):
    def __init__(self, tokenized_data, model, tokenizer):
        self.tokenized_data = tokenized_data
        self.model = model
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        idxData = self.tokenized_data[idx]
        input_ids = idxData["input_ids"]
        attention_mask = idxData["attention_mask"]

        if "decoder_input_ids" in idxData:
            labels = idxData["decoder_input_ids"]
        elif "token_type_ids" in idxData:
            labels = idxData["token_type_ids"]
        else:
            labels = input_ids.copy()

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    
class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [example["input_ids"] for example in batch]
        attention_mask = [example["attention_mask"] for example in batch]
        labels = [example["labels"] for example in batch]

        max_length = max([x.size(0) for x in input_ids])

        padded_input_ids = [torch.cat((x, x.new_full((max_length - x.size(0),), self.tokenizer.pad_token_id))) for x in input_ids]
        padded_attention_mask = [torch.cat((x, x.new_zeros(max_length - x.size(0)))) for x in attention_mask]
        padded_labels = [torch.cat((x, x.new_full((max_length - x.size(0),), -100))) for x in labels]

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels)
        }
    
# def custom_training_step(trainer, model: PreTrainedModel, inputs):
#     """
#     Custom training step for the Trainer that computes the loss manually.
#     """
#     model.train()
#     for key, value in inputs.items():
#         if isinstance(value, torch.Tensor):
#             inputs[key] = value.to(trainer.args.device)

#     # Get the logits from the model
#     outputs = model(**inputs)
#     logits = outputs.logits

#     # Get the labels
#     labels = inputs["labels"]

#     # Compute the loss
#     loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
#     active_loss = labels.view(-1) != -100
#     active_logits = logits.view(-1, model.config.num_labels).masked_select(active_loss.unsqueeze(1)).view(-1, model.config.num_labels)
#     active_labels = torch.where(
#         active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
#     )[active_loss]

#     # Debugging print statements
#     print("Logits shape: ", logits.shape)
#     print("Labels shape: ", labels.shape)
#     print("Active loss shape: ", active_loss.shape)
#     print("Active logits shape: ", active_logits.shape)
#     print("Active labels shape: ", active_labels.shape)

#     loss = loss_fct(active_logits, active_labels)

#     return loss

# def custom_training_step(trainer, model: PreTrainedModel, inputs):
#     """
#     Custom training step for the Trainer that computes the loss manually.
#     """
#     model.train()
#     for key, value in inputs.items():
#         if isinstance(value, torch.Tensor):
#             inputs[key] = value.to(trainer.args.device)

#     # Get the logits from the model
#     outputs = model(**inputs)
#     logits = outputs.logits

#     # Get the labels
#     labels = inputs["labels"]

#     # Compute the loss
#     loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
#     active_loss = labels.view(-1) != -100
#     active_logits = logits.view(-1, model.config.num_labels)[active_loss]
#     active_labels = labels.view(-1)[active_loss]

#     loss = loss_fct(active_logits, active_labels)

#     return loss

def custom_training_step(trainer, model: PreTrainedModel, inputs):
    """
    Custom training step for the Trainer that computes the loss manually.
    """
    model.train()
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(trainer.args.device)

    # Get the logits from the model
    outputs = model(**inputs)
    logits = outputs.logits

    # Get the labels
    labels = inputs["labels"]

    # Debugging print statements
    print("Logits shape: ", logits.shape)
    print("Labels shape: ", labels.shape)

    # Compute the loss
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    active_loss = labels.view(-1) != -100
    active_logits = logits.view(-1, model.config.num_labels)[active_loss]
    active_labels = torch.where(
        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
    )[active_loss]

    # Debugging print statements
    print("Active loss shape: ", active_loss.shape)
    print("Active logits shape: ", active_logits.shape)
    print("Active labels shape: ", active_labels.shape)

    loss = loss_fct(active_logits, active_labels)

    return loss
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
        tokenized = self.tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        if "labels" in batch:
            tokenized["labels"] = batch["labels"]
        return tokenized
    
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
                try:
                    with open(f"{output_dir}/{split}.pkl", "rb") as f:
                        prepared_dataset = pickle.load(f)
                except Exception as e:
                    print("Error in loading prepared dataset : ", e, "\n\n File Path : ", f"{output_dir}/{split}.pkl")
                    prepared_dataset = None
                    break
        return prepared_dataset

    def fetch_reviews(self, github_token, gitlab_token, gitlab_organization_id):
        fetcher = GitHubGitLabFetcher(github_token, gitlab_token, gitlab_organization_id)
        return fetcher.fetch_reviews()
    
    def get_collator(self):
        return CustomDataCollator(tokenizer=self.tokenizer)
    
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

        # Create instances of CodeReviewDataset for training and validation
        train_code_review_dataset = CodeReviewDataset(train_dataset, self.model, self.tokenizer)
        val_code_review_dataset = CodeReviewDataset(eval_dataset, self.model, self.tokenizer)

        # Create data loaders for training and validation datasets
        # train_dataloader = DataLoader(train_code_review_dataset, batch_size=train_batch_size, shuffle=True)
        # val_dataloader = DataLoader(val_code_review_dataset, batch_size=eval_batch_size, shuffle=False)


        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_code_review_dataset,
            eval_dataset=val_code_review_dataset,
            data_collator=data_collator,
        )
        
        # Set the custom training step for the trainer
        self.trainer.training_step = custom_training_step.__get__(self.trainer)
        self.trainer.train()


    def evaluate(self):
        return self.trainer.evaluate()


class BERTCodeReviewModel(CodeReviewModel):
    def __init__(self):
        super().__init__("bert-base-uncased", BertForSequenceClassification, BertTokenizer)


class GPT2CodeReviewModel(CodeReviewModel):
    def __init__(self):
        super().__init__("gpt2", GPT2ForSequenceClassification, GPT2Tokenizer)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Update the model's configuration
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

class T5CodeReviewModel(CodeReviewModel):
    def __init__(self):
        super().__init__("t5-small", T5ForConditionalGeneration, T5Tokenizer)

    def tokenize(self, batch):
        tokenized = self.tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt")
        if "labels" in batch:
            tokenized["labels"] = batch["labels"]
        return tokenized
        
    def get_collator(self):
        return DataCollatorForTokenClassification(tokenizer=self.tokenizer)