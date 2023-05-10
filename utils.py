from code_review_model import BERTCodeReviewModel, GPT2CodeReviewModel, T5CodeReviewModel
import threading

def prepare_models():
    models = [
        BERTCodeReviewModel(),
        GPT2CodeReviewModel(),
        T5CodeReviewModel(),
    ]
    return models

def train_model(model, train_dataset, validation_dataset, output_dir="output"):
    model.train(train_dataset, validation_dataset, output_dir=output_dir)

def prepare_dataset_wrapper(model, dataset_path, force_prepare):
    prepared_dataset = model.prepare_dataset(dataset_path, force_prepare=force_prepare)
    return prepared_dataset
    
def prepare_datasets_parallel(models, dataset_path, force_prepare):
    prepared_datasets = dict()
    for model in models:
        prepared_datasets[model.model_name] = prepare_dataset_wrapper(model, dataset_path, force_prepare=force_prepare)

    return prepared_datasets

def train_models_in_threads(models):
    threads = []
    for i, model in enumerate(models):
        print(f"Training {model.model_name}...")

        output_dir = f"."
        tokenized_dataset = model.load_tokenized_dataset(output_dir)
        print(tokenized_dataset)

        t = threading.Thread(target=train_model, args=(model, tokenized_dataset["train"], tokenized_dataset["validation"]))
        t.start()
        threads.append(t)

    # Wait for all threads to finish
    for t in threads:
        t.join()