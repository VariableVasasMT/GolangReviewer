# main.py

from repository_fetcher import RepositoryFetcher
from repository_manager import RepositoryManager
from code_review_model import BERTCodeReviewModel, GPT2CodeReviewModel, T5CodeReviewModel
import threading
import os
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import pprint

load_dotenv()

# Set stage flags
fetch_repos = False
preprocess_repos = False
analyze_existing = True
train_models = True
evaluate_models = True
force_tokenize = False
prepare_dataset = False


gitlab_token = os.getenv("GITLAB_TOKEN")
github_token = os.getenv("GITHUB_TOKEN")
gitlab_organization_id = os.getenv("GITLAB_ORGANIZATION_ID")

print("Environment variables:")
pprint.pprint(dict(os.environ))

# Stage 1: Fetch repositories from GitHub and GitLab
if fetch_repos:
    fetcher = RepositoryFetcher(github_token, gitlab_token, gitlab_organization_id)
    github_repos, gitlab_repos = fetcher.fetch_repositories()

# Stage 2: Clone and preprocess the repositories
if preprocess_repos:
    manager = RepositoryManager()
    repo_paths = manager.clone_and_preprocess(github_repos, gitlab_repos)

# if analyze_existing:
#     analyzer = GitHubGitLabFetcher(github_token, gitlab_token)
#     analyzer.analyze_pull_requests(github_repos, gitlab_repos)

# Stage 3: Instantiate different models and train them to compete with each other

def tokenize_model(model, train_dataset, validation_dataset, output_dir):
    if not os.path.exists(output_dir) or force_tokenize:
        model.tokenize_dataset(train_dataset, validation_dataset, output_dir)
    return output_dir

if train_models:
    dataset_path = "preprocessed_golang_code.txt"
    models = [
        BERTCodeReviewModel(),
        GPT2CodeReviewModel(),
        T5CodeReviewModel(),
    ]

    # Load the dataset
    dataset = models[0].prepare_dataset(dataset_path)
    train_dataset = dataset['train']
    validation_dataset = dataset['validation']

    # Tokenize the dataset and store it in a file, if it doesn't exist or if force_tokenize is set
    tokenized_dirs = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for i, model in enumerate(models):
            output_dir = f"tokenized_data_{model.model_name}"
            future = executor.submit(tokenize_model, model, train_dataset, validation_dataset, output_dir)
            futures.append(future)

        for future in futures:
            tokenized_dirs.append(future.result())

    # Use separate threads to train each model
    threads = []
    for i, model in enumerate(models):
        print(f"Training {model.model_name}...")

        # Load the tokenized dataset from a file
        tokenized_dataset = model.load_tokenized_dataset(tokenized_dirs[i])

        # Train the model in a separate thread
        t = threading.Thread(target=train_model_thread, args=(model, tokenized_dataset["train"], tokenized_dataset["validation"]))
        t.start()
        threads.append(t)

    # Wait for all threads to finish
    for t in threads:
        t.join()

    print("All models have finished training.")

# Stage 4: Evaluate and compare the performance of the models
if evaluate_models:
    for model in models:
        evaluation_results = model.evaluate()
        print(f"{model.model_name} evaluation results: {evaluation_results}")