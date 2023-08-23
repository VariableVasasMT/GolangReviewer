# main.py

import faulthandler
import json
import os
import pprint

from dotenv import load_dotenv

from repository_fetcher import RepositoryFetcher
from repository_manager import RepositoryManager
from utils import (prepare_datasets_parallel, prepare_models,
                   train_models_in_threads)

faulthandler.enable()

load_dotenv()

# Set stage flags
fetch_repos = False
preprocess_repos = False
force_prepare_dataset = False
train_models = True
evaluate_models = True


gitlab_token = os.getenv("GITLAB_TOKEN")
github_token = os.getenv("GITHUB_TOKEN")
gitlab_organization_id = os.getenv("GITLAB_ORGANIZATION_ID")

print("Environment variables:")
pprint.pprint(dict(os.environ))

# Stage 1: Fetch repositories from GitHub and GitLab
repo_info_file = "repo_info.json"

if fetch_repos:
    fetcher = RepositoryFetcher(github_token, gitlab_token, gitlab_organization_id)
    github_repos, gitlab_repos = fetcher.fetch_repositories()

    # Save repository information to a JSON file
    repo_info = [{"name": repo, "source": "github"} for repo in github_repos] + [{"name": repo, "source": "gitlab"} for repo in gitlab_repos]

    with open(repo_info_file, "w") as f:
        json.dump(repo_info, f)
else:
    # Load already cloned repositories from the JSON file
    with open(repo_info_file, "r") as f:
        repo_info = json.load(f)

    github_repos = [repo["name"] for repo in repo_info if repo["source"] == "github"]
    gitlab_repos = [repo["name"] for repo in repo_info if repo["source"] == "gitlab"]

# Stage 2: Clone and preprocess the repositories
if preprocess_repos:
    manager = RepositoryManager()
    repo_paths = manager.clone_and_preprocess(github_repos, gitlab_repos)

# Stage 3: Prepare datasets and train the models
if train_models:
    print("Training models...")
    dataset_path = "preprocessed_golang_code.txt"
    models = prepare_models()

    # Load the dataset
    prepared_datasets = prepare_datasets_parallel(models, dataset_path, force_prepare_dataset)

    # Use separate threads to train each model
    train_models_in_threads(models)

    print("All models have finished training.")


# Stage 4: Evaluate and compare the performance of the models
if evaluate_models:
    for model in models:
        evaluation_results = model.evaluate()
        print(f"{model.model_name} evaluation results: {evaluation_results}")
