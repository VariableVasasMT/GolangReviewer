
import subprocess
from preprocessor import Preprocessor

class RepositoryManager:
    @staticmethod
    def clone_repository(clone_url, local_path):
        try:
            subprocess.run(["git", "clone", clone_url, local_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository {clone_url}: {e}")

    def clone_and_preprocess(self, github_repos, gitlab_repos):
        repo_paths = []

        for repo in github_repos['items'] + gitlab_repos:
            clone_url = repo["ssh_url"] if "ssh_url" in repo else repo["ssh_url_to_repo"]
            local_path = f"cloned_repos/{repo['name']}"
            self.clone_repository(clone_url, local_path)
            repo_paths.append(local_path)

        preprocessed_files = Preprocessor.preprocess_repositories(repo_paths)

        with open("preprocessed_golang_code.txt", "w") as f:
            for code in preprocessed_files:
                f.write(code["normalized_code"] + "\n")
