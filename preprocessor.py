
import os
import subprocess

class Preprocessor:
    @staticmethod
    def get_go_files(repo_path):
        go_files = []
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith(".go"):
                    go_files.append(os.path.join(root, file))
        return go_files

    @staticmethod
    def normalize_formatting(file_path):
        try:
            print(f"Formatting file {file_path}...")
            result = subprocess.run(["gofmt", "-s", file_path], check=True, capture_output=True, text=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Error formatting file {file_path}: {e}")
            print("Error output:", e.stderr)
            return None

    @staticmethod
    def preprocess_repositories(repo_paths):
        preprocessed_files = []

        for repo_path in repo_paths:
            go_files = Preprocessor.get_go_files(repo_path)
            for file_path in go_files:
                normalized_code = Preprocessor.normalize_formatting(file_path)
                if normalized_code:
                    file_data = {
                        "file_path": file_path,
                        "normalized_code": normalized_code,
                    }
                    preprocessed_files.append(file_data)

        return preprocessed_files