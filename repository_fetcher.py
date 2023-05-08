import requests

class RepositoryFetcher:
    def __init__(self, github_token, gitlab_token, gitlab_organization_id):
        self.github_token = github_token
        self.gitlab_token = gitlab_token
        self.gitlab_organization_id = gitlab_organization_id

    def fetch_repositories(self):
        github_headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"token {self.github_token}"
        }
        gitlab_headers = {"Authorization": f"Bearer {self.gitlab_token}"}

        search_query = "language:Go"
        sort = "stars"
        order = "desc"
        github_url = f"https://api.github.com/search/repositories?q={search_query}&sort={sort}&order={order}&per_page=5"
        gitlab_url = f"https://gitlab.com/api/v4/groups/{self.gitlab_organization_id}/projects?language=Go&order_by=created_at&sort=desc&per_page=5"

        github_response = requests.get(github_url, headers=github_headers)
        gitlab_response = requests.get(gitlab_url, headers=gitlab_headers)


        if github_response.status_code == 200 and gitlab_response.status_code == 200:
            return github_response.json(), gitlab_response.json()
        else:
            print("In the error")
            print(gitlab_response.json())
            raise Exception(f"GitHub Error: {github_response.status_code}\nGitLab Error: {gitlab_response.status_code}")