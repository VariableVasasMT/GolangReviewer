import requests

class GitHubGitLabFetcher:
    def __init__(self, github_token, gitlab_token, gitlab_organization_id):
        self.github_token = github_token
        self.gitlab_token = gitlab_token
        self.gitlab_organization_id = gitlab_organization_id

    def fetch_github_repositories(self):
        headers = {"Authorization": f"Bearer {self.github_token}"}
        response = requests.get("https://api.github.com/user/repos", headers=headers)
        repos = response.json()
        return [{"url": repo["url"], "name": repo["name"], "platform": "github"} for repo in repos]

    def fetch_gitlab_repositories(self):
        headers = {"Authorization": f"Bearer {self.gitlab_token}"}
        response = requests.get(f"https://gitlab.com/api/v4/groups/{self.gitlab_organization_id}/projects?language=Go&order_by=created_at&sort=desc&per_page=5", headers=headers)
        repos = response.json()
        return [{"url": repo["web_url"], "name": repo["name"], "platform": "gitlab"} for repo in repos]

    def fetch_repositories(self):
        github_repos = self.fetch_github_repositories()
        gitlab_repos = self.fetch_gitlab_repositories()
        return github_repos, gitlab_repos

    def get_closed_pull_requests_with_reviews(self, repo):
        closed_pull_requests = []
        if repo["platform"] == "github":
            headers = {"Authorization": f"Bearer {self.github_token}"}
            response = requests.get(f"{repo['url']}/pulls?state=closed", headers=headers)
            pull_requests = response.json()

            for pr in pull_requests:
                pr_response = requests.get(pr["_links"]["self"]["href"], headers=headers)
                pr_data = pr_response.json()
                if pr_data["state"] == "closed" and pr_data["review_comments"] >= 4:
                    closed_pull_requests.append(pr_data)

        elif repo["platform"] == "gitlab":
            headers = {"Authorization": f"Bearer {self.gitlab_token}"}
            response = requests.get(f"{repo['url']}/merge_requests?state=closed", headers=headers)
            pull_requests = response.json()

            for pr in pull_requests:
                pr_response = requests.get(f"{repo['url']}/merge_requests/{pr['iid']}", headers=headers)
                pr_data = pr_response.json()
                if pr_data["state"] == "closed" and pr_data["user_notes_count"] >= 4:
                    closed_pull_requests.append(pr_data)

        return closed_pull_requests
