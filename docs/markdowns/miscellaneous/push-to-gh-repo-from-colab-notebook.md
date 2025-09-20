# Connect to a GitHub Repository and push changes from a Google Colab Notebook

```py
!git config --global user.email "your_github_email"
!git config --global user.name "your_github_username"
```

You must set your GitHub token as a secret in your google colab notebook.

```py
import os
from google.colab import userdata
os.environ['GITHUB_TOKEN'] = userdata.get('GITHUB_TOKEN')
!gh auth login --with-token <<< $GITHUB_TOKEN
```

```py
GITHUB_EMAIL = "your_github_email"
GITHUB_USERNAME = "your_github_username"
GITHUB_REPO_NAME = "your_repository_name"
GITHUB_TOKEN = os.environ['GITHUB_TOKEN']
GITHUB_REPO_REMOTE_URL = f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{GITHUB_REPO_NAME}.git"

!git config --global user.email {GITHUB_EMAIL}
!git config --global user.name {GITHUB_USERNAME}
!gh repo clone {GITHUB_USERNAME}/{GITHUB_REPO_NAME} /content/{GITHUB_REPO_NAME}
%cd /content/{GITHUB_REPO_NAME}
!git remote set-url origin {GITHUB_REPO_REMOTE_URL}
!git log
```

Now You have already cloned your GitHub repository to your Google Colab environment. You can now add, commit, and push changes to your repository.
