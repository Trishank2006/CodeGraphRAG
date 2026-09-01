from pathlib import Path
from git import Repo


def clone_repository(repo_url: str, destination: str) -> Path:
    """
    Clone a Git repository into the given destination directory.

    Args:
        repo_url: URL of the Git repository.
        destination: Local directory where the repository will be cloned.

    Returns:
        Path object representing the cloned repository.
    """

    destination_path = Path(destination)

    if destination_path.exists():
        if any(destination_path.iterdir()):
            raise FileExistsError(
                f"Destination already exists and is not empty: {destination}"
            )
    else:
        destination_path.mkdir(parents=True)

    Repo.clone_from(repo_url, destination_path)

    return destination_path