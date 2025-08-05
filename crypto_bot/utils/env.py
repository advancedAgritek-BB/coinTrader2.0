import os


def env_or_prompt(name: str, prompt: str) -> str:
    """Return the value of an environment variable or prompt the user."""
    return os.getenv(name) or input(prompt)
