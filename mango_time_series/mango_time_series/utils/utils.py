import os


def cast_env_to_bool(env_var: str, default=False) -> bool:
    """Cast an environment variable to a boolean."""
    return (
        os.getenv(env_var).lower() in ["true", "1"] if os.getenv(env_var) else default
    )
