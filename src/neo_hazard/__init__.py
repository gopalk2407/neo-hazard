def main() -> None:
    import json
    from .rank import rank
    from .train import train

    print(json.dumps(train(), indent=2, default=float))
    print(rank(25).to_string(index=False))
