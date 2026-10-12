from typing import Protocol


class Pipeline(Protocol):
    """What a format's code/pipeline.py exposes as PIPELINE. The engine knows nothing else."""

    format_key: str
    delegated: bool            # True: clips come from OpenShorts via quillcut.delegate
    auto_template: str         # script template used by `quillcut run` (native formats)
