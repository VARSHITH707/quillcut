"""Gaming format. P0: signal-driven peaks + script templates; VLM understanding comes in P1."""


class GamingPipeline:
    format_key = "gaming"
    delegated = False
    auto_template = "cold_open"


PIPELINE = GamingPipeline()
