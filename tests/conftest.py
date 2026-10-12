import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

from quillcut.paths import apply_cache_env  # noqa: E402

apply_cache_env()


@pytest.fixture(scope="session")
def media():
    from make_fixtures import make_all
    return make_all()
