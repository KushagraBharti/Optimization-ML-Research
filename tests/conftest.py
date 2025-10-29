from __future__ import annotations

import pytest

try:
    from hypothesis import HealthCheck, settings
except ImportError:  # pragma: no cover
    settings = None
    HealthCheck = None

try:
    from coverage_planning.common.constants import (
        EPS_GEOM,
        RNG_SEEDS,
        TOL_NUM,
        seed_everywhere,
    )
except ImportError:  # pragma: no cover - layout fallback
    import sys
    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    from coverage_planning.common.constants import (
        EPS_GEOM,
        RNG_SEEDS,
        TOL_NUM,
        seed_everywhere,
    )


TEST_SEED = RNG_SEEDS.get("tests", 1337)
seed_everywhere(TEST_SEED)

if settings is not None:  # pragma: no branch
    settings.register_profile(
        "ci",
        max_examples=60,
        deadline=None,
        seed=TEST_SEED,
        print_blob=True,
        suppress_health_check=(
            HealthCheck.filter_too_much,
            HealthCheck.too_slow,
        ),
    )
    settings.load_profile("ci")


def pytest_configure(config: pytest.Config) -> None:  # pragma: no cover
    config.addinivalue_line("markers", "slow: mark test as slow")


@pytest.fixture(scope="session")
def tol() -> float:
    return TOL_NUM


@pytest.fixture(scope="session")
def eps() -> float:
    return EPS_GEOM
