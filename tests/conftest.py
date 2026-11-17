import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from syco.data import read_mcq, read_opinion  # noqa: E402
from syco.eval.harness import EvalData  # noqa: E402
from syco.models.fake import FakeModel  # noqa: E402

SAMPLES = ROOT / "data" / "samples"


@pytest.fixture(scope="session")
def samples() -> EvalData:
    return EvalData(
        pressure=read_mcq(SAMPLES / "pressure.jsonl"),
        capability=read_mcq(SAMPLES / "capability.jsonl"),
        opinion=read_opinion(SAMPLES / "opinion.jsonl"),
    )


@pytest.fixture
def make_fake(samples):
    def make(**kw) -> FakeModel:
        kw.setdefault("knowledge", 1.0)
        return FakeModel(mcq=[*samples.pressure, *samples.capability], opinions=samples.opinion, **kw)

    return make
