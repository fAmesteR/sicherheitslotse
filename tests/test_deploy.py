import importlib.util
from types import SimpleNamespace

import pytest

from lotse.config import ROOT

spec = importlib.util.spec_from_file_location("deploy_space", ROOT / "scripts" / "deploy_space.py")
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


class Space:
    """The Space's runtime as the Hub reports it, one entry per poll."""

    def __init__(self, *runtimes: tuple[str, str | None]):
        self.runtimes = list(runtimes)

    def get_space_runtime(self, repo: str) -> SimpleNamespace:
        stage, sha = self.runtimes.pop(0) if len(self.runtimes) > 1 else self.runtimes[0]
        return SimpleNamespace(stage=stage, raw={"sha": sha, "errorMessage": "it broke"})


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(deploy.time, "sleep", lambda seconds: None)


def test_the_wait_ends_once_the_space_runs_the_new_commit():
    space = Space(("RUNNING_BUILDING", "old"), ("RUNNING_APP_STARTING", None), ("RUNNING", "new"))
    deploy.wait_until_running(space, "new")


def test_a_start_that_fails_after_the_upload_stops_the_wait_at_once():
    # A failed Space reports no commit, which once made the script wait its full half hour.
    space = Space(("RUNNING_BUILDING", "old"), ("RUNTIME_ERROR", None))
    with pytest.raises(SystemExit, match="RUNTIME_ERROR: it broke"):
        deploy.wait_until_running(space, "new", limit=60)


def test_an_error_left_from_an_earlier_deploy_is_waited_out():
    space = Space(("RUNTIME_ERROR", None), ("BUILDING", None), ("RUNNING", "new"))
    deploy.wait_until_running(space, "new")
