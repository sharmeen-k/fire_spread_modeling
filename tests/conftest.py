from __future__ import annotations

import sys

import pytest

from firespread import config, http

from . import fakes, synth


@pytest.fixture
def web(monkeypatch):
    w = fakes.FakeWeb()
    monkeypatch.setattr(http, "get", w)
    return w


@pytest.fixture
def fake_herbie(monkeypatch):
    fakes.FakeHerbie.calls.clear()
    monkeypatch.setitem(sys.modules, "herbie", fakes.herbie_module)
    return fakes.FakeHerbie


@pytest.fixture
def fake_nfdrs4(monkeypatch):
    monkeypatch.setitem(sys.modules, "nfdrs4", fakes.nfdrs4_module)
    return fakes.nfdrs4_module


@pytest.fixture(scope="session")
def project(tmp_path_factory):
    """Synthetic project root with config and raw inputs (built once per session)."""
    root = tmp_path_factory.mktemp("project")
    path = synth.build(root)
    return path


@pytest.fixture
def cfg(project, monkeypatch):
    monkeypatch.setenv(config.CONFIG_ENV, str(project))
    monkeypatch.delenv("SYNOPTIC_TOKEN", raising=False)
    return config.load(project)
