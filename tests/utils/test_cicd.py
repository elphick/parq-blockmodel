"""Tests for parq_blockmodel.utils.cicd and parq_blockmodel.utils.plotly_scraper"""
import os
import pytest

from parq_blockmodel.utils.cicd import is_github_runner


def test_is_github_runner_false_by_default():
    # In a local dev environment GITHUB_ACTIONS is not set.
    os.environ.pop("GITHUB_ACTIONS", None)
    assert is_github_runner() is False


def test_is_github_runner_true_when_env_set(monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert is_github_runner() is True


def test_is_github_runner_false_when_env_other_value(monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "false")
    assert is_github_runner() is False

