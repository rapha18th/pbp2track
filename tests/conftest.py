"""Shared fixtures: a small synthetic league, its library on disk and a built Library.

Everything here is invented by pbp2track.synth; the tests need no NBA data.
"""
import pandas as pd
import pytest

from pbp2track import synth
from pbp2track.library import Library
from pbp2track.sportvu import build_library


def quiet(_msg):
    pass


@pytest.fixture(scope="session")
def games():
    return synth.league(5, seed=3, period_s=120.0)


@pytest.fixture(scope="session")
def held_id(games):
    return games[-1]["sportvu"]["gameid"]


@pytest.fixture(scope="session")
def held_game(games):
    return games[-1]


@pytest.fixture(scope="session")
def built(games, tmp_path_factory):
    root = tmp_path_factory.mktemp("library")
    pbp = pd.concat([g["pbp"] for g in games], ignore_index=True)
    report = build_library([g["sportvu"] for g in games], pbp, str(root), log=quiet)
    return str(root), report


@pytest.fixture(scope="session")
def library_root(built):
    return built[0]


@pytest.fixture(scope="session")
def library(library_root, held_id):
    lib = Library(library_root, exclude_games=[held_id]).build(log=quiet)
    lib.legacy_to_court()
    return lib
