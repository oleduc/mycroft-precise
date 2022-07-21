#!/usr/bin/env python3
# Copyright 2019 Mycroft AI Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import shutil

import pytest

from precise.scripts.train import TrainScript
from test.scripts.test_utils.temp_folder import TempFolder
from test.scripts.test_utils.dummy_train_folder import DummyTrainFolder

@pytest.fixture()
def train_folder():
    folder = DummyTrainFolder()
    folder.generate_default()
    try:
        yield folder
    finally:
        folder.cleanup()


@pytest.fixture()
def temp_folder():
    folder = TempFolder()
    try:
        yield folder
    finally:
        folder.cleanup()


@pytest.fixture(scope='session')
def _trained_model():
    """Session wide model that gets trained once"""
    folder = DummyTrainFolder()
    folder.generate_default()
    script = TrainScript.create(model=folder.model, folder=folder.root, epochs=100)
    script.run()
    try:
        yield folder.model
    finally:
        folder.cleanup()


@pytest.fixture()
def trained_model(_trained_model, temp_folder):
    """Copy of session wide model"""
    model = temp_folder.path('trained_model.h5')
    shutil.copy(_trained_model, model)
    shutil.copy(_trained_model + '.params', model + '.params')
    return model