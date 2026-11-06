import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import paths


class PortablePathTests(unittest.TestCase):
    def test_relative_resource_override_is_independent_of_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            before = Path.cwd()
            try:
                os.chdir(directory)
                with mock.patch.dict(os.environ, {'VOICE_TEST_RESOURCE_DIR': 'custom/models'}):
                    self.assertEqual(paths.location('VOICE_TEST_RESOURCE_DIR', Path(directory)),
                                     paths.ROOT/'custom/models')
            finally:
                os.chdir(before)

    def test_config_falls_back_to_example_without_creating_personal_settings(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(paths, 'ROOT', Path(directory)):
            self.assertEqual(paths.config_path(), Path(directory)/'config.example.json')
            custom = Path(directory)/'config.json'
            self.assertFalse(custom.exists())
            custom.write_text('{}', encoding='utf-8')
            self.assertEqual(paths.config_path(), custom)
