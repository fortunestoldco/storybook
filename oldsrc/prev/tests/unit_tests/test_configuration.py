# test_configuration.py

import unittest
from storybook.configuration import settings

class TestConfiguration(unittest.TestCase):

    def test_settings_initialization(self):
        self.assertEqual(settings.database_name, "creative_writing_db")
        self.assertEqual(settings.max_concurrency, 5)
        self.assertEqual(settings.logging_level, "INFO")

if __name__ == "__main__":
    unittest.main()
