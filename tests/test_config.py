import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scholarseek.config import get_api_config, load_dotenv, mask_secret


class ConfigTest(unittest.TestCase):
    def test_load_dotenv_does_not_override_existing_env(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("DASHSCOPE_API_KEY=from_file\nQWEN_MODEL=qwen-plus\n", encoding="utf-8")
            with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "existing"}, clear=False):
                load_dotenv(env_file)

                self.assertEqual(os.environ["DASHSCOPE_API_KEY"], "existing")
                self.assertEqual(os.environ["QWEN_MODEL"], "qwen-plus")

    def test_get_api_config_reads_env(self):
        with patch.dict(
            os.environ,
            {
                "DASHSCOPE_API_KEY": "dash-key",
                "SEMANTIC_SCHOLAR_API_KEY": "s2-key",
                "OPENALEX_EMAIL": "me@example.com",
                "OPENALEX_API_KEY": "openalex-key",
            },
            clear=False,
        ):
            config = get_api_config(load_env_file=False)

        self.assertEqual(config.qwen_api_key, "dash-key")
        self.assertEqual(config.semantic_scholar_api_key, "s2-key")
        self.assertEqual(config.openalex_email, "me@example.com")
        self.assertEqual(config.openalex_api_key, "openalex-key")

    def test_mask_secret(self):
        self.assertEqual(mask_secret(None), "missing")
        self.assertEqual(mask_secret("short"), "configured")
        self.assertEqual(mask_secret("abcdefghijkl"), "abcd...ijkl")

    def test_placeholder_values_are_treated_as_missing(self):
        with patch.dict(
            os.environ,
            {
                "SEMANTIC_SCHOLAR_API_KEY": "replace_with_your_semantic_scholar_api_key",
                "OPENALEX_EMAIL": "your_email@example.com",
                "OPENALEX_API_KEY": "replace_with_your_openalex_api_key",
            },
            clear=False,
        ):
            config = get_api_config(load_env_file=False)

        self.assertIsNone(config.semantic_scholar_api_key)
        self.assertIsNone(config.openalex_email)
        self.assertIsNone(config.openalex_api_key)


if __name__ == "__main__":
    unittest.main()
