import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import conversation


class TestArchiveConversation(unittest.TestCase):
    def test_list_content_does_not_crash_and_shows_image_note(self):
        with tempfile.TemporaryDirectory() as d:
            old = conversation.CONVERSATION_DIR
            conversation.CONVERSATION_DIR = pathlib.Path(d)
            try:
                messages = [
                    {"role": "user", "content": "look at this"},
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": [
                            {"type": "text", "text": "Image loaded from /x.png (image/png)"},
                            {"type": "image_ref", "path": "/x.png", "mime": "image/png"},
                        ],
                    },
                ]
                path = conversation.archive_conversation("sess1", messages)
                with open(path) as f:
                    text = f.read()
            finally:
                conversation.CONVERSATION_DIR = old
        self.assertIn("[image: /x.png]", text)
        self.assertIn("Image loaded from /x.png (image/png)", text)


if __name__ == "__main__":
    unittest.main()
