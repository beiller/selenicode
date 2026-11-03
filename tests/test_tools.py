import json
import os
import struct
import sys
import tempfile
import unittest
import zlib
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config
from core.tools import edit_file, read_file, write_file


def ok(result: str) -> dict:
    return json.loads(result)


class FileToolsBase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "sample.txt")

    def write_raw(self, content: str, bom: bytes = b""):
        with open(self.path, "wb") as f:
            f.write(bom + content.encode("utf-8"))

    def read_raw(self) -> str:
        with open(self.path, "rb") as f:
            data = f.read()
        for bom in (b"\xef\xbb\xbf", b"\xff\xfe", b"\xfe\xff"):
            if data.startswith(bom):
                return data[len(bom) :].decode("utf-8")
        return data.decode("utf-8")


class TestReadFile(FileToolsBase):
    def test_basic_read(self):
        self.write_raw("hello\nworld\n")
        r = ok(read_file(self.path))
        self.assertEqual(r["status"], "success")
        self.assertIn("1: hello", r["result"])
        self.assertIn("2: world", r["result"])

    def test_offset_and_limit(self):
        self.write_raw("one\ntwo\nthree\nfour\n")
        r = ok(read_file(self.path, offset=2, limit=2))
        self.assertIn("2: two", r["result"])
        self.assertIn("3: three", r["result"])
        self.assertNotIn("4: four", r["result"])
        self.assertIn("Use offset=4 to continue", r["result"])

    def test_missing_file_with_suggestion(self):
        self.write_raw("x\n")
        r = ok(read_file(self.path + ".extra"))
        self.assertIn("Did you mean", r["error"])
        self.assertIn(self.path, r["error"])

    def test_directory_listing(self):
        os.makedirs(os.path.join(self.dir.name, "subdir"))
        with open(os.path.join(self.dir.name, "a.txt"), "w") as f:
            f.write("x\n")
        r = ok(read_file(self.dir.name))
        self.assertIn("subdir/", r["result"])
        self.assertIn("a.txt", r["result"])

    def test_binary_rejected(self):
        with open(os.path.join(self.dir.name, "blob.bin"), "wb") as f:
            f.write(b"\x00\x01\x02data")
        r = ok(read_file(os.path.join(self.dir.name, "blob.bin")))
        self.assertIn("binary", r["error"].lower())

    def test_long_line_truncated(self):
        self.write_raw("start\n" + "x" * 2500 + "\nend\n")
        r = ok(read_file(self.path))
        self.assertIn("truncated to 2000 chars", r["result"])


class TestWriteFile(FileToolsBase):
    def test_write_new_file(self):
        r = ok(write_file(self.path, "hello\n"))
        self.assertTrue(r["success"])
        self.assertEqual(self.read_raw(), "hello\n")

    def test_creates_parent_dirs(self):
        nested = os.path.join(self.dir.name, "a", "b", "c.txt")
        r = ok(write_file(nested, "deep\n"))
        self.assertTrue(r["success"])
        with open(nested) as f:
            self.assertEqual(f.read(), "deep\n")

    def test_overwrite_produces_diff(self):
        self.write_raw("old line\n")
        r = ok(write_file(self.path, "new line\n"))
        self.assertTrue(r["success"])
        self.assertIn("-old line", r["diff"])
        self.assertIn("+new line", r["diff"])

    def test_bom_preserved_on_overwrite(self):
        self.write_raw("old\n", bom=b"\xef\xbb\xbf")
        ok(write_file(self.path, "new\n"))
        with open(self.path, "rb") as f:
            self.assertTrue(f.read().startswith(b"\xef\xbb\xbf"))


class TestEditFile(FileToolsBase):
    def test_exact_match(self):
        self.write_raw("alpha\nbeta\ngamma\n")
        r = ok(edit_file(self.path, "beta", "BETA"))
        self.assertTrue(r["success"])
        self.assertEqual(self.read_raw(), "alpha\nBETA\ngamma\n")
        self.assertIn("-beta", r["diff"])
        self.assertIn("+BETA", r["diff"])

    def test_multiple_matches_fails(self):
        self.write_raw("dup\ndup\n")
        r = ok(edit_file(self.path, "dup", "x"))
        self.assertIn("multiple", r["error"].lower())

    def test_replace_all(self):
        self.write_raw("foo\nfoo\nfoo\n")
        r = ok(edit_file(self.path, "foo", "bar", True))
        self.assertTrue(r["success"])
        self.assertEqual(self.read_raw(), "bar\nbar\nbar\n")

    def test_not_found(self):
        self.write_raw("present\n")
        r = ok(edit_file(self.path, "absent", "x"))
        self.assertIn("could not find", r["error"].lower())

    def test_empty_old_string_replaces_whole_file(self):
        self.write_raw("old content\nmore\n")
        r = ok(edit_file(self.path, "", "brand new\n"))
        self.assertTrue(r["success"])
        self.assertEqual(self.read_raw(), "brand new\n")

    def test_indentation_flexible_match(self):
        self.write_raw("class A:\n    def m(self):\n        return 1\n")
        r = ok(edit_file(self.path, "def m(self):\n      return 1", "def m(self):\n        return 99"))
        self.assertTrue(r["success"])
        self.assertIn("return 99", self.read_raw())

    def test_crlf_preserved(self):
        with open(self.path, "wb") as f:
            f.write(b"line one\r\nline two\r\n")
        r = ok(edit_file(self.path, "line one", "LINE ONE"))
        self.assertTrue(r["success"])
        with open(self.path, "rb") as f:
            data = f.read()
        self.assertIn(b"LINE ONE\r\n", data)

    def test_emoji_and_special_chars_in_match_and_replacement(self):
        self.write_raw("greeting: héllo 🎉\nsecond: 日本語\n")
        r = ok(edit_file(self.path, "héllo 🎉", "wörld 💯"))
        self.assertTrue(r["success"])
        self.assertEqual(self.read_raw(), "greeting: wörld 💯\nsecond: 日本語\n")

    def test_additions_deletions_counts(self):
        self.write_raw("one\ntwo\nthree\n")
        r = ok(edit_file(self.path, "two", "TWO"))
        self.assertTrue(r["success"])
        self.assertEqual(r["additions"], 1)
        self.assertEqual(r["deletions"], 1)


def make_png() -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + b"\xff\x00\x00\xff" * 4 for _ in range(4))
    ihdr = struct.pack(">IIBBBBB", 4, 4, 8, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


class TestReadFileImage(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.vision_patcher = mock.patch.object(config, "VISION_SUPPORTED", True)
        self.vision_patcher.start()

    def write_png(self, name: str) -> str:
        path = os.path.join(self.dir.name, name)
        with open(path, "wb") as f:
            f.write(make_png())
        return path

    def test_png_returns_marker(self):
        path = self.write_png("pic.png")
        r = ok(read_file(path))
        self.assertEqual(r["type"], "image")
        self.assertEqual(r["path"], path)
        self.assertEqual(r["mime"], "image/png")

    def test_jpeg_magic_without_extension(self):
        path = os.path.join(self.dir.name, "data.bin")
        with open(path, "wb") as f:
            f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 16)
        r = ok(read_file(path))
        self.assertEqual(r["type"], "image")
        self.assertEqual(r["mime"], "image/jpeg")

    def test_text_file_unchanged(self):
        path = os.path.join(self.dir.name, "sample.txt")
        with open(path, "w") as f:
            f.write("hello\n")
        r = ok(read_file(path))
        self.assertEqual(r["status"], "success")

    def test_over_cap_rejected(self):
        path = self.write_png("big.png")
        with mock.patch.object(config, "MAX_IMAGE_BYTES", 10):
            r = ok(read_file(path))
        self.assertIn("too large for vision", r["error"])

    def test_vision_off_rejected(self):
        path = self.write_png("pic.png")
        with mock.patch.object(config, "VISION_SUPPORTED", False):
            r = ok(read_file(path))
        self.assertIn("does not support vision", r["error"])


if __name__ == "__main__":
    unittest.main()
