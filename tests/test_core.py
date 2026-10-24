import unittest
import io
import tempfile
from pathlib import Path

import app
from fastapi.testclient import TestClient
from app import cosine_similarity, split_text
from pypdf import PdfWriter


class ChunkingTests(unittest.TestCase):
    def test_empty_input_returns_no_chunks(self):
        self.assertEqual(split_text(""), [])

    def test_short_text_stays_in_one_chunk(self):
        self.assertEqual(split_text("one two three", size=5, overlap=1), ["one two three"])

    def test_chunks_overlap_and_keep_all_words(self):
        chunks = split_text("a b c d e f g", size=4, overlap=1)
        self.assertEqual(chunks, ["a b c d", "d e f g"])

    def test_overlap_must_be_smaller_than_chunk_size(self):
        with self.assertRaises(ValueError):
            split_text("a b c", size=2, overlap=2)


class SimilarityTests(unittest.TestCase):
    def test_identical_vectors_have_similarity_one(self):
        self.assertAlmostEqual(cosine_similarity([1, 2], [1, 2]), 1.0)

    def test_orthogonal_vectors_have_similarity_zero(self):
        self.assertEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_zero_vector_is_safe(self):
        self.assertEqual(cosine_similarity([0, 0], [1, 1]), 0.0)


class AppSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.old_db_path = app.DB_PATH
        app.DB_PATH = Path(self.temp_dir.name) / "test.db"
        app.init_db()
        self.client = TestClient(app.app)

    def tearDown(self):
        self.client.close()
        app.DB_PATH = self.old_db_path
        self.temp_dir.cleanup()

    def test_homepage_and_empty_document_list(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        response = self.client.get("/api/documents")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    def test_retrieval_returns_closest_chunk_and_page_metadata(self):
        with app.connection() as db:
            db.execute("INSERT INTO documents(id, filename) VALUES (?, ?)", ("doc-1", "guide.pdf"))
            db.executemany("INSERT INTO chunks(document_id, page, content, embedding) VALUES (?, ?, ?, ?)", [
                ("doc-1", 2, "closest passage", "[1.0, 0.0]"),
                ("doc-1", 3, "less relevant passage", "[0.0, 1.0]"),
            ])
        result = app.retrieve([1.0, 0.1], ["doc-1"])
        self.assertEqual(result[0]["page"], 2)
        self.assertEqual(result[0]["filename"], "guide.pdf")

    def test_pdf_upload_parses_bytes_as_stream(self):
        output = io.BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        writer.write(output)
        response = self.client.post("/api/documents", files={"file": ("blank.pdf", output.getvalue(), "application/pdf")})
        self.assertEqual(response.status_code, 422)
        self.assertIn("No selectable text", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
