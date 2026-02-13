import unittest

from rag.traglast_rag import _extract_long_records_from_matrix


class TraglastParserTest(unittest.TestCase):
    def test_extract_long_records_from_bkl_like_matrix(self):
        matrix = [
            ["m", "", "33,9", "", "51", "", "", "54", "", "", "m"],
            ["m", "0°", "10°", "20°", "0°", "10°", "20°", "0°", "10°", "20°", "m"],
            ["10", "54,9", "", "", "", "", "", "", "", "", "10"],
            ["14", "47,3", "", "", "49,1", "", "", "48,6", "", "", "14"],
            ["20", "38,9", "32,9", "28,2", "41,7", "34,3", "29", "41,7", "34,5", "28,8", "20"],
        ]

        records = _extract_long_records_from_matrix(matrix, pdf_name="sample.pdf", page_idx=1, table_idx=1)

        self.assertTrue(records)
        hit = [r for r in records if r["radius_m"] == 20.0 and r["boom_length_m"] == 33.9 and r["angle_deg"] == 20.0]
        self.assertEqual(len(hit), 1)
        self.assertEqual(hit[0]["load_t"], 28.2)


if __name__ == "__main__":
    unittest.main()
