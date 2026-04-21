import unittest

from src.summary_builder import SummaryBuilder


class EventPromptTests(unittest.TestCase):
    def test_event_prompt_emphasizes_dense_query_useful_content(self) -> None:
        prompt = SummaryBuilder._build_event_vlm_prompt(
            bulleted="- Scene 1 [0.0s-3.0s]: A person cooks at a stove.",
            n_frames=2,
            n_scenes=1,
            target="roughly 40-80 words, one short paragraph",
        )

        self.assertIn("high-information-density", prompt)
        self.assertIn("Put the most query-useful facts early", prompt)
        self.assertIn("Avoid filler", prompt)
        self.assertIn("mood, professionalism, intensity", prompt)


if __name__ == "__main__":
    unittest.main()
