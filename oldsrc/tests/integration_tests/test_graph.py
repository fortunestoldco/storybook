# test_graph.py

import unittest
from storybook.graph import StoryBookGraph

class TestStoryBookGraph(unittest.TestCase):

    def setUp(self):
        self.graph = StoryBookGraph()

    def test_graph_initialization(self):
        self.assertIsNotNone(self.graph)

    def test_graph_run(self):
        initial_state = {"project_requirements": "Write a fantasy novel"}
        result = self.graph.run(initial_state)
        self.assertIn("supervisor_report", result)

if __name__ == "__main__":
    unittest.main()
