import unittest
from .runners import Runners


class TestRunners(unittest.TestCase):
    def test_runners(self):
        runners = Runners()
        print(runners.runners)
