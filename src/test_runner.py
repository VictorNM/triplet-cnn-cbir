import unittest
from .runner import Runner


class TestRunner(unittest.TestCase):
    def test_create_default_runner(self):
        runner = Runner()
        runner.prepare()
        runner.model_cnn.summary()

    def test_create_runner(self):
        env = {
            "data_root": "../data",
            "configs_dir": "../configs"
        }
        runner = Runner(env, "default.json")

    def test_train_cnn(self):
        runner = Runner()
        runner.prepare()
        runner.train_cnn()
