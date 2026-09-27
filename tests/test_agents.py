import unittest
from cuas_sim import Config, State2D, Target, Interceptor

class TestAgents(unittest.TestCase):
    def test_target_update(self):
        cfg = Config(time_step=0.2, random_seed=42)
        state = State2D(x=0.0, y=0.0, vx=1.5, vy=-0.5)
        target = Target(state=state, config=cfg)
        target.update()
        self.assertAlmostEqual(target.state.x, 0.3)  # 1.5 * 0.2
        self.assertAlmostEqual(target.state.y, -0.1)  # -0.5 * 0.2

    def test_interceptor_update(self):
        cfg = Config(time_step=0.1)
        state = State2D(x=5.0, y=5.0, vx=0.0, vy=2.0)
        interceptor = Interceptor(state=state, config=cfg)
        interceptor.update()
        self.assertAlmostEqual(interceptor.state.x, 5.0)
        self.assertAlmostEqual(interceptor.state.y, 5.2)

if __name__ == "__main__":
    unittest.main()
