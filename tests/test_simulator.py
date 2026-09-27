import unittest
from cuas_sim import Config, State2D, Target, Interceptor, Simulator

class TestSimulator(unittest.TestCase):
    def test_run_two_steps(self):
        cfg = Config(time_step=0.5, random_seed=1)
        target_state = State2D(x=0.0, y=0.0, vx=2.0, vy=0.0)
        interceptor_state = State2D(x=10.0, y=0.0, vx=-1.0, vy=0.0)
        target = Target(state=target_state, config=cfg)
        interceptor = Interceptor(state=interceptor_state, config=cfg)
        sim = Simulator(config=cfg, target=target, interceptor=interceptor)
        sim.run(steps=2)
        # After 2 steps of 0.5s each, total time = 1.0s
        self.assertAlmostEqual(sim.time, 1.0)
        # Target moved 2.0 * 1.0 = 2.0 in x
        self.assertAlmostEqual(sim.target.state.x, 2.0)
        # Interceptor moved -1.0 * 1.0 = -1.0 in x from 10.0
        self.assertAlmostEqual(sim.interceptor.state.x, 9.0)
        # History should have two entries
        self.assertEqual(len(sim.history), 2)
        # First snapshot time should be 0.5
        self.assertAlmostEqual(sim.history[0][0], 0.5)
        # Second snapshot time should be 1.0
        self.assertAlmostEqual(sim.history[1][0], 1.0)

if __name__ == "__main__":
    unittest.main()
