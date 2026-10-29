"""Keep a closed brace stationary while the placer moves.

The shared cell rebases the TCP position after closing, but continues solving IK
against the planned orientation. That can move the arm under load. Freeze the
measured seven arm joints for hold; finger targets still come from the cell.
"""


class BraceHoldIK:
    def __init__(self, example):
        self.example = example
        self.delegate = example.ik
        self.joints = None

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def step(self, q_in, q_out, *args, **kwargs):
        ex = self.example
        holding = ex.A.phase == "hold"
        if holding and self.joints is None:
            self.joints = ex.state_0.joint_q.numpy()[:7].copy()
        elif not holding:
            self.joints = None
        result = self.delegate.step(q_in, q_out, *args, **kwargs)
        if holding:
            q = q_out.numpy()
            q[0, :7] = self.joints
            q_out.assign(q)
        return result


def install(example):
    """Use the normal IK call so the hold latch runs outside CUDA capture."""
    if isinstance(example.ik, BraceHoldIK):
        return
    example.ik = BraceHoldIK(example)
    example.graph_ik = None
