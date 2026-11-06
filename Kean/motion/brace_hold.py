"""Keep a closed brace stationary while the placer moves.

The shared cell rebases the TCP position after closing, but continues solving IK
against the planned orientation. That can move the arm under load. Freeze the
measured seven arm joints for hold; finger targets still come from the cell.
"""


class BraceHoldIK:
    def __init__(self, example):
        self.example = example
        self.delegate = example.ik
        self.joints = {}

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def step(self, q_in, q_out, *args, **kwargs):
        ex = self.example
        measured = ex.state_0.joint_q.numpy()
        for k, arm in enumerate((ex.A, getattr(ex, 'B', None))):
            if arm is not None and arm.phase == 'hold':
                self.joints.setdefault(k, measured[9*k:9*k+7].copy())
            else:
                self.joints.pop(k, None)
        result = self.delegate.step(q_in, q_out, *args, **kwargs)
        if self.joints:
            q = q_out.numpy()
            for k, joints in self.joints.items():
                q[k, :7] = joints
            q_out.assign(q)
        return result


def install(example):
    """Use the normal IK call so the hold latch runs outside CUDA capture."""
    if isinstance(example.ik, BraceHoldIK):
        return
    example.ik = BraceHoldIK(example)
    example.graph_ik = None
