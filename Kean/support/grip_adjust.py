"""Arm A holds the brick, Arm B regrasps, the scan runs, then the new grasp is frozen."""


class GripAdjust:
    def __init__(self, brick_id, grip_pose):
        self.brick_id = brick_id
        self.grip_pose = grip_pose
        self.phase = "idle"

    def start(self, handoff=None, target=None, other_pos=None):
        if self.phase not in ("idle", "done"):
            raise RuntimeError(self.phase)
        if handoff is not None:
            handoff.begin("A", target, other_pos)
        self.phase = "a_approach"
        return {"arm": "A", "do": "hold_brick", "brick_id": self.brick_id}

    def arrived(self, arm, store, schedule, hand_xyz=None, hand_q=None, handoff=None, target=None, other_pos=None):
        if self.phase == "a_approach" and arm == "A":
            if handoff is not None:
                handoff.settled("A")
            self.phase = "b_open"
            return {"arm": "B", "do": "open", "brick_id": self.brick_id}
        if self.phase == "b_open" and arm == "B":
            store.release("B")
            schedule.begin_adjust(self.brick_id)
            self.phase = "scan"
            return {"arm": None, "do": "scan", "brick_id": self.brick_id}
        if self.phase == "b_regrasp" and arm == "B":
            xyz, quat = store.pose(self.brick_id)
            store.freeze("B", self.brick_id, hand_xyz, hand_q, xyz, quat)
            schedule.end_adjust()
            if handoff is not None:
                handoff.settled("B")
                handoff.begin("A", target, other_pos)
            self.phase = "a_return"
            return {"arm": "A", "do": "return_hold"}
        if self.phase == "a_return" and arm == "A":
            if handoff is not None:
                handoff.settled("A")
            self.phase = "done"
            return {"arm": None, "do": "done"}
        raise RuntimeError("%s arrived in %s" % (arm, self.phase))

    def scanned(self, store, schedule, detections, handoff=None, target=None, other_pos=None):
        if self.phase != "scan":
            raise RuntimeError(self.phase)
        schedule.apply(store, detections)
        if handoff is not None:
            handoff.begin("B", target, other_pos)
        self.phase = "b_regrasp"
        return {"arm": "B", "do": "regrasp", "brick_id": self.brick_id, "pose": self.grip_pose}
