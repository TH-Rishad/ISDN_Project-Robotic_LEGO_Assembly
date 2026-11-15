"""Step-boundary scheduling for two builders that occasionally need each other."""
from dataclasses import dataclass


@dataclass
class BuildState:
    supports: tuple
    step: int = 0
    busy: bool = False

    @property
    def done(self):
        return self.step == len(self.supports) and not self.busy


class Coordinator:
    def __init__(self, supports):
        self.builds = {a: BuildState(tuple(supports[a])) for a in 'AB'}
        self.pending = {}
        self.active = None
        self.next_request = None
        self.events = []

    def event(self, kind, arm, time, **data):
        row = dict(event=kind, arm=arm, step=self.builds[arm].step, time=float(time), **data)
        self.events.append(row)
        return row

    def complete(self, arm, time):
        state = self.builds[arm]
        if not state.busy:
            raise RuntimeError('cannot complete an idle builder')
        self.event('step_complete', arm, time)
        if self.active == arm:
            helper = 'B' if arm == 'A' else 'A'
            self.event('support_released', arm, time, helper=helper)
            self.active = None
            self.next_request = helper if helper in self.pending else None
        state.step += 1
        state.busy = False

    def _needed_as_helper(self, helper):
        """True if starting helper now would race past an upcoming brace need.

        studfit_012/013: A finished pyramid step 8 while B was still on S3 step 8,
        started drop-press A_b_005, then could not help B's braced step 9 → gate_miss.
        Hold only when the helper has already moved past the other's current step
        (same-layer parallel free steps still start together).
        """
        other = 'B' if helper == 'A' else 'A'
        o, h = self.builds[other], self.builds[helper]
        if o.done or not o.busy:
            return False
        nxt = o.step + 1
        return (nxt < len(o.supports) and bool(o.supports[nxt]) and h.step > o.step)

    def dispatch(self, ready, time):
        """Return (builder, helper-or-None) starts; collect BOTH requests first."""
        if self.active is not None:
            return []
        for arm, state in self.builds.items():
            if state.done or state.busy or not ready[arm]:
                continue
            if state.supports[state.step] and arm not in self.pending:
                self.pending[arm] = state.step
                self.event('support_requested', arm, time)
        if self.pending:
            if not all(ready.values()) or any(s.busy for s in self.builds.values()):
                return []
            arm = self.next_request if self.next_request in self.pending else min(self.pending)
            helper = 'B' if arm == 'A' else 'A'
            self.pending.pop(arm)
            self.next_request = None
            self.active = arm
            self.builds[arm].busy = True
            self.event('support_granted', arm, time, helper=helper)
            return [(arm, helper)]
        starts = []
        for arm, state in self.builds.items():
            if ready[arm] and not state.busy and not state.done:
                if self._needed_as_helper(arm):
                    continue
                state.busy = True
                self.event('step_started', arm, time)
                starts.append((arm, None))
        return starts

    @property
    def done(self):
        return all(s.done for s in self.builds.values())
