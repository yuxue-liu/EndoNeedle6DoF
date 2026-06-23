class PointManager:
    """
    Stores user prompt points keyed by (frame_idx, obj_id).

    labels:
        1 -> positive (foreground)
        0 -> negative (background)

    Used by the multi-class annotator: every class is a separate obj_id, and
    each frame keeps its own point buffer so corrections on one frame don't
    leak to another. The buffer is the *source of truth* we re-send to SAM with
    clear_old_points=True, which makes "clear this frame" trivial.
    """

    def __init__(self):
        # {(frame_idx, obj_id): {"points": [[x,y],...], "labels": [1/0,...]}}
        self._store = {}

    def _key(self, frame_idx, obj_id):
        return (int(frame_idx), int(obj_id))

    def add_point(self, frame_idx, obj_id, x, y, label):
        buf = self._store.setdefault(self._key(frame_idx, obj_id),
                                     {"points": [], "labels": []})
        buf["points"].append([int(x), int(y)])
        buf["labels"].append(int(label))

    def get(self, frame_idx, obj_id):
        buf = self._store.get(self._key(frame_idx, obj_id))
        if not buf:
            return [], []
        return buf["points"], buf["labels"]

    def has_points(self, frame_idx, obj_id):
        pts, _ = self.get(frame_idx, obj_id)
        return len(pts) > 0

    def clear_frame_obj(self, frame_idx, obj_id):
        self._store.pop(self._key(frame_idx, obj_id), None)

    def clear_frame(self, frame_idx):
        for key in [k for k in self._store if k[0] == int(frame_idx)]:
            self._store.pop(key, None)

    def objs_on_frame(self, frame_idx):
        return sorted({k[1] for k in self._store if k[0] == int(frame_idx)})
