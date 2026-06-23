import os
import cv2
import numpy as np

from engine import (
    logits_to_bool, image_to_mask_path,
    save_index_mask, load_index_mask,
)


class AnnotationSession:
    """
    Controller around a SAM-2-Plus video predictor for multi-class, multi-frame
    interactive annotation of ONE part-folder (<=200 frames).

    Each class is a SAM object (obj_id == class id, 1..N). Background = 0.

    Classes are kept fully INDEPENDENT: the source of truth is a per-class
    binary mask store `class_masks[frame][cls]`. Editing one class only ever
    writes that class's entry, so corrections on class 3 can never alter class 2
    (and vice versa), even where they occlude each other. The single-channel
    index PNG is only *composed* at save time, where overlaps are resolved by a
    fixed occlusion order (higher class id wins -> drawn on top).
    """

    def __init__(self, predictor, state, frames, num_classes, point_manager):
        self.predictor = predictor
        self.state = state
        self.frames = frames                      # list of image paths (sorted)
        self.num_classes = num_classes
        self.pm = point_manager
        self.H = state["video_height"]
        self.W = state["video_width"]
        self.num_frames = len(frames)
        # frame_idx -> {cls: bool ndarray(H,W)} ; independent per class
        self.class_masks = {}
        self._preload_existing_masks()

    # ---------------------------------------------------------------- helpers
    def _preload_existing_masks(self):
        """Split any previously saved index PNG back into per-class binaries."""
        for i, fp in enumerate(self.frames):
            m = load_index_mask(image_to_mask_path(fp))
            if m is None:
                continue
            if m.shape != (self.H, self.W):
                m = cv2.resize(m, (self.W, self.H),
                               interpolation=cv2.INTER_NEAREST)
            for cls in range(1, self.num_classes + 1):
                binm = (m == cls)
                if binm.any():
                    self.class_masks.setdefault(i, {})[cls] = binm

    def mask_path(self, frame_idx):
        return image_to_mask_path(self.frames[frame_idx])

    def _compose(self, frame_idx):
        """Build the single-channel index map from the per-class binaries.
        Ascending class order -> higher class id overwrites (drawn on top)."""
        idx = np.zeros((self.H, self.W), dtype=np.uint8)
        cm = self.class_masks.get(frame_idx)
        if cm:
            for cls in sorted(cm):
                m = cm[cls]
                if m is not None and m.any():
                    idx[m] = cls
        return idx

    def get_mask(self, frame_idx):
        cm = self.class_masks.get(frame_idx)
        if not cm:
            return None
        return self._compose(frame_idx)

    def _save(self, frame_idx):
        """Compose + persist this frame's index PNG (or delete if now empty)."""
        idx = self._compose(frame_idx)
        if idx.any():
            save_index_mask(self.mask_path(frame_idx), idx)
        else:
            mp = self.mask_path(frame_idx)
            if os.path.exists(mp):
                os.remove(mp)
        return idx if idx.any() else None

    def _set_obj_from_consolidated(self, frame_idx, cls, per_obj_bool, obj_ids):
        """Update ONLY `cls` from SAM's consolidated per-object output."""
        ids = list(obj_ids)
        if cls in ids:
            m = per_obj_bool[ids.index(cls)]
            if m.shape != (self.H, self.W):
                m = cv2.resize(m.astype(np.uint8), (self.W, self.H),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            if m.any():
                self.class_masks.setdefault(frame_idx, {})[cls] = m
            else:
                cm = self.class_masks.get(frame_idx)
                if cm:
                    cm.pop(cls, None)
        return self._save(frame_idx)

    # ----------------------------------------------------------- interaction
    def apply_points(self, frame_idx, cls):
        """
        Re-send the FULL point buffer for (frame_idx, cls) to SAM with
        clear_old_points=True, then recompose+save this frame's mask.
        Returns the updated index map.
        """
        points, labels = self.pm.get(frame_idx, cls)
        if len(points) == 0:
            # all points removed -> drop this object's prompt on the frame
            return self.clear_frame_obj(frame_idx, cls)

        _, obj_ids, video_res_masks, _ = self.predictor.add_new_points_or_box(
            inference_state=self.state,
            frame_idx=frame_idx,
            obj_id=cls,
            points=np.array(points, dtype=np.float32),
            labels=np.array(labels, dtype=np.int32),
            clear_old_points=True,
        )
        # update ONLY this class -> other classes are untouched
        return self._set_obj_from_consolidated(
            frame_idx, cls, logits_to_bool(video_res_masks), obj_ids)

    def clear_frame_obj(self, frame_idx, cls):
        self.pm.clear_frame_obj(frame_idx, cls)
        self.predictor.clear_all_prompts_in_frame(
            self.state, frame_idx, obj_id=cls, need_output=False)
        cm = self.class_masks.get(frame_idx)
        if cm:
            cm.pop(cls, None)            # drop only this class
        return self._save(frame_idx)

    def clear_frame(self, frame_idx):
        for cls in self.pm.objs_on_frame(frame_idx):
            self.predictor.clear_all_prompts_in_frame(
                self.state, frame_idx, obj_id=cls, need_output=False)
        self.pm.clear_frame(frame_idx)
        self.class_masks.pop(frame_idx, None)
        return self._save(frame_idx)

    # ----------------------------------------------------- direct pixel paint
    def paint(self, frame_idx, cls, cx, cy, radius, erase=False):
        """
        Directly paint (or erase) pixels for a class, bypassing SAM. Needed for
        thin structures (needle / suture) the model won't select from clicks.

        Painting is EXCLUSIVE: the stroke pixels are claimed by `cls` and removed
        from every other class on this frame. A single-channel index mask allows
        one label per pixel, so this is what makes a line painted under an
        over-extended grasper actually become visible (the grasper loses those
        pixels). Erase removes the stroke from `cls` only.

        Does NOT save or touch SAM (call commit_paint at the end of a stroke).
        Returns the composed index map for live preview.
        """
        cm = self.class_masks.setdefault(frame_idx, {})
        disk = np.zeros((self.H, self.W), dtype=np.uint8)
        cv2.circle(disk, (int(cx), int(cy)), max(1, int(radius)), 1, thickness=-1)
        disk = disk.astype(bool)

        if erase:
            m = cm.get(cls)
            if m is not None:
                m = m & ~disk
                if m.any():
                    cm[cls] = m
                else:
                    cm.pop(cls, None)
        else:
            m = cm.get(cls)
            cm[cls] = disk if m is None else (m | disk)
            for other in list(cm.keys()):       # exclusive: take from others
                if other == cls:
                    continue
                om = cm[other] & ~disk
                if om.any():
                    cm[other] = om
                else:
                    cm.pop(other, None)
        return self._compose(frame_idx)

    def commit_paint(self, frame_idx, cls):
        """End a paint stroke: persist the PNG and re-anchor every affected
        class into SAM so its memory matches the painted GT on this frame
        (the active class gained pixels; occluded classes may have lost some)."""
        cm = self.class_masks.get(frame_idx, {})
        existing = set(self.state["obj_id_to_idx"].keys())
        classes = set(cm.keys()) | (existing & set(range(1, self.num_classes + 1)))
        for c in sorted(classes):
            m = cm.get(c)
            if m is not None and m.any():
                self.predictor.add_new_mask(
                    self.state, frame_idx=frame_idx, obj_id=c, mask=m)
            elif c in existing:
                self.predictor.clear_all_prompts_in_frame(
                    self.state, frame_idx, obj_id=c, need_output=False)
        return self._save(frame_idx)

    # ----------------------------------------------------------- propagation
    def make_propagator(self, start_frame_idx):
        """
        New forward-propagation generator from start_frame_idx. Recreated on
        every Resume so the latest corrections become the conditioning.
        Yields (frame_idx, index_map) and auto-saves each frame.
        """
        gen = self.predictor.propagate_in_video(
            self.state, start_frame_idx=start_frame_idx, reverse=False)
        for frame_idx, obj_ids, video_res_masks, _, _ in gen:
            per_obj = logits_to_bool(video_res_masks)
            ids = list(obj_ids)
            # each class is tracked independently -> store each on its own
            for pos, cls in enumerate(ids):
                m = per_obj[pos]
                if m.shape != (self.H, self.W):
                    m = cv2.resize(m.astype(np.uint8), (self.W, self.H),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
                if m.any():
                    self.class_masks.setdefault(frame_idx, {})[cls] = m
                else:
                    cm = self.class_masks.get(frame_idx)
                    if cm:
                        cm.pop(cls, None)
            idx = self._save(frame_idx)
            yield frame_idx, idx

    # ------------------------------------------------- seed from saved masks
    def has_inputs(self):
        """True if SAM already holds any point/mask conditioning input."""
        st = self.state
        if any(st["point_inputs_per_obj"].get(i) for i in st["point_inputs_per_obj"]):
            return True
        if any(st["mask_inputs_per_obj"].get(i) for i in st["mask_inputs_per_obj"]):
            return True
        return False

    def seed_from_mask(self, frame_idx):
        """
        Register a frame's already-loaded/saved index mask back into SAM as
        per-class mask prompts, so propagation (or further edits) can continue
        from previously-saved annotations rather than starting from scratch.
        Returns the number of classes seeded.
        """
        # source: the per-class store (preloaded), else the saved PNG on disk
        cm = self.class_masks.get(frame_idx)
        if not cm:
            idx = load_index_mask(self.mask_path(frame_idx))
            if idx is None:
                return 0
            if idx.shape != (self.H, self.W):
                idx = cv2.resize(idx, (self.W, self.H),
                                 interpolation=cv2.INTER_NEAREST)
            cm = {}
            for cls in range(1, self.num_classes + 1):
                binm = (idx == cls)
                if binm.any():
                    cm[cls] = binm
            if cm:
                self.class_masks[frame_idx] = cm
        seeded = 0
        for cls, binm in sorted(cm.items()):
            self.predictor.add_new_mask(
                self.state, frame_idx=frame_idx, obj_id=cls, mask=binm)
            seeded += 1
        return seeded

    def prune_dangling_objects(self):
        """Remove SAM objects that hold NO point/mask input on any frame.

        Such a 'dangling' object (e.g. a class registered in obj_id_to_idx whose
        only prompt was later cleared on an edit) makes propagate_in_video's
        preflight raise 'No input points or masks ... for object id N'. We drop
        them so propagation can start from whatever IS seeded.
        """
        st = self.state
        dangling = []
        for obj_id, obj_idx in st["obj_id_to_idx"].items():
            pts = st["point_inputs_per_obj"].get(obj_idx) or {}
            msk = st["mask_inputs_per_obj"].get(obj_idx) or {}
            if not pts and not msk:
                dangling.append(obj_id)
        for obj_id in dangling:
            try:
                self.predictor.remove_object(st, obj_id, need_output=False)
            except Exception:
                pass
        return dangling
