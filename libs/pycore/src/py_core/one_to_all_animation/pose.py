from __future__ import annotations

from pathlib import Path
from typing import Callable


_POSE2D_CACHE: dict[tuple[str, str], object] = {}


def resizecrop(image, th: int, tw: int):  # type: ignore[no-untyped-def]
    w, h = image.size
    if h / w > th / tw:
        new_w = int(w)
        new_h = int(new_w * th / tw)
    else:
        new_h = int(h)
        new_w = int(new_h * tw / th)
    left = (w - new_w) / 2
    top = (h - new_h) / 2
    right = (w + new_w) / 2
    bottom = (h + new_h) / 2
    image = image.crop((left, top, right, bottom))
    image = image.resize((tw, th))
    return image


def _get_pose2d(
    *,
    pose2d_checkpoint_path: Path,
    det_checkpoint_path: Path,
    device: str,
):
    key = (str(pose2d_checkpoint_path), str(det_checkpoint_path))
    pose2d = _POSE2D_CACHE.get(key)
    if pose2d is None:
        from wanpose_utils.pose2d import Pose2d

        pose2d = Pose2d(
            checkpoint=str(pose2d_checkpoint_path),
            detector_checkpoint=str(det_checkpoint_path),
            device="cpu",
        )
        _POSE2D_CACHE[key] = pose2d

    try:
        detector = getattr(pose2d, "detector", None)
        if detector is not None and hasattr(detector, "set_device"):
            detector.set_device(device)
        model = getattr(pose2d, "model", None)
        if model is not None and hasattr(model, "set_device"):
            model.set_device(device)
    except Exception:
        pass

    return pose2d


def warp_ref_to_pose(
    tgt_img,
    ref_pose: dict,
    tgt_pose: dict,
    *,
    bg_val=(0, 0, 0),
    conf_th: float = 0.9,
    align_center: bool = False,
):
    import cv2
    import numpy as np

    from opensora.dataset.utils import draw_pose_aligned
    from infer_function import scale_and_translate_pose

    h, w = tgt_img.shape[:2]
    img_tgt_pose = draw_pose_aligned(tgt_pose, h, w, without_face=True)

    tgt_kpt = tgt_pose["bodies"]["candidate"].astype(np.float32)
    ref_kpt = ref_pose["bodies"]["candidate"].astype(np.float32)

    scale_ratio = scale_and_translate_pose(tgt_pose, ref_pose, conf_th=conf_th, return_ratio=True)

    anchor_idx = 1
    x0 = tgt_kpt[anchor_idx][0] * w
    y0 = tgt_kpt[anchor_idx][1] * h

    ref_x = ref_kpt[anchor_idx][0] * w if not align_center else w / 2
    ref_y = ref_kpt[anchor_idx][1] * h

    dx = ref_x - x0
    dy = ref_y - y0

    m = np.array(
        [[scale_ratio, 0, (1 - scale_ratio) * x0 + dx], [0, scale_ratio, (1 - scale_ratio) * y0 + dy]],
        dtype=np.float32,
    )
    img_warp = cv2.warpAffine(
        tgt_img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=bg_val
    )
    img_tgt_pose_warp = cv2.warpAffine(
        img_tgt_pose, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=bg_val
    )
    zeros = np.zeros((h, w), dtype=np.uint8)
    mask_warp = cv2.warpAffine(
        zeros, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=255
    )
    return img_warp, img_tgt_pose_warp, mask_warp


def load_poses_whole_video(
    *,
    video_path: str,
    reference: str,
    pose2d_checkpoint_path: Path,
    det_checkpoint_path: Path,
    device: str,
    frame_interval: int = 2,
    transform: Callable | None = None,
    do_align: bool = False,
    alignmode: str = "ref",
    face_change: bool = False,
    head_change: bool = False,
    anchor_idx: int = 0,
    without_face: bool = False,
):
    import glob
    import os

    import numpy as np
    import torch
    from einops import rearrange
    from PIL import Image

    from opensora.dataset.utils import draw_pose_aligned
    from infer_function import aaposemeta_to_dwpose, align_to_pose, align_to_reference

    if transform is None:
        transform = lambda x: x  # noqa: E731

    pose2d = _get_pose2d(
        pose2d_checkpoint_path=pose2d_checkpoint_path,
        det_checkpoint_path=det_checkpoint_path,
        device=device,
    )

    ref_img = Image.open(reference).convert("RGB")
    h_ref, w_ref = ref_img.height, ref_img.width
    ref_rgb = np.array(ref_img)

    ref_pose_meta = pose2d([ref_rgb])[0]
    ref_dwpose = aaposemeta_to_dwpose(ref_pose_meta)

    if os.path.isdir(video_path):
        img_paths = sorted(glob.glob(os.path.join(video_path, "*.png")))
        if not img_paths:
            raise FileNotFoundError(f"NO PNG in {video_path}!!")
        frames_list = []
        for p in img_paths:
            img = Image.open(p).convert("RGB")
            img = resizecrop(img, th=h_ref, tw=w_ref)
            frames_list.append(np.array(img))
        frames = np.stack(frames_list)
        total_frames = len(frames_list)
        frame_indices = np.arange(0, total_frames, frame_interval).astype(int)
        frames = frames[frame_indices]
        h, w = frames[0].shape[:2]
    else:
        import decord

        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        frame_indices = np.arange(0, total_frames, frame_interval).astype(int)
        frames = vr.get_batch(frame_indices).asnumpy()
        frames_resized = []
        for fr in frames:
            pil_fr = Image.fromarray(fr)
            pil_fr = resizecrop(pil_fr, th=h_ref, tw=w_ref)
            frames_resized.append(np.array(pil_fr))
        frames = np.stack(frames_resized)
        h, w = frames[0].shape[:2]

    tpl_pose_metas = pose2d(frames)
    tpl_dwposes = [aaposemeta_to_dwpose(meta) for meta in tpl_pose_metas]

    if alignmode == "ref":
        image_input = transform(ref_img)
        pose_input = draw_pose_aligned(ref_dwpose, h, w, without_face=True)
        pose_input = transform(Image.fromarray(pose_input))
        mask_input = Image.new("RGB", image_input.size, (0, 0, 0))
        if do_align:
            tpl_dwposes = align_to_reference(ref_pose_meta, tpl_pose_metas, tpl_dwposes, anchor_idx)
    elif alignmode == "pose":
        image_input, pose_input, mask_input = warp_ref_to_pose(
            ref_rgb, tpl_dwposes[anchor_idx], ref_dwpose
        )
        image_input = transform(Image.fromarray(image_input))
        pose_input = transform(Image.fromarray(pose_input))
        mask_input = transform(Image.fromarray(mask_input).convert("RGB"))
        if do_align:
            tpl_dwposes = align_to_pose(ref_dwpose, tpl_dwposes, anchor_idx)
    else:
        raise ValueError(f"Unknown alignmode: {alignmode!r} (expected 'ref' or 'pose')")

    pose_imgs = []
    for pose_np in tpl_dwposes:
        pose_img = draw_pose_aligned(
            pose_np,
            h,
            w,
            without_face=without_face,
            face_change=face_change,
            head_change=head_change,
        )
        pose_img = transform(Image.fromarray(pose_img))
        pose_img = torch.from_numpy(np.array(pose_img))
        pose_img = rearrange(pose_img, "h w c -> c h w")
        pose_imgs.append(pose_img)

    pose_tensor = torch.stack(pose_imgs)
    return pose_tensor, image_input, pose_input, mask_input

