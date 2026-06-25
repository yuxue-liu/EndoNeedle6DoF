"""Export the segmentation model to a FULL-PRECISION (FP32) accelerated engine.

FP32 sibling of export_seg_engine.py. Same drop-in contract:
    forward(x: (B,3,H,W) fp32) -> logits (B, nclass, mh, mw)
loaded at runtime by infer_accel.load_seg_engine and used by the v3 accelerated
drivers via `--seg-engine`.

WHY THIS EXISTS
---------------
export_seg_engine.py hardcodes FP16 (model.half() + enabled_precisions={half}).
On GPUs WITHOUT tensor cores (e.g. GTX 1650 Ti) FP16 has no fast path and is
~4x SLOWER than FP32, so the FP16 engine is the wrong tool there. The plain
"fp32" engines shipped so far are TorchScript-frozen FP32 — they remove Python
overhead but get NO TensorRT layer/kernel fusion.

This script adds the missing piece: a Torch-TensorRT engine built at
enabled_precisions={torch.float32}. TRT still does operator/kernel fusion,
memory-layout optimization and kernel auto-tuning at FULL precision, so it is
typically faster than eager/TorchScript FP32 *with bit-for-bit-comparable
numerics* (no accuracy loss). This is the "faster without lowering precision"
path for tensor-core-less GPUs.

The engine is FIXED-SHAPE and HARDWARE-SPECIFIC: it is built for one input size
(B, 3, infer_h, infer_w) derived from --seg-size + the source frame size + the
backbone patch (see infer_size below). Pass the SAME --seg-size and source
--src-h/--src-w you will run with, and BUILD ON THE TARGET GPU. Batch is 2
(both eyes in one forward).

Requires (TensorRT path): `pip install torch-tensorrt` matching your torch/CUDA.
If torch_tensorrt is missing or compile fails, falls back to a TorchScript FP32
engine (still removes Python/kernel-launch overhead, just no TRT fusion).

Example (1080x1920 source, seg-size 320, both eyes, local 1650 Ti):
    python export_seg_engine_fp32.py \
        --config configs/surgical_combined_base.yaml \
        --checkpoint exp/combined_r100_base/best.pth \
        --src-h 1080 --src-w 1920 --seg-size 320 \
        --out weights/seg_engine_trtfp32_320.ts
Then run inference with:
    python realtime_stereo_keypoints_v3_accel.py ... --seg-size 320 \
        --seg-engine weights/seg_engine_trtfp32_320.ts
"""
import argparse
import contextlib
import os
import sys

import torch
import yaml

# build_inference_model / util / model.backbone live in the private UniMatch-V2
# training repo, NOT in this published EndoNeedle6DoF tree. Resolve it and put it
# on sys.path so `from test import ...` finds the real module instead of the
# stdlib `test` package. Inserted at index 1 (after the script dir at index 0) so
# local siblings like infer_accel.py still win over any same-named file in the repo.
_DEFAULT_REPO = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'UniMatch-V2_local'))


def _bootstrap_repo(repo):
    if not os.path.isdir(repo):
        raise SystemExit(
            f"[export] UniMatch-V2 repo not found at: {repo}\n"
            f"         It provides test.py/util/model (build_inference_model).\n"
            f"         Pass --repo /path/to/UniMatch-V2_local or set it correctly.")
    if repo not in sys.path:
        sys.path.insert(1, repo)


def infer_size(src_h, src_w, seg_size, patch):
    if seg_size and max(src_h, src_w) > seg_size:
        s = seg_size / float(max(src_h, src_w))
        sh, sw = int(round(src_h * s)), int(round(src_w * s))
    else:
        sh, sw = src_h, src_w
    ih = max(patch, int(round(sh / patch)) * patch)
    iw = max(patch, int(round(sw / patch)) * patch)
    return ih, iw


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True, help='output engine path (.ts)')
    p.add_argument('--src-h', type=int, default=1080, help='source frame height (left eye)')
    p.add_argument('--src-w', type=int, default=1920, help='source frame width (left eye)')
    p.add_argument('--seg-size', type=int, default=320, help='MUST match the run-time --seg-size')
    p.add_argument('--batch', type=int, default=2, help='2 = both eyes in one forward')
    p.add_argument('--format', choices=['tensorrt', 'torchscript', 'auto'], default='auto',
                   help='auto = try TensorRT FP32, fall back to TorchScript FP32')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--repo', default=_DEFAULT_REPO,
                   help='path to the UniMatch-V2 training repo that provides '
                        'test.py/util/model (default: ../../UniMatch-V2_local)')
    args = p.parse_args()

    # put the training repo on sys.path BEFORE importing its modules
    _bootstrap_repo(args.repo)
    from test import build_inference_model

    # xformers' memory_efficient_attention runs through a custom autograd Function
    # (_Unbind) that TorchScript/TensorRT cannot serialize -> torch.jit.save fails
    # with "Could not export Python function call '_Unbind'". DINOv2's MemEffAttention
    # has a built-in fallback to the equivalent PyTorch SDPA path (Attention.forward,
    # same math) when XFORMERS_AVAILABLE is False. Force that path for export so the
    # graph is traceable/serializable; runtime accuracy is unchanged.
    try:
        from model.backbone.dinov2_layers import attention as _attn
        if getattr(_attn, 'XFORMERS_AVAILABLE', False):
            _attn.XFORMERS_AVAILABLE = False
            print('[export] xformers attention disabled for export -> PyTorch SDPA '
                  '(same math); the exported engine is serializable.')
    except Exception as e:  # noqa
        print(f'[export] note: could not toggle xformers attention flag ({e}); '
              'export may fail on _Unbind if xformers is active.')

    # TensorRT >=10 lowers F.scaled_dot_product_attention to its IAttention layer,
    # which REQUIRES CUDA capability >= 8.0 (Ampere). On sm75 (GTX 1650 Ti / Turing)
    # the build fails: "IAttention must be used on GPUs with CUDA capability 8.0 or
    # higher". Replace SDPA with the mathematically-identical manual attention
    # (q@k.T * scale -> softmax -> @v) so TRT sees plain matmul+softmax and builds
    # standard layers. Same numerics (math backend), patched only for this export.
    try:
        from model.backbone.dinov2_layers import attention as _attn

        def _manual_attention_forward(self, x):
            B, N, C = x.shape
            qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
                   .permute(2, 0, 3, 1, 4))
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            o = (attn @ v).transpose(1, 2).reshape(B, N, C)
            return self.proj_drop(self.proj(o))

        _attn.Attention.forward = _manual_attention_forward
        print('[export] SDPA -> manual matmul+softmax attention (sm75-compatible, '
              'same math); avoids TRT IAttention (Ampere-only).')
    except Exception as e:  # noqa
        print(f'[export] note: could not patch attention to manual path ({e}); '
              'TRT build may fail on sm75 with an IAttention capability error.')

    cfg = yaml.load(open(args.config, encoding='utf-8'), Loader=yaml.Loader)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    bundle = build_inference_model(cfg, args.checkpoint, device, visual_adapter=False)
    if bundle.get('affinity_side') is not None or bundle.get('use_edge_enhance'):
        raise SystemExit('[export] model has an affinity/edge head -> the batched forward '
                         'is bypassed at runtime, so an exported engine would never be used. '
                         'Export a plain segmentation model instead.')
    model = bundle['model'].eval().float()        # keep FP32 weights (no .half())
    patch = bundle['patch_size']
    ih, iw = infer_size(args.src_h, args.src_w, args.seg_size, patch)
    shape = (args.batch, 3, ih, iw)
    print(f'[export] FP32  patch={patch}  src={args.src_h}x{args.src_w}  '
          f'seg_size={args.seg_size}  -> engine input shape {shape}')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)

    def export_torchscript():
        m = model.float()
        ex = torch.randn(*shape, device=device, dtype=torch.float32)
        with torch.inference_mode():
            ts = torch.jit.trace(m, ex, check_trace=False)
            ts = torch.jit.freeze(ts)
        torch.jit.save(ts, args.out)
        print(f'[export] TorchScript FP32 engine -> {args.out}')

    def export_tensorrt():
        from infer_accel import _add_trt_dll_dir
        _add_trt_dll_dir()           # Windows: put tensorrt_libs on the DLL path
        import torch_tensorrt
        m = model.float()                         # full-precision weights
        ex_in = torch_tensorrt.Input(shape, dtype=torch.float32)

        @contextlib.contextmanager
        def _ignore_export_cache_attrs():
            orig_setattr = torch.nn.Module.__setattr__

            def patched_setattr(self, name, value):
                if name == 'last_edge_boundary_logits':
                    return None
                return orig_setattr(self, name, value)

            torch.nn.Module.__setattr__ = patched_setattr
            try:
                yield
            finally:
                torch.nn.Module.__setattr__ = orig_setattr

        with torch.inference_mode(), _ignore_export_cache_attrs():
            trt = torch_tensorrt.compile(
                m, inputs=[ex_in], enabled_precisions={torch.float32})
        # torch_tensorrt>=2.x dynamo `compile` returns an fx GraphModule, not a
        # ScriptModule, so torch.jit.save does not apply. Re-serialize to
        # TorchScript so the runtime loader (infer_accel.load_seg_engine ->
        # torch.jit.load) stays unchanged and the engine remains a drop-in .ts.
        ex = torch.randn(*shape, device=device, dtype=torch.float32)
        torch_tensorrt.save(trt, args.out, output_format='torchscript', inputs=[ex])
        print(f'[export] Torch-TensorRT FP32 engine -> {args.out}')

    if args.format == 'torchscript':
        export_torchscript()
    elif args.format == 'tensorrt':
        export_tensorrt()
    else:  # auto
        try:
            export_tensorrt()
        except Exception as e:  # noqa
            print(f'[export] TensorRT FP32 export failed ({e})\n'
                  '[export] falling back to TorchScript FP32')
            export_torchscript()

    # sanity: reload through the real runtime path. load_seg_engine(precision='fp32')
    # wraps with _FloatInputEngine (casts x.float()), which is exactly how the v3
    # drivers will feed this engine — so this verifies the actual inference contract.
    from infer_accel import load_seg_engine
    eng = load_seg_engine(args.out, device, precision='fp32')
    with torch.inference_mode():
        y = eng(torch.randn(*shape, device=device))     # float32 input, cast inside
    print(f'[export] OK — reload forward produced logits {tuple(y.shape)}')


if __name__ == '__main__':
    main()
