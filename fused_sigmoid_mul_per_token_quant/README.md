# Attention sigmoid × mul + per-token INT8 quantization

This Triton operator fuses `attention * sigmoid(gate)` and symmetric
per-token INT8 quantization for the activation before attention `o_proj`.
It rounds the gated activation to the input dtype before computing the row
amax. The BF16 path for hidden size 3072 processes three 1024-element segments
with one shared row scale. Other shapes and FP16 use the generic masked kernel.

```python
from fused_sigmoid_mul_per_token_quant import fuse_sigmoid_mul_int8_quant

quantized, scales = fuse_sigmoid_mul_int8_quant(attention, gate)
```

Pass contiguous BF16 or FP16 activation of shape `[tokens, hidden]`. The
matching gate may be `[tokens, hidden]` or a strided
`[tokens, heads, head_dim]` view with unit last-dimension stride. Outputs are
INT8 `[tokens, hidden]` and FP32 scales `[tokens, 1]`. Existing callers can
also use the `fused_sigmoid_mul_per_token_quant` alias.

The operator needs PyTorch and Triton. The test and benchmark additionally use
LightOp's `per_token_quant_int8` as the reference. Run from the repository root
on a supported GPU.

```sh
python -m pytest -q fused_sigmoid_mul_per_token_quant
python -m fused_sigmoid_mul_per_token_quant.benchmark --hidden 3072 --output /tmp/sigmoid_int8.json
```

The BF16 Triton implementation is shared with [LightOp PR #182](https://github.com/HYGON-AI/lightop/pull/182).
