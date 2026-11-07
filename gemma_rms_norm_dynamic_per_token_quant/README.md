# Gemma RMSNorm + per-token INT8

对应 LightOp kernel：`gemma_rms_norm_dynamic_per_token_quant.cu`。

Python 接口在 `python_api.py`，保留 Gemma `gamma = 1 + weight`、可选 residual/update_input 和 per-token symmetric INT8 量化语义。

来源：LightOp Gemma PR !73，提交 `faae42f3c843357b06835106b87cdfbe64213e89`。