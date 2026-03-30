#!/usr/bin/env python3
import json
from pathlib import Path


def load_params(params_dir: str):
    am_file = Path(params_dir) / "array_metadatas" / "process_0"
    with open(am_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        (x["array_metadata"]["param_name"].removeprefix("params.").replace(".", "/"),
         x["array_metadata"]["write_shape"])
        for x in data["array_metadatas"]
    ]


def is_action(k: str) -> bool:
    """True if the key belongs to the action expert.

    All action-expert weights use the _1 naming convention:
      - attn: /q_einsum_1/, /kv_einsum_1/, /attn_vec_einsum_1/
      - ffn:  /mlp_1/  (covers gating_einsum and linear)
      - norm: /pre_attention_norm_1/, /pre_ffw_norm_1/

    Note: each expert has its OWN Q/K/V projections (different input dims:
    language=2048, action=1024), so nothing is truly "shared" at the weight level.
    The two experts cross-attend by concatenating their projected Q/K/V before softmax.
    """
    return any(x in k for x in [
        "/q_einsum_1/",
        "/kv_einsum_1/",
        "/attn_vec_einsum_1/",
        "/mlp_1/",
        "/pre_attention_norm_1/",
        "/pre_ffw_norm_1/",
    ])


def main():
    params_dir = "/storage/yukaichengLab/lishiwen/jiayusun/openpi_pt/pi05_model/pi05_base/params"
    params = load_params(params_dir)

    print(f"Total params: {len(params)}\n")

    llm = [(k, s) for k, s in params
           if k.startswith("PaliGemma/llm/layers/") or k.startswith("PaliGemma/llm/final_norm")]
    action   = sorted([(k, s) for k, s in llm if     is_action(k) or "/final_norm_1" in k], key=lambda x: x[0])
    language = sorted([(k, s) for k, s in llm if not is_action(k) and "/final_norm_1" not in k], key=lambda x: x[0])

    print("- action expert (attn + ffn + norm + final_norm_1):")
    for k, s in action:
        print(f"  {k}  {s}")
    print("\n- language (PaliGemma) (attn + ffn + norm + final_norm):")
    for k, s in language:
        print(f"  {k}  {s}")

    non_llm_layer = [(k, s) for k, s in params
                     if not k.startswith("PaliGemma/llm/layers/")
                     and not k.startswith("PaliGemma/llm/final_norm")]
    print("\n- non-llm-layer params (vision encoder, embedder, action i/o proj, time mlp):")
    for k, s in sorted(non_llm_layer):
        print(f"  {k}  {s}")


if __name__ == "__main__":
    main()
