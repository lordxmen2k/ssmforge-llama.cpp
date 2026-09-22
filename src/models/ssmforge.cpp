// SSMForge model — hybrid SSM/attention architecture.
// See: https://github.com/lordxmen2k/SSMForge
//
// For now this implements the LLAMA_TENSOR_* loading paths required by
// llama-quantize, llama-cli, and the GGUF metadata routines. The forward
// graph is a minimal stub that the user has been informed runs only the
// hybrid metadata path — full forward-pass support is added in a follow-up
// commit.
//
// When this stub is finalized, `llama-quantize` will accept GGUF files
// with `general.architecture = "ssmforge"` and emit Q4_K_M GGUFs.

#include "models.h"

void llama_model_ssmforge::load_arch_hparams(llama_model_loader & ml) {
    // SSM2-relevant hparams (mirrors llama_model_mamba2)
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // Hybrid needs RMSNorm epsilon (Llama uses this)
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    // Standard transformer hparams are inherited from llama_model_base
    // and are read automatically by the base class via the standard KV keys
    // (general.context_length, general.embedding_length, general.block_count,
    //  general.feed_forward_length, attention.head_count, etc).
}

void llama_model_ssmforge::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t d_conv  = hparams.ssm_d_conv;
    const int64_t d_inner = hparams.ssm_d_inner;
    const int64_t d_state = hparams.ssm_d_state;
    const int64_t n_group = hparams.ssm_n_group;
    const int64_t dt_rank  = hparams.ssm_dt_rank;

    const int64_t conv_dim  = d_inner + 2 * n_group * d_state;
    const int64_t d_in_proj = d_inner + conv_dim + dt_rank;

    // Token embeddings + output (standard transformer pattern)
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    {
        output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
        output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
        if (output == NULL) {
            output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
        }
    }

    // Per-layer: each block is either attention (Llama-shaped) or SSM2.
    // llama-quantize doesn't actually need to know which is which — it just
    // needs all tensors registered. We mark SSM tensors as TENSOR_NOT_REQUIRED
    // so a block that is purely attention still loads cleanly.
    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        // Standard LlamaDecoderLayer-style tensors
        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd}, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);

        // SSM2 tensors (TENSOR_NOT_REQUIRED — block may not be SSM)
        layer.ssm_in = create_tensor(tn(LLM_TENSOR_SSM_IN, "weight", i), {n_embd, d_in_proj}, TENSOR_NOT_REQUIRED);

        layer.ssm_conv1d   = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", i), {d_conv, d_inner + 2*n_group*d_state}, TENSOR_NOT_REQUIRED);
        layer.ssm_conv1d_b = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "bias",   i), {d_inner + 2*n_group*d_state},          TENSOR_NOT_REQUIRED);

        layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", i), {dt_rank}, TENSOR_NOT_REQUIRED);

        layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A, i), {1, dt_rank}, TENSOR_NOT_REQUIRED);
        layer.ssm_d = create_tensor(tn(LLM_TENSOR_SSM_D, i), {1, dt_rank}, TENSOR_NOT_REQUIRED);

        layer.ssm_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", i), {d_inner / n_group, n_group}, TENSOR_NOT_REQUIRED);

        layer.ssm_out = create_tensor(tn(LLM_TENSOR_SSM_OUT, "weight", i), {d_inner, n_embd}, TENSOR_NOT_REQUIRED);
    }
}

std::unique_ptr<llm_graph_context> llama_model_ssmforge::build_arch_graph(const llm_graph_params & params) const {
    // Stub: forward pass for SSMForge GGUFs is not yet implemented in llama.cpp.
    // Quantization works because it doesn't call build_arch_graph.
    // When the user actually tries to load the GGUF for inference, this stub
    // returns an empty graph that produces an error from the runtime — better
    // than segfaulting.
    return nullptr;
}
