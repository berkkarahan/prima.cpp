// Deterministic contracts for the decomposed Qwen3.5 graph primitives.
// Run on CPU by default, or pass an exact backend device name (e.g. Vulkan0).
// No scheduler is used: an unsupported op fails rather than falling back to CPU.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "../src/unicode.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

struct graph_case {
    ggml_context * ctx;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_backend_t backend;

    explicit graph_case(ggml_backend_t backend) : backend(backend) {
        ctx = ggml_init({4*1024*1024, nullptr, true});
        require(ctx != nullptr, "graph metadata allocation failed");
    }
    ~graph_case() {
        if (buffer) ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    void compute(ggml_tensor * output, const std::function<void()> & initialize) {
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto * node = ggml_graph_node(graph, i);
            if (!ggml_backend_supports_op(backend, node)) {
                throw std::runtime_error(std::string("backend does not support ") + ggml_op_desc(node));
            }
        }
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        require(buffer != nullptr, "tensor allocation failed");
        initialize();
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph computation failed");
        ggml_backend_synchronize(backend);
    }
};

static void set_f32(ggml_tensor * tensor, const std::vector<float> & values) {
    require(ggml_nbytes(tensor) == values.size()*sizeof(float), "input size mismatch");
    ggml_backend_tensor_set(tensor, values.data(), 0, values.size()*sizeof(float));
}

static std::vector<float> get_f32(ggml_tensor * tensor) {
    std::vector<float> values(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size()*sizeof(float));
    return values;
}

static void close(float actual, double expected, double tolerance = 1e-5) {
    if (std::isnan(expected)) {
        require(std::isnan(actual), "expected NaN");
    } else if (std::isinf(expected)) {
        require(actual == expected, "infinity mismatch");
    } else if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance*(1 + std::fabs(expected))) {
        throw std::runtime_error("primitive output differs from its mathematical contract");
    }
}

static void cumsum_strided(ggml_backend_t backend) {
    graph_case test(backend);
    auto * source = ggml_new_tensor_4d(test.ctx, GGML_TYPE_F32, 13, 3, 2, 2);
    auto * view = ggml_view_4d(test.ctx, source, 7, 2, 2, 2,
        source->nb[1], source->nb[2], source->nb[3], sizeof(float));
    auto * output = ggml_cumsum(test.ctx, view);
    std::vector<float> input(ggml_nelements(source), 999.0f);
    const float row[] = {1, -2, 3, 0, 4, -5, 2};
    for (int batch = 0; batch < 4; ++batch) {
        for (int r = 0; r < 2; ++r) {
            for (int i = 0; i < 7; ++i) input[batch*39 + r*13 + 1 + i] = row[i];
        }
    }
    test.compute(output, [&] { set_f32(source, input); });
    const float expected[] = {1, -1, 2, 2, 6, 1, 3};
    const auto result = get_f32(output);
    for (size_t i = 0; i < result.size(); ++i) close(result[i], expected[i%7], 0);
}

static void softplus_extremes(ggml_backend_t backend, bool inplace) {
    graph_case test(backend);
    auto * input = ggml_new_tensor_4d(test.ctx, GGML_TYPE_F32, 7, 2, 2, 2);
    auto * output = inplace ? ggml_softplus_inplace(test.ctx, input) : ggml_softplus(test.ctx, input);
    std::vector<float> values(ggml_nelements(input));
    const float row[] = {-100, -20, -1, 0, 1, 20, 100};
    for (size_t i = 0; i < values.size(); ++i) values[i] = row[i%7];
    test.compute(output, [&] { set_f32(input, values); });
    const auto result = get_f32(output);
    for (size_t i = 0; i < result.size(); ++i) close(result[i], std::log1p(std::exp(double(values[i]))));
    if (!inplace) require(get_f32(input) == values, "out-of-place softplus changed its input");
}

static void fill(ggml_backend_t backend, ggml_type type, bool inplace) {
    graph_case test(backend);
    auto * input = ggml_new_tensor_4d(test.ctx, type, 7, 3, 2, 2);
    auto * output = inplace ? ggml_fill_inplace(test.ctx, input, -1.25f) : ggml_fill(test.ctx, input, -1.25f);
    test.compute(output, [&] {
        const std::vector<uint8_t> initial(ggml_nbytes(input), 0xff);
        ggml_backend_tensor_set(input, initial.data(), 0, initial.size());
    });
    if (type == GGML_TYPE_F32) {
        for (float value : get_f32(output)) close(value, -1.25, 0);
    } else {
        std::vector<ggml_fp16_t> values(ggml_nelements(output));
        ggml_backend_tensor_get(output, values.data(), 0, ggml_nbytes(output));
        for (auto value : values) close(ggml_fp16_to_fp32(value), -1.25, 0);
    }
}

static void triangle(ggml_backend_t backend, ggml_tri_type type) {
    graph_case test(backend);
    auto * input = ggml_new_tensor_4d(test.ctx, GGML_TYPE_F32, 5, 5, 2, 2);
    auto * output = ggml_tri(test.ctx, input, type);
    std::vector<float> values(ggml_nelements(input));
    for (size_t i = 0; i < values.size(); ++i) values[i] = float(i + 1);
    test.compute(output, [&] { set_f32(input, values); });
    const auto result = get_f32(output);
    for (size_t i = 0; i < result.size(); ++i) {
        const size_t col = i%5, row = i/5%5;
        const bool lower = type == GGML_TRI_TYPE_LOWER || type == GGML_TRI_TYPE_LOWER_DIAG;
        const bool diagonal = type == GGML_TRI_TYPE_LOWER_DIAG || type == GGML_TRI_TYPE_UPPER_DIAG;
        const bool keep = (lower ? row > col : row < col) || (diagonal && row == col);
        close(result[i], keep ? values[i] : 0, 0);
    }
    require(get_f32(input) == values, "triangle changed its input");
}

static void triangular_solve(ggml_backend_t backend) {
    graph_case test(backend);
    auto * a = ggml_new_tensor_4d(test.ctx, GGML_TYPE_F32, 3, 3, 2, 2);
    auto * b = ggml_new_tensor_4d(test.ctx, GGML_TYPE_F32, 2, 3, 2, 2);
    auto * output = ggml_solve_tri(test.ctx, a, b, true, true, false);
    // Known solutions, two RHS columns and four different batches. Upper
    // triangle deliberately nonzero: it must not contribute to the solve.
    std::vector<float> av, bv, expected;
    for (int batch = 0; batch < 4; ++batch) {
        const float d = float(batch + 2);
        const float matrix[] = {d, 99, 88, -1, 2, 77, 3, 1, 4};
        av.insert(av.end(), matrix, matrix + 9);
        const float rhs[] = {d, -d, 3, 7, 17, 4};
        bv.insert(bv.end(), rhs, rhs + 6);
        const float solution[] = {1, -1, 2, 3, 3, 1};
        expected.insert(expected.end(), solution, solution + 6);
    }
    test.compute(output, [&] { set_f32(a, av); set_f32(b, bv); });
    const auto result = get_f32(output);
    for (size_t i = 0; i < result.size(); ++i) close(result[i], expected[i]);
    require(get_f32(a) == av && get_f32(b) == bv, "solve changed its operands");
}

static void repeat_4d(ggml_backend_t backend) {
    graph_case test(backend);
    auto * input = ggml_new_tensor_4d(test.ctx, GGML_TYPE_F32, 2, 1, 2, 1);
    auto * output = ggml_repeat_4d(test.ctx, input, 4, 3, 4, 2);
    test.compute(output, [&] { set_f32(input, {1, 2, 3, 4}); });
    const auto result = get_f32(output);
    for (size_t i = 0; i < result.size(); ++i) close(result[i], double(1 + i%2 + 2*(i/12%2)), 0);
}

static void composed_unaries(ggml_backend_t backend) {
    graph_case test(backend);
    auto * a = ggml_new_tensor_4d(test.ctx, GGML_TYPE_F32, 3, 2, 2, 2);
    auto * b = ggml_new_tensor_1d(test.ctx, GGML_TYPE_F32, 3);
    auto * out = ggml_sigmoid(test.ctx, ggml_exp(test.ctx, ggml_neg(test.ctx, ggml_sub(test.ctx, a, b))));
    std::vector<float> values(ggml_nelements(a));
    for (size_t i = 0; i < values.size(); ++i) values[i] = float(i)*0.1f;
    test.compute(out, [&] { set_f32(a, values); set_f32(b, {0.1f, 0.2f, 0.3f}); });
    const auto result = get_f32(out);
    for (size_t i = 0; i < result.size(); ++i) close(result[i], 1/(1 + std::exp(-std::exp(-values[i] + 0.1*(i%3 + 1)))));
}

static void convolution(ggml_backend_t backend) {
    graph_case test(backend);
    auto * a = ggml_new_tensor_3d(test.ctx, GGML_TYPE_F32, 8, 3, 2);
    auto * kernel = ggml_new_tensor_2d(test.ctx, GGML_TYPE_F32, 4, 3);
    auto * out = ggml_ssm_conv(test.ctx, a, kernel);
    std::vector<float> values(48), weights(12);
    for (size_t i = 0; i < values.size(); ++i) values[i] = float(i)*0.01f;
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = float(int(i%4) - 2)*0.125f;
    test.compute(out, [&] { set_f32(a, values); set_f32(kernel, weights); });
    const auto result = get_f32(out);
    for (int sequence = 0; sequence < 2; ++sequence) for (int token = 0; token < 5; ++token) for (int channel = 0; channel < 3; ++channel) {
        double expected = 0;
        for (int k = 0; k < 4; ++k) expected += values[sequence*24 + channel*8 + token + k]*weights[channel*4 + k];
        close(result[sequence*15 + token*3 + channel], expected);
    }
}

static void diagonal_set(ggml_backend_t backend, bool inplace) {
    graph_case test(backend);
    auto * a = ggml_new_tensor_4d(test.ctx, GGML_TYPE_F32, 3, 3, 2, 2);
    auto * v = ggml_new_tensor_1d(test.ctx, GGML_TYPE_F32, 3);
    auto * diag = ggml_diag(test.ctx, v);
    auto * out = inplace ? ggml_set_inplace(test.ctx, a, diag, a->nb[1], a->nb[2], a->nb[3], a->nb[2]) :
                          ggml_set(test.ctx, a, diag, a->nb[1], a->nb[2], a->nb[3], a->nb[2]);
    test.compute(out, [&] { set_f32(a, std::vector<float>(36, -1)); set_f32(v, {1, 2, 3}); });
    const auto result = get_f32(out);
    for (size_t i = 0; i < result.size(); ++i) {
        double expected = -1;
        if (i >= 9 && i < 18) expected = (i - 9)%4 == 0 ? double((i - 9)/4 + 1) : 0;
        close(result[i], expected, 0);
    }
}

static void model_iq_blocks(ggml_backend_t backend, const char * path) {
    ggml_context * metadata = nullptr;
    auto * file = gguf_init_from_file(path, {true, &metadata});
    require(file && metadata, "Cannot read GGUF tensor descriptors");
    std::set<ggml_type> needed = {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
        GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M, GGML_TYPE_IQ4_XS};
    std::ifstream input(path, std::ios::binary);
    for (int index = 0; index < gguf_get_n_tensors(file) && !needed.empty(); ++index) {
        const auto type = gguf_get_tensor_type(file, index);
        if (!needed.count(type)) continue;
        auto * tensor = ggml_get_tensor(metadata, gguf_get_tensor_name(file, index));
        if (tensor->ne[0] < 512 || tensor->ne[1] < 3) continue;
        const size_t bytes_per_row = ggml_row_size(type, 512);
        std::vector<uint8_t> packed(3*bytes_per_row);
        std::vector<float> decoded(3*512);
        for (int row = 0; row < 3; ++row) {
            input.seekg(gguf_get_data_offset(file) + gguf_get_tensor_offset(file, index) + row*ggml_row_size(type, tensor->ne[0]));
            input.read(reinterpret_cast<char *>(packed.data() + row*bytes_per_row), bytes_per_row);
            require(bool(input), "Cannot read original GGUF IQ blocks");
            ggml_internal_get_type_traits(type).to_float(packed.data() + row*bytes_per_row, decoded.data() + row*512, 512);
        }
        for (bool gather : {false, true}) {
            graph_case test(backend);
            auto * a = ggml_new_tensor_2d(test.ctx, type, 512, 3);
            auto * b = gather ? ggml_new_tensor_1d(test.ctx, GGML_TYPE_I32, 3) : ggml_new_tensor_3d(test.ctx, GGML_TYPE_F32, 512, 3, 2);
            auto * out = gather ? ggml_get_rows(test.ctx, a, b) : ggml_mul_mat(test.ctx, a, b);
            const int32_t ids[] = {2, 0, 1};
            std::vector<float> vectors(512*6);
            for (size_t i = 0; i < vectors.size(); ++i) vectors[i] = 0.01f*std::sin(float(i)*0.07f);
            test.compute(out, [&] {
                ggml_backend_tensor_set(a, packed.data(), 0, packed.size());
                if (gather) ggml_backend_tensor_set(b, ids, 0, sizeof(ids));
                else set_f32(b, vectors);
            });
            const auto result = get_f32(out);
            for (size_t i = 0; i < result.size(); ++i) {
                double expected = 0;
                if (gather) expected = decoded[ids[i/512]*512 + i%512];
                else for (size_t k = 0; k < 512; ++k) expected += double(decoded[i%3*512 + k])*vectors[i/3*512 + k];
                close(result[i], expected, 5e-4);
            }
        }
        std::printf("Original GGUF %s gather/matrix contracts passed\n", ggml_type_name(type));
        needed.erase(type);
    }
    ggml_free(metadata);
    gguf_free(file);
    require(needed.empty(), "GGUF lacks one of the required IQ fixtures");
}

static void tokenizer_splits() {
    const std::string regex = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    const auto split = [&](const std::string & text, const std::vector<std::string> & expected) {
        std::vector<std::string> encoded;
        for (const auto & piece : expected) {
            std::string value;
            for (unsigned char byte : piece) value += unicode_byte_to_utf8(byte);
            encoded.push_back(value);
        }
        const auto actual = unicode_regex_split(text, {regex});
        if (actual != encoded) {
            std::fprintf(stderr, "Qwen3.5 split input: %s\nactual:", text.c_str());
            for (const auto & part : actual) {
                std::string raw;
                for (auto cp : unicode_cpts_from_utf8(part)) raw += char(unicode_utf8_to_byte(unicode_cpt_to_utf8(cp)));
                std::fprintf(stderr, " [%s]", raw.c_str());
            }
            std::fprintf(stderr, "\nexpected:");
            for (const auto & part : expected) std::fprintf(stderr, " [%s]", part.c_str());
            std::fprintf(stderr, "\n");
            throw std::runtime_error("Qwen3.5 pre-tokenizer split mismatch");
        }
    };
    split(u8" cafe\u0301 123", {u8" cafe\u0301", " ", "1", "2", "3"});
    split("WE'RE\r\n  ok", {"WE", "'RE", "\r\n", " ", " ok"});
    // The optional-space punctuation alternative consumes " !" before the
    // following combining mark can begin the next letter/mark alternative.
    split(u8"\u0301a !\u0301", {u8"\u0301a", " !", u8"\u0301"});
    split("", {});
}

int main(int argc, char ** argv) {
    ggml_backend_t backend = nullptr;
    try {
        backend = argc > 1 ? ggml_backend_init_by_name(argv[1], nullptr) : ggml_backend_cpu_init();
        require(backend != nullptr, "requested backend is unavailable");
        for (int threads : {1, 4}) {
            if (ggml_backend_is_cpu(backend)) ggml_backend_cpu_set_n_threads(backend, threads);
            cumsum_strided(backend);
            softplus_extremes(backend, false);
            softplus_extremes(backend, true);
            for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
                fill(backend, type, false);
                fill(backend, type, true);
            }
            for (auto type : {GGML_TRI_TYPE_UPPER_DIAG, GGML_TRI_TYPE_UPPER, GGML_TRI_TYPE_LOWER_DIAG, GGML_TRI_TYPE_LOWER}) {
                triangle(backend, type);
            }
            triangular_solve(backend);
            repeat_4d(backend);
            composed_unaries(backend);
            convolution(backend);
            diagonal_set(backend, false);
            diagonal_set(backend, true);
        }
        tokenizer_splits();
        if (argc > 2) model_iq_blocks(backend, argv[2]);
        std::printf("Qwen3.5 primitive contracts passed on %s\n", ggml_backend_name(backend));
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        if (backend) ggml_backend_free(backend);
        return 1;
    }
}
