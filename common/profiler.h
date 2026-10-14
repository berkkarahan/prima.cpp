#ifndef PROFILER_H
#define PROFILER_H

#include "ggml.h"
#include "llama.h"
#include "prima-profile-cost.h"

#define EPS                  1e-9
#define DISK_TEST_TOTAL_BYTE 500L * 1024 * 1024
#define DISK_TEST_SEQ_BLOCK  100L * 1024 * 1024
#define DISK_TEST_RND_BLOCK  4096
#define MEM_TEST_BLOCK_SIZE  64 * 1024


struct cpu_props {
    const char * name;
    const char * description;
    uint32_t     cores;
    double       flops_f32_f32;     // in GFLOPS
    double       flops_f16_f32;     // in GFLOPS
    double       flops_q2k_f32;     // in GFLOPS
    double       flops_q4k_f32;     // in GFLOPS
    double       flops_q5k_f32;     // in GFLOPS
    double       flops_q6k_f32;     // in GFLOPS
    double       flops_iq2xxs_f32;  // in GFLOPS
    double       flops_q50_f32;     // in GFLOPS    
    double       flops_q80_f32;     // in GFLOPS
    double       flops_iq1s_f32;    // in GFLOPS
    double       flops_iq4nl_f32;   // in GFLOPS
    double       flops_iq1m_f32;    // in GFLOPS

    cpu_props()
        : name            (""), 
          description     (""), 
          cores           (0), 
          flops_f32_f32   (0.0), 
          flops_f16_f32   (0.0), 
          flops_q2k_f32   (0.0),
          flops_q4k_f32   (0.0),
          flops_q5k_f32   (0.0),
          flops_q6k_f32   (0.0),
          flops_iq2xxs_f32(0.0),
          flops_q50_f32   (0.0),
          flops_q80_f32   (0.0),
          flops_iq1s_f32  (0.0), 
          flops_iq4nl_f32 (0.0),
          flops_iq1m_f32  (0.0)
    {}
};

struct memory_info {
    double       total_physical;     // in GiB
    double       available_physical; // in GiB
    double       used_can_swap;      // in GiB
    double       total_swap;         // in GiB
    double       available_swap;     // in GiB
    double       cpu_read_ram_bw;    // in GB/s
    double       mem_cpy_delay;      // in ms

    memory_info() : 
        total_physical    (0.0), 
        available_physical(0.0), 
        used_can_swap     (0.0),
        total_swap        (0.0), 
        available_swap    (0.0), 
        cpu_read_ram_bw   (0.0),
        mem_cpy_delay     (0.0) {}
};

struct gpu_support {
    bool         metal;
    bool         cuda;
    bool         vulkan;
    bool         kompute;
    bool         gpublas;
    bool         blas;
    bool         sycl;

    gpu_support() : 
        metal  (false), 
        cuda   (false), 
        vulkan (false), 
        kompute(false), 
        gpublas(false), 
        blas   (false), 
        sycl   (false) {}
};

struct gpu_props {
    const char * name;
    const char * description;
    double       memory_free;               // in GiB
    double       memory_total;              // in GiB
    double       metal_read_vram_bw;        // in GB/s
    double       metal_flops_f32_f32;       // in GFLOPS
    double       metal_flops_f16_f32;       // in GFLOPS
    double       metal_flops_q2k_f32;       // in GFLOPS
    double       metal_flops_q4k_f32;     // in GFLOPS
    double       metal_flops_q5k_f32;     // in GFLOPS
    double       metal_flops_q6k_f32;     // in GFLOPS
    double       metal_flops_iq2xxs_f32;  // in GFLOPS
    double       metal_flops_q50_f32;     // in GFLOPS
    double       metal_flops_q80_f32;     // in GFLOPS
    double       metal_flops_iq1s_f32;    // in GFLOPS
    double       metal_flops_iq4nl_f32;   // in GFLOPS
    double       metal_flops_iq1m_f32;    // in GFLOPS
    double       metal_mem_cpy_delay;     // in ms
    double       cuda_read_vram_bw;       // in GB/s
    double       cuda_flops_f32_f32;      // in GFLOPS
    double       cuda_flops_f16_f32;      // in GFLOPS
    double       cuda_flops_q2k_f32;      // in GFLOPS
    double       cuda_flops_q4k_f32;      // in GFLOPS
    double       cuda_flops_q5k_f32;      // in GFLOPS
    double       cuda_flops_q6k_f32;      // in GFLOPS
    double       cuda_flops_iq2xxs_f32;   // in GFLOPS
    double       cuda_flops_q50_f32;      // in GFLOPS
    double       cuda_flops_q80_f32;      // in GFLOPS
    double       cuda_flops_iq1s_f32;     // in GFLOPS
    double       cuda_flops_iq4nl_f32;    // in GFLOPS
    double       cuda_flops_iq1m_f32;     // in GFLOPS
    double       cuda_mem_cpy_delay;      // in ms

    gpu_props() : 
        name                    (""), 
        description             (""), 
        memory_free             (0.0), 
        memory_total            (0.0), 
        metal_read_vram_bw      (0.0),
        metal_flops_f32_f32     (0.0), 
        metal_flops_f16_f32     (0.0),
        metal_flops_q2k_f32     (0.0),
        metal_flops_q4k_f32     (0.0),
        metal_flops_q5k_f32     (0.0),
        metal_flops_q6k_f32     (0.0),
        metal_flops_iq2xxs_f32  (0.0),
        metal_flops_q50_f32     (0.0),
        metal_flops_q80_f32     (0.0),
        metal_flops_iq1s_f32    (0.0),
        metal_flops_iq4nl_f32   (0.0),
        metal_flops_iq1m_f32    (0.0),
        metal_mem_cpy_delay     (0.0),
        cuda_read_vram_bw       (0.0),
        cuda_flops_f32_f32      (0.0), 
        cuda_flops_f16_f32      (0.0), 
        cuda_flops_q2k_f32      (0.0),
        cuda_flops_q4k_f32      (0.0),
        cuda_flops_q5k_f32      (0.0),
        cuda_flops_q6k_f32      (0.0),
        cuda_flops_iq2xxs_f32   (0.0),
        cuda_flops_q50_f32      (0.0),
        cuda_flops_q80_f32      (0.0),
        cuda_flops_iq1s_f32     (0.0),
        cuda_flops_iq4nl_f32    (0.0),
        cuda_flops_iq1m_f32     (0.0),
        cuda_mem_cpy_delay      (0.0) {}
};

struct model_flops {
    double  inp_embd_ms;
    int64_t output_f32_f32;
    int64_t output_f16_f32;
    int64_t output_q2k_f32;
    int64_t output_q4k_f32;
    int64_t output_q5k_f32;
    int64_t output_q6k_f32;
    int64_t output_iq2xxs_f32;
    int64_t output_q50_f32;
    int64_t output_q80_f32;
    int64_t output_iq1s_f32;
    int64_t output_iq4nl_f32;
    int64_t output_iq1m_f32;
    int64_t layer_f32_f32;
    int64_t layer_f16_f32;
    int64_t layer_q2k_f32;
    int64_t layer_q4k_f32;
    int64_t layer_q5k_f32;
    int64_t layer_q6k_f32;
    int64_t layer_iq2xxs_f32;
    int64_t layer_q50_f32;
    int64_t layer_q80_f32;
    int64_t layer_iq1s_f32;
    int64_t layer_iq4nl_f32;
    int64_t layer_iq1m_f32;

    model_flops() : 
        inp_embd_ms        (0.0),
        output_f32_f32     (0), 
        output_f16_f32     (0),
        output_q2k_f32     (0),
        output_q4k_f32     (0),
        output_q5k_f32     (0),
        output_q6k_f32     (0), 
        output_iq2xxs_f32  (0),
        output_q50_f32     (0),
        output_q80_f32     (0),
        output_iq1s_f32    (0),
        output_iq4nl_f32   (0),
        output_iq1m_f32    (0),
        layer_f32_f32      (0),
        layer_f16_f32      (0),
        layer_q2k_f32      (0),
        layer_q4k_f32      (0),
        layer_q5k_f32      (0),
        layer_q6k_f32      (0),
        layer_iq2xxs_f32   (0),
        layer_q50_f32      (0),
        layer_q80_f32      (0),
        layer_iq1s_f32     (0),
        layer_iq4nl_f32    (0), 
        layer_iq1m_f32     (0)
        {}
};

struct model_params {
    int64_t input_f32;
    int64_t input_f16;
    int64_t input_q2k;
    int64_t input_q4k;
    int64_t input_q5k;
    int64_t input_q6k;
    int64_t input_iq2xxs;
    int64_t input_q50;
    int64_t input_q80;
    int64_t input_iq1s;
    int64_t input_iq4nl;
    int64_t input_iq1m;
    int64_t output_f32;
    int64_t output_f16;
    int64_t output_q2k;
    int64_t output_q4k;
    int64_t output_q5k;
    int64_t output_q6k;
    int64_t output_iq2xxs;
    int64_t output_q50;
    int64_t output_q80;
    int64_t output_iq1s;
    int64_t output_iq4nl;
    int64_t output_iq1m;
    int64_t layer_f32;
    int64_t layer_f16;
    int64_t layer_q2k;
    int64_t layer_q4k;
    int64_t layer_q5k;
    int64_t layer_q6k;
    int64_t layer_iq2xxs;
    int64_t layer_q50;
    int64_t layer_q80;
    int64_t layer_iq1s;
    int64_t layer_iq4nl;
    int64_t layer_iq1m;

    model_params() :
        input_f32       (0),
        input_f16       (0),
        input_q2k       (0),
        input_q4k       (0),
        input_q5k       (0),
        input_q6k       (0),
        input_iq2xxs    (0),
        input_q50       (0),
        input_q80       (0),
        input_iq1s      (0),
        input_iq4nl     (0),
        input_iq1m      (0),
        output_f32      (0),
        output_f16      (0),
        output_q2k      (0),
        output_q4k      (0),
        output_q5k      (0),
        output_q6k      (0),
        output_iq2xxs   (0),
        output_q50      (0),
        output_q80      (0),
        output_iq1s     (0),
        output_iq4nl    (0),
        output_iq1m     (0),
        layer_f32       (0),
        layer_f16       (0),
        layer_q2k       (0),
        layer_q4k       (0),
        layer_q5k       (0),
        layer_q6k       (0),
        layer_iq2xxs    (0),
        layer_q50       (0),
        layer_q80       (0),
        layer_iq1s      (0),
        layer_iq4nl     (0), 
        layer_iq1m      (0)
        {}
};

struct model_bytes {
    int64_t nb_input;
    int64_t nb_layer;
    int64_t nb_output;

    // used to estimate the compute buffer size 
    int64_t nb_output_w;
    int64_t nb_output_norm_w;
    int64_t nb_attn_norm_w;
    int64_t nb_attn_q_w;
    int64_t nb_ffn_gate_w;
    int64_t nb_ffn_down_w;

    model_bytes() :
        nb_input        (0),
        nb_layer        (0),
        nb_output       (0), 
        nb_output_w     (0),
        nb_output_norm_w(0),
        nb_attn_norm_w  (0),
        nb_attn_q_w     (0),
        nb_ffn_gate_w   (0),
        nb_ffn_down_w   (0) {}
};

struct disk_props {
    double read_seq_bw;  // in GB/s
    double read_rnd_bw;  // in GB/s
    double write_seq_bw; // in GB/s
    double write_rnd_bw; // in GB/s

    disk_props() :
        read_seq_bw (0.0),
        read_rnd_bw (0.0),
        write_seq_bw(0.0),
        write_rnd_bw(0.0) {}
};

struct startup_args{
    bool     should_profile;
    uint32_t n_ctx;
};

struct device_info {
    uint32_t            rank;
    const char *        device_name;
    const char *        device_os;
    const char *        next_ip;
    struct disk_props   disk;
    struct cpu_props    cpu_props;
    struct memory_info  memory;
    struct gpu_support  gpu_support;
    struct gpu_props    gpu_props;
    struct model_flops  model_flops;
    struct model_params model_params;
    struct model_bytes  model_bytes;

    device_info() : 
        rank(0), 
        device_name(""), 
        device_os(""),
        next_ip(""),
        disk(),
        cpu_props(), 
        memory(), 
        gpu_support(), 
        gpu_props(), 
        model_flops(),
        model_params(),
        model_bytes() {}
};

struct TopoRebuildHelperInfo{
    struct device_info dev_info;
    char               is_forwarder;
    
    TopoRebuildHelperInfo():
        dev_info(),
        is_forwarder(0){}
    
    void   deserialize(const char * buffer);
    size_t serialize(char ** buffer) const;
};

enum profiler_backend_type {
    PROFILER_BACKEND_TYPE_CPU   = 0,
    PROFILER_BACKEND_TYPE_METAL = 1,
    PROFILER_BACKEND_TYPE_CUDA  = 2,
    PROFILER_BACKEND_TYPE_VULKAN = 3,
};

enum profiler_layer_type {
    PROFILER_LAYER_INPUT   = 0,
    PROFILER_LAYER_OUTPUT  = 1,
    PROFILER_LAYER_BACKEND = 2,
};

struct prima_host_transfer {
    uint64_t bytes;
    std::vector<double> upload_ms, download_ms;
};

prima_host_transfer device_host_transfer(int embedding, profiler_backend_type backend);

const char * device_name(void); 
const char * device_os(void);
prima_model_costs device_model_costs(const struct llama_model * model, const struct llama_model_loader * metadata, int64_t history, int64_t context, bool flash_attention);
void device_free_model_metadata(struct llama_model_loader * metadata);

uint32_t device_cpu_cores         (void);
double   device_cpu_flops         (struct llama_model * model, enum ggml_type src0t, enum ggml_type src1t, int n_threads);
double   device_matrix_flops      (int n_embd, enum ggml_type src0t, enum ggml_type src1t, enum profiler_backend_type backend, int n_threads);
double   device_metal_flops       (struct llama_model * model, enum ggml_type src0t, enum ggml_type src1t);
double   device_cuda_flops        (struct llama_model * model, enum ggml_type src0t, enum ggml_type src1t);
double   device_vulkan_flops      (struct llama_model * model, enum ggml_type src0t, enum ggml_type src1t);
double   device_inp_embd_delay    (struct llama_model * model, enum ggml_type src0t, int n_tokens, int n_threads);
uint64_t device_physical_memory   (bool available);
uint64_t device_swap_memory       (bool available);
uint64_t device_swappable_memory  ();
void     device_disk_seq_bw       (double * read_seq_bw, double * write_seq_bw, int n_threads);
void     device_disk_rnd_bw       (double * read_rnd_bw, double * write_rnd_bw, int n_threads);
double   device_memory_bw         (int n_thread);
double   device_cpu_mem_copy      (struct llama_model * model, int n_threads);
double   device_metal_mem_copy    (struct llama_model * model);
double   device_cuda_mem_copy     (struct llama_model * model);
double   device_vulkan_mem_copy   (struct llama_model * model);
double   device_metal_read_vram_bw();
double   device_cuda_read_vram_bw ();
double   device_vulkan_read_vram_bw();
void     device_get_props         (struct llama_model * model, int device, struct ggml_backend_dev_props * props); 
void     device_print_props       (struct device_info * dev_info_set, int n, struct llama_model * model, const struct llama_context_params cparams);

int      device_has_metal  (void);
int      device_has_cuda   (void);
int      device_has_vulkan (void);
int      device_has_kompute(void);
int      device_has_gpublas(void);
int      device_has_blas   (void);
int      device_has_sycl   (void);

size_t   serialize  (const struct device_info * dev_info, char ** buffer);
size_t   deserialize(const char * buffer, struct device_info * dev_info);

#endif // PROFILER_H
