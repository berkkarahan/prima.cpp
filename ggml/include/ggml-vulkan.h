#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_VK_NAME "Vulkan"
#define GGML_VK_MAX_DEVICES 16

GGML_API void ggml_vk_instance_init(void);

// backend API
GGML_API ggml_backend_t ggml_backend_vk_init(size_t dev_num);
GGML_API ggml_backend_reg_t ggml_backend_vk_reg(void);

GGML_API bool ggml_backend_is_vk(ggml_backend_t backend);
GGML_API bool ggml_backend_vk_buffer_is_device(ggml_backend_buffer_t buffer);
// Instrumented builds are diagnostic and cannot supply paired acceptance timings.
GGML_API bool ggml_backend_vk_diagnostic_timing(void);
GGML_API int  ggml_backend_vk_get_device_count(void);
GGML_API void ggml_backend_vk_get_device_description(int device, char * description, size_t description_size);
// Physical identity remains stable when visible devices are filtered/reordered.
GGML_API void ggml_backend_vk_get_device_identity(int device, char * uuid, size_t uuid_size, uint32_t * driver_version, int * physical_index);
GGML_API void ggml_backend_vk_get_device_memory(int device, size_t * free, size_t * total);

GGML_API ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num);
// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_API ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type(void);

#ifdef  __cplusplus
}
#endif
