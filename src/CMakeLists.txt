# TODO: should not use this
if (WIN32)
    if (BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()

#
# libraries
#

# llama

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-vocab.cpp
            llama-grammar.cpp
            llama-sampling.cpp
            unicode.h
            unicode.cpp
            unicode-data.cpp
            network-utils.cpp
            ../common/profiler.cpp
            ../common/log.cpp
            )

target_include_directories(llama PUBLIC . ../include ../common ${cppzmq_SOURCE_DIR} ${libzmq_SOURCE_DIR}/include)
target_compile_features   (llama PUBLIC cxx_std_11) # don't bump

target_link_libraries(llama PUBLIC ggml libzmq-static)
foreach(backend METAL CUDA VULKAN HIP)
    if(GGML_${backend})
        target_compile_definitions(llama PRIVATE GGML_USE_${backend})
    endif()
endforeach()

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_SHARED LLAMA_BUILD)
endif()

if (WIN32)
    target_link_libraries(llama PUBLIC ws2_32 psapi)
endif()
