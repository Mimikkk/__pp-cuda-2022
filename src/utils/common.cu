#include "common.hpp"
#include "console.hpp"

fn gpu_assert(cudaError_t code, const char *file, int line) -> void {
  if (code != cudaSuccess) {
    console::error("GPU: ", cudaGetErrorString(code), " in ", file, ":", line);
    exit(code);
  }
}
