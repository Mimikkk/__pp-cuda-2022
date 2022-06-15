#include "imports.hpp"
#include "utils/console.hpp"

#define kernel_fn __global__ void

template<typename T>
fn generate_array(usize len) {
  var data = new T[len];
  for (var i = 0u; i < len; ++i) data[i] = i;
  return data;
}

template<typename T>
kernel_fn reduce_kernel(T *const blocks, const T *const array, usize len) {
  extern __shared__ T shared[];

  let global_tid = (blockDim.x * 2) * blockIdx.x + threadIdx.x;
  let tid = threadIdx.x;

  shared[tid] = 0;

  if (global_tid < len) shared[tid] = array[global_tid] + array[global_tid + blockDim.x];

  __syncthreads();

  for (var s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) shared[tid] += shared[tid + s];

    __syncthreads();
  }
  __syncthreads();

  if (tid == 0) blocks[blockIdx.x] = shared[0];
}

namespace cpu {
  template<typename T>
  fn sum(const T *array, usize len) {
    T sum = 0;
    for (var i = 0; i < len; ++i)sum += array[i];
    return sum;
  }
}

namespace gpu {
  enum Direction {
    CpuToGpu = cudaMemcpyHostToDevice,
    CpuToCpu = cudaMemcpyHostToHost,
    GpuToCpu = cudaMemcpyDeviceToHost,
    GpuToGpu = cudaMemcpyDeviceToDevice,
  };

  template<typename T>
  inline fn malloc(usize len) {
    T *device_in;
    catching(cudaMalloc(&device_in, len * sizeof(T)));
    return device_in;
  }

  template<typename T>
  inline fn copy(const T *source, T *destination, usize len, Direction direction) {
    catching(cudaMemcpy(destination, source, len * sizeof(T), cudaMemcpyKind(direction)));
  }

  template<typename T>
  inline fn from(const T *source, usize len) {
    let destination = malloc<T>(len);
    copy(source, destination, len, Direction::CpuToGpu);
    return destination;
  }

  template<typename T>
  inline fn to(const T *source, usize len) {
    let destination = new T[len];
    copy(source, destination, len, Direction::GpuToCpu);
    return destination;
  }

  template<typename T>
  inline fn pass_to(const T *source, usize len) {
    let destination = malloc<T>(len);
    copy(source, destination, len, Direction::GpuToGpu);
    return destination;
  }

  template<typename T>
  inline fn calloc(usize len, T value) {
    T *destination = malloc<T>(len);
    catching(cudaMemset(destination, value, sizeof(T) * len));
    return destination;
  }

  template<typename ...Args>
  inline fn free(Args &... source) {
    (catching(cudaFree(source)), ...);
  }

  inline fn finalize() { cudaDeviceReset(); }

  template<typename T>
  inline fn sum(const T *array, usize len, usize block_size = 1024) -> T {
    let span = sizeof(T);
    let grid_size = usize(len <= block_size
                          ? std::ceil(float(len) / float(block_size))
                          : len / block_size + (len % block_size != 0));

    var d_block_sums = gpu::calloc<T>(grid_size, 0);

    reduce_kernel<<<grid_size, block_size, span * block_size>>>(d_block_sums, array, len);
    if (grid_size <= block_size) {
      let d_total_sum = gpu::calloc<T>(1, 0);

      reduce_kernel<<<1, block_size, span * block_size>>>(d_total_sum, d_block_sums, grid_size);

      let total_sum = gpu::to(d_total_sum, 1)[0];
      gpu::free(d_total_sum, d_block_sums);
      return total_sum;
    }

    let total_sum = gpu::sum(d_block_sums, grid_size, block_size);
    gpu::free(d_block_sums);
    return total_sum;
  }
}

namespace hybrid {
  template<typename T>
  fn sum(const T *h_in, const T *d_in, usize len, usize block_size) {
    return len < 2 << 18 ? cpu::sum(h_in, len) : gpu::sum(d_in, len, block_size);
  }
}

template<typename Fn>
fn timeit(Fn function) -> f64 {
  let start = std::chrono::high_resolution_clock::now();
  function();
  let end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration<f64>(end - start).count();
}

fn main() -> i32 {
  let block_size = 1024;
  u32 type;

  for (var n = 1; n < 28; ++n) {
    let length = (1 << n);

    console::print();
    console::event("array length: ", length);

    let h_in = generate_array<decltype(type)>(length);

    decltype(type) cpu_total;
    let cpu_duration = timeit([&]() { cpu_total = cpu::sum(h_in, length); });

    let d_in = gpu::from(h_in, length);

    decltype(type) gpu_total;
    let gpu_duration = timeit([&]() { gpu_total = gpu::sum(d_in, length, block_size); });

    decltype(type) hybrid_total;
    let hybrid_duration = timeit([&]() { hybrid_total = hybrid::sum(h_in, d_in, length, block_size); });

    console::info("CPU time: ", cpu_duration, "s");
    console::log(" - with sum: ", cpu_total);
    console::info("GPU time: ", gpu_duration, "s");
    console::log(" - with sum: ", gpu_total);
    console::info("Hybrid time: ", hybrid_duration, "s");
    console::log(" - with sum: ", hybrid_total);

    free(h_in);
  }
  gpu::finalize();
}
