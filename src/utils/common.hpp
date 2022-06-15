#pragma once
#include "../imports.hpp"

typedef double f64;
typedef float f32;

typedef size_t usize;
typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;

typedef int64_t i64;
typedef int32_t i32;
typedef int16_t i16;
typedef int8_t i8;
typedef char byte;
typedef unsigned char ubyte;

#define var auto
#define let const var
#define loop for(;;)
#define fn auto

#define catching(ans) gpu_assert((ans), __FILE__, __LINE__)
extern fn gpu_assert(cudaError_t code, const char *file, int line) ->  void;
