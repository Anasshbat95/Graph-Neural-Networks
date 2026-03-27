from libc.stdint cimport int32_t, int64_t

ctypedef fused integral:
    int
    long
    int32_t
    int64_t