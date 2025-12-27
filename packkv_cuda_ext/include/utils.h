#ifndef PACKKV_UTILS_H
#define PACKKV_UTILS_H

#include <iostream>
#include <cstdlib>
#include <cstdint>

#define packkv_assert(cond, msg) \
    do { \
        if (!(cond)) { \
                std::cerr << "PackKV Assertion failed: " << msg << "\n" \
                      << "File: " << __FILE__ << "\n" \
                      << "Line: " << __LINE__ << std::endl; \
            std::abort(); \
        } \
    } while (0)

void copy_data(
    const uint8_t *from,
    uint8_t *to,
    size_t size
);

constexpr size_t min_bits_for_encode_len(size_t n) {
    n++;
    if (n == 0) {
        return 0;
    }
    size_t bits = 0;
    size_t val = n - 1;
    while (val > 0) {
        val >>= 1;
        bits++;
    }
    return bits > 0 ? bits : 1;
}

#endif //PACKKV_UTILS_H