#include <utils.h>

void copy_data(
    const uint8_t *from,
    uint8_t *to,
    size_t size
) {
    for (size_t i = 0; i < size; i++) {
        to[i] = from[i];
    }
}
