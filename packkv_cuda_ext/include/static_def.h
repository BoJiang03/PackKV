#ifndef PACKKV_STATIC_DEF_H
#define PACKKV_STATIC_DEF_H

constexpr size_t WARP_THREAD_NUM = 32;
constexpr size_t HEAD_DIM = 128;
constexpr size_t PACK_ELE_NUM = 16;
// k setting
constexpr size_t PACK_NUM_PER_UNIT = 8; // the number of united packs, the same as the number of elements for each decompress+mat_vec_mul thread
constexpr size_t THREAD_NUM_PER_HEAD_DIM = HEAD_DIM / PACK_NUM_PER_UNIT;

#endif //PACKKV_STATIC_DEF_H
