#pragma once

#include <map>
#include <unordered_map>

// Ranks: 2,3,4,5,6,7,8,9,T,J,Q,K,A (0-12)
constexpr const char* RANKS = "23456789TJQKA";

// Map from rank character to index (0-12)
inline std::unordered_map<char, int> RANK_TO_INDEX = {
    {'2', 0}, {'3', 1}, {'4', 2}, {'5', 3}, {'6', 4},
    {'7', 5}, {'8', 6}, {'9', 7}, {'T', 8}, {'J', 9},
    {'Q', 10}, {'K', 11}, {'A', 12}
};

// Generate all straight bitmasks
inline std::map<int, int> STRAIGHT_MASKS = []() {
    std::map<int, int> masks;
    // Regular straights (0-8)
    for (int start = 0; start <= 8; ++start) {
        int mask = 0;
        for (int i = 0; i < 5; ++i) {
            mask |= (1 << (start + i));
        }
        masks[mask] = start + 4;  // high card index
    }
    // Wheel straight (A-2-3-4-5)
    int wheelMask = (1 << 12) | (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3);
    masks[wheelMask] = 3;  // 5-high straight
    return masks;
}();

