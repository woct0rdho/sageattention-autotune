#include "../../csrc/qattn_cutlass/qk_int8_sv_f16_bwd_kernel_cutlass_sm80.cuh"

#include <cstdio>
#include <vector>

using namespace sageattention::qattn_cutlass_bwd;

int main()
{
  using Traits = BwdTileTraits<64, 64, 128, 8>;
  constexpr auto mn_to_tv = make_dQ_workspace_mn_layout<Traits>();
  constexpr auto fragment_layout = make_dQ_workspace_fragment_layout<Traits>();

  std::vector<int> seen(256, -1);
  for (int row = 0; row < 16; ++row)
  {
    for (int col = 0; col < 16; ++col)
    {
      const int lane = (row & 7) * 4 + ((col & 7) >> 1);
      const int value = (col & 1) | (((row >> 3) & 1) << 1) | (((col >> 3) & 1) << 2);
      const int expected = lane + 32 * value;
      const int actual = mn_to_tv(row + 16 * col);
      if (actual != expected || fragment_layout(lane, value) != expected || seen[expected] != -1)
      {
        std::printf("tile mapping mismatch row=%d col=%d actual=%d expected=%d fragment=%d\n",
                    row, col, actual, expected, fragment_layout(lane, value));
        return 1;
      }
      seen[expected] = row * 16 + col;
    }
  }

  for (int seq_len : {512, 513})
  {
    constexpr int head_dim = 64;
    const int padded_rows = ((seq_len + 31) / 32) * 32;
    std::vector<int> global_seen(padded_rows * head_dim, -1);
    for (int row = 0; row < padded_rows; ++row)
    {
      for (int col = 0; col < head_dim; ++col)
      {
        const int lane = (row & 7) * 4 + ((col & 7) >> 1);
        const int value = (col & 1) | (((row >> 3) & 1) << 1) | (((col >> 3) & 1) << 2);
        const int tile_offset = (row / 16) * (16 * head_dim) + (col / 16) * 256;
        const int offset = tile_offset + lane + 32 * value;
        if (offset < 0 || offset >= static_cast<int>(global_seen.size()) || global_seen[offset] != -1)
        {
          std::printf("global mapping mismatch seq=%d row=%d col=%d offset=%d\n", seq_len, row, col, offset);
          return 1;
        }
        global_seen[offset] = row * head_dim + col;
      }
    }
    for (int value : global_seen)
    {
      if (value == -1)
      {
        std::printf("global mapping left an unassigned word seq=%d\n", seq_len);
        return 1;
      }
    }
  }

  std::printf("dQ workspace CuTe/MMA-TV mapping is bijective for 16x16 tiles and seq_len 512/513\n");
  return 0;
}
