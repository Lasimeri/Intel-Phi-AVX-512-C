# q4_0-kernels.c

Prints the six Q4_0 kernels of `card/vpu/kernels.S`: `phi_q4_0_1`,
`phi_q4_0_4`, `phi_q4_0_8` and their float16-activation twins
`phi_q4_0_1h`, `phi_q4_0_4h`, `phi_q4_0_8h`, in that file's line format
(the `mvex.inc` macro that assembles, a tab, the instruction in Intel
syntax as a comment). Its output was appended to `kernels.S` on
2026-11-09; the program stays as the record of how those 1263 lines were
made and as the way to make them again.

```
tcc -run tools/q4_0-kernels.c > q4_0.S
```

## What a kernel does

One call takes one superblock of a Q4_0 weight row, eight 18-byte blocks
(144 bytes, 256 weights), against 1, 4 or 8 activation rows, in the
calling convention every quantized kernel shares (`card/vpu/kernels.md`,
"Calling convention"). The format (ggml-common.h `block_q4_0`,
ggml-quants.c `dequantize_row_q4_0`): a float16 scale `d` at byte 0 of a
block, then sixteen bytes; the low nibbles are weights 0..15 of the
block, the high nibbles weights 16..31, each `(q - 8) * d`.

| step | instructions | where |
| --- | --- | --- |
| prefetch the superblock this thread takes next | four `vprefetch0` two calls ahead, four `vprefetch1` four ahead | 144 bytes at any alignment touch at most four lines |
| the eight scales | a masked `vloadunpacklps`/`hps` `{float16}` pair per block at `18 i`, lane `i` of zmm16 | the single unmasked lane receives the element at the address (`kernels.md`, the unpack facts) |
| store them | `vmovaps [rcx+0], zmm16` | the scratch's first 32 bytes; both nibble vectors of block `i` use `[rcx + 4 i]`, so no expanding permute is needed |
| the bytes | a `vloadunpacklps`/`hps` `{uint8}` pair at `18 i + 2`, four blocks a batch into zmm8..11 | float lanes, 0..255 |
| the high nibble | `vmulps` by 2^-4 (`[r9+24]`), `vrndfxpntps` round down | zmm12..15 |
| the low nibble | `vfnmadd231ps`: `b - 16 hi` (`[r9+0]`) | zmm8..11 |
| `q - 8` | `vsubps` by 8.0 (`[r9+44]`, the constant added to `g_consts` with these kernels) | all eight vectors |
| `(q - 8) * d` | `vmulps` by the block's scale, broadcast from the scratch | all eight vectors |
| the products | `vfmadd231ps acc, w, [row + off]`, one per activation row; `{float16}` and a 32-byte stride in the twins | vector `2 i` is the low nibbles of block `i`, `2 i + 1` the high ones |

Every value before the scale multiply is an exact small integer in a
float lane, and the subtraction comes before the multiply, so a weight
is rounded once, exactly where `dequantize_row_q4_0` rounds it: the
card's weights are the reference's bit for bit. (The K formats of
`kernels.md` fold their constant into a minuend, `w * sc - mn`, and
round twice.)

The registers are the family's: zmm0..7 accumulate (the one-row kernel
rotates over four and adds them at the end), zmm8..15 are the batch,
zmm16 the scales; the activation rows' bases are `rsi`, `r10`, `r11`,
`rax`, `rbx`, `r13`, `r14`, `r15`, the last four pushed and popped by the
eight-row kernels. A stage is written for the whole batch of four blocks
before the next stage, so dependent instructions sit a batch apart.

## Checks

- Assembled alone with `mvex.inc`: 11136 bytes of text, six symbols.
- `card/vpu/build.sh`: the worker with them is 272424 bytes, audit `0
  illegal, 0 suspect`.
- On card 0 (2026-11-09), `phi-vpu matmul-check -c 0 --only q4_0
  --threads 114 --repeat 3`, with `--act 0` and `--act 1`: every plain,
  mixture, several-matrix and fused feed-forward case ok. The card's
  compute time at 4096 x 5120: float32 activations 139.3 GFLOP/s at one
  row, 246.4 at eight, 251.3 at sixty-four; float16 126.4, 452.9, 490.3.
