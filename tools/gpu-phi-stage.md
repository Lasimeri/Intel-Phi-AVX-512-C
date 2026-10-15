# gpu-phi-stage.c: a GPU to a Phi card through system memory (step 1)

The host-staged path of
[`../docs/research/phi-vram-dma.md`](../docs/research/phi-vram-dma.md):
system memory is the buffer between a GPU and a Phi card, and each
transfer stays on its owner's normal path. The GPU's copy engine (OpenCL
`clEnqueueReadBuffer`) writes 2 MiB chunks of a device buffer into the
card's host window, in the 16 staging slots of the `RESIDENT` area
(window offset 96 MiB, `vpu_fexec.h`); the card's DMA engine pulls each
chunk into a pool page through the worker's slot protocol (`K_RESIDENT`:
`fn_id` the chunk's tag, `entry` the staging slot, class 0, every 64 KiB
unit, the fill range the whole chunk). No device's memory is mapped for
another.

One coordinator per GPU and card: up to 16 GPU copies ahead of the card,
a staging slot reused only after the card's reply for it (the card serves
one request at a time per slot). Pass 1 is timed per leg (the GPU's from
OpenCL profiling events, the card's around each request) and end to end.
Pass 2 sends every chunk again, pulls it back from the card's pool page
with `K_PUSH` into a staging slot poisoned with `0xa5`, and compares every
32-bit word with the pattern the GPU wrote, `(i * 2654435761) ^
0x5a5a5a5a` for word `i`.

```
gcc -O2 -Wall -Wextra -I$OPENCL_HEADERS -Icard/vpu -o gpu-phi-stage tools/gpu-phi-stage.c -lOpenCL
scripts/phi-vpu.sh -c 0 stop; PHI_VPU_ARGS="-S 1 -C 55 -T 2 -R 128 -s 1000" scripts/phi-vpu.sh -c 0 start 57
./gpu-phi-stage 0 c7 256          # card 0, the GPU on bus c7 (the card's neighbour), 256 MiB
```

The card's worker must run with slots (`-S 1`); nothing else may hold the
card (`scripts/card-users.sh`). The rack has no `opencl-headers`:
Khronos' OpenCL-Headers were cloned to `/mnt/raid5/phi/nvidia-p2p`.
`CL_DEVICE_PCI_BUS_ID_NV` (0x4008, `cl_nv_device_attribute_query`; the
value from hwloc's `include/hwloc/opencl.h` and NVIDIA's `cl_ext.h`)
picks the GPU by its PCI bus.

## Measured (2026-10-15, card 0 and GPU c7, 256 MiB)

| | rate | time |
| --- | --- | --- |
| GPU to the host window | 13.16 GB/s | 20.4 ms of copies |
| the host window to the card (`K_RESIDENT`) | 3.51 GB/s | 76.6 ms of requests, about 0.6 ms a chunk |
| end to end, pipelined | 2.89 GB/s | 93.0 ms |
| GPU to host to card to host | 0 of 67,108,864 words differ | PASS |

The legs barely overlap (in series they would take 97 ms): the reads go
to ordinary, unpinned host memory, and their completion appears to wait
for the driver's own copy out of its pinned bounce buffer. A pinned
staging area is the next step.
