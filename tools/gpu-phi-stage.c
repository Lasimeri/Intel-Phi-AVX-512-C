/* gpu-phi-stage [CARD [GPU_BUS_HEX [MIB]]]: step 1 of the host-staged path
 * from a GPU to a Phi card (docs/research/phi-vram-dma.md, the fallback):
 * system memory is the buffer between them. The GPU's copy engine writes
 * 2 MiB chunks of a device buffer into the card's host window (the
 * RESIDENT staging slots); the card's DMA engine pulls each chunk into a
 * pool page (K_RESIDENT). One coordinator keeps up to 16 GPU copies ahead
 * and reuses a staging slot only after the card has replied for it.
 * Pass 1 is timed per leg and end to end; pass 2 pulls every chunk back
 * (K_PUSH) and compares every 32-bit word with the pattern the GPU wrote.
 * Every transfer is on its owner's normal path: OpenCL for the GPU, the
 * worker's slot protocol (vpu_fexec.h) for the card; no device's memory is
 * mapped for another. See gpu-phi-stage.md. */
#define _GNU_SOURCE
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "vpu_fexec.h"
#include "vpu_proto.h"

/* cl_nv_device_attribute_query: the device's PCI bus (hwloc include/hwloc/opencl.h,
 * NVIDIA's cl_ext.h: 0x4008). */
#define CL_DEVICE_PCI_BUS_ID_NV 0x4008

#define CHUNK (2u << 20)
#define NSTAGE 16 /* VPU_RESIDENT_BYTES / CHUNK */
#define TAG_BASE 0x600000000000ull /* chunk tags inside the worker's reserved range */
#define POOL_USE 120               /* distinct tags, under the worker's -R 128 pool */
#define WIN_MAP (128u << 20)       /* the slots, the slot data and the RESIDENT area */
#define TIMEOUT_NS 5000000000ull

static const char *fill_src = "__kernel void fill(__global uint *p, uint base)\n"
                              "{ uint i = (uint)get_global_id(0); p[i] = ((base + i) * 2654435761u) ^ 0x5a5a5a5au; }\n";

static uint32_t pat(uint32_t i) { return (i * 2654435761u) ^ 0x5a5a5a5au; }

static uint64_t now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

static uint8_t *win;
static volatile struct vpu_slot_req *rq;
static volatile struct vpu_slot_reply *rp;
static volatile uint16_t *bell;
static uint64_t seq;

/* The request cleared, its fields set by the caller, then posted: the
 * sequence number and kind, a fence, the doorbell (kind << 11 | seq). */
static void clear_req(void) { memset((void *)&rq->kind, 0, sizeof(struct vpu_slot_req) - 8); }

static int post_wait(uint32_t kind)
{
    rq->seq = ++seq;
    rq->kind = kind;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    *bell = (uint16_t)((kind << 11) | (seq & 0x7ff));
    uint64_t t0 = now_ns();
    while (rp->seq != seq) {
        __builtin_ia32_pause();
        if (now_ns() - t0 > TIMEOUT_NS)
            return 1;
    }
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    return rp->status;
}

static int card_resident(uint64_t tag, unsigned stage)
{
    clear_req();
    rq->fn_id = tag;
    rq->entry = stage;
    rq->threads = 0;          /* the immutable class: a pool page filled by DMA */
    rq->flags = 0xffffffffu;  /* every 64 KiB unit of the chunk */
    rq->fill_off = 0;
    rq->fill_len = CHUNK;
    return post_wait(VPU_K_RESIDENT);
}

static int card_push(uint64_t tag, unsigned stage)
{
    clear_req();
    rq->fn_id = tag;
    rq->entry = stage;
    rq->fill_len = CHUNK;
    return post_wait(VPU_K_PUSH);
}

static uint8_t *stage_at(unsigned s) { return win + VPU_OFF_RESIDENT + (size_t)s * CHUNK; }

#define CLCHECK(x)                                                                                       \
    do {                                                                                                 \
        cl_int e_ = (x);                                                                                 \
        if (e_ != CL_SUCCESS) {                                                                          \
            fprintf(stderr, "gpu-phi-stage: %s: OpenCL error %d\n", #x, e_);                             \
            return 1;                                                                                    \
        }                                                                                                \
    } while (0)

int main(int argc, char **argv)
{
    int card = argc > 1 ? atoi(argv[1]) : 0;
    unsigned bus = argc > 2 ? (unsigned)strtoul(argv[2], NULL, 16) : 0xc7;
    unsigned mib = argc > 3 ? (unsigned)atoi(argv[3]) : 256;
    if (mib < 2 || mib % 2) {
        fprintf(stderr, "usage: gpu-phi-stage [CARD [GPU_BUS_HEX [MIB, even]]]\n");
        return 2;
    }
    size_t total = (size_t)mib << 20, nchunks = total / CHUNK;

    /* The GPU: its device buffer, filled with the pattern by the GPU. */
    cl_platform_id plats[8];
    cl_uint np = 0;
    CLCHECK(clGetPlatformIDs(8, plats, &np));
    cl_device_id dev = NULL;
    for (cl_uint p = 0; p < np && !dev; p++) {
        cl_device_id devs[16];
        cl_uint nd = 0;
        if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 16, devs, &nd) != CL_SUCCESS)
            continue;
        for (cl_uint d = 0; d < nd; d++) {
            cl_uint b = 0;
            if (clGetDeviceInfo(devs[d], CL_DEVICE_PCI_BUS_ID_NV, sizeof b, &b, NULL) == CL_SUCCESS && b == bus)
                dev = devs[d];
        }
    }
    if (!dev) {
        fprintf(stderr, "gpu-phi-stage: no OpenCL GPU on bus %02x\n", bus);
        return 1;
    }
    cl_int e;
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &e);
    CLCHECK(e);
    cl_queue_properties qp[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, qp, &e);
    CLCHECK(e);
    cl_mem dbuf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, total, NULL, &e);
    CLCHECK(e);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &fill_src, NULL, &e);
    CLCHECK(e);
    CLCHECK(clBuildProgram(prog, 1, &dev, "", NULL, NULL));
    cl_kernel k = clCreateKernel(prog, "fill", &e);
    CLCHECK(e);
    cl_uint base = 0;
    CLCHECK(clSetKernelArg(k, 0, sizeof dbuf, &dbuf));
    CLCHECK(clSetKernelArg(k, 1, sizeof base, &base));
    size_t words = total / 4;
    CLCHECK(clEnqueueNDRangeKernel(q, k, 1, NULL, &words, NULL, 0, NULL, NULL));
    CLCHECK(clFinish(q));

    /* The card: its window, the slot opened, the residency flushed. */
    char path[64];
    if (card == 0)
        snprintf(path, sizeof path, "/dev/shm/phi-hostmem");
    else
        snprintf(path, sizeof path, "/dev/shm/phi-hostmem-%d", card);
    int wfd = open(path, O_RDWR);
    if (wfd < 0) {
        fprintf(stderr, "gpu-phi-stage: %s: %s\n", path, strerror(errno));
        return 1;
    }
    win = mmap(NULL, WIN_MAP, PROT_READ | PROT_WRITE, MAP_SHARED, wfd, 0);
    if (win == MAP_FAILED) {
        fprintf(stderr, "gpu-phi-stage: mmap %s: %s\n", path, strerror(errno));
        return 1;
    }
    if (*(volatile uint64_t *)(win + VPU_OFF_READY) != VPU_MAGIC) {
        fprintf(stderr, "gpu-phi-stage: card %d: no worker polling\n", card);
        return 1;
    }
    unsigned slot = 0;
    rq = (volatile struct vpu_slot_req *)(win + VPU_OFF_SLOTS + slot * VPU_SLOT_BYTES);
    rp = (volatile struct vpu_slot_reply *)((volatile uint8_t *)rq + VPU_SLOT_REPLY);
    bell = (volatile uint16_t *)(win + VPU_OFF_SLOTSEQ) + slot;
    *bell = 0;
    rp->seq = 0;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    usleep(1000);
    clear_req();
    int st = post_wait(VPU_K_FLUSH);
    if (st != 0) {
        fprintf(stderr, "gpu-phi-stage: K_FLUSH: status %d (a worker in slot mode is needed: -S 1)\n", st);
        return 1;
    }
    printf("card %d slot %d open; GPU bus %02x; %u MiB in %zu chunks of 2 MiB through %d staging slots\n", card,
           slot, bus, mib, nchunks, NSTAGE);

    /* Pass 1: the GPU's copies up to NSTAGE ahead, the card pulling in order. */
    cl_event ev[NSTAGE];
    uint64_t gpu_ns = 0, card_ns = 0;
    size_t next_gpu = 0, next_card = 0;
    uint64_t t0 = now_ns();
    while (next_card < nchunks) {
        while (next_gpu < nchunks && next_gpu - next_card < NSTAGE) {
            unsigned s = (unsigned)(next_gpu % NSTAGE);
            CLCHECK(clEnqueueReadBuffer(q, dbuf, CL_FALSE, next_gpu * CHUNK, CHUNK, stage_at(s), 0, NULL, &ev[s]));
            next_gpu++;
        }
        clFlush(q);
        unsigned s = (unsigned)(next_card % NSTAGE);
        CLCHECK(clWaitForEvents(1, &ev[s]));
        cl_ulong a = 0, b = 0;
        clGetEventProfilingInfo(ev[s], CL_PROFILING_COMMAND_START, sizeof a, &a, NULL);
        clGetEventProfilingInfo(ev[s], CL_PROFILING_COMMAND_END, sizeof b, &b, NULL);
        gpu_ns += b - a;
        clReleaseEvent(ev[s]);
        uint64_t c0 = now_ns();
        st = card_resident(TAG_BASE + (uint64_t)(next_card % POOL_USE) * CHUNK, s);
        card_ns += now_ns() - c0;
        if (st != 0) {
            fprintf(stderr, "gpu-phi-stage: K_RESIDENT of chunk %zu: status %d\n", next_card, st);
            return 1;
        }
        next_card++;
    }
    uint64_t wall = now_ns() - t0;
    printf("pass 1: %u MiB end to end in %.1f ms = %.2f GB/s; GPU to host %.2f GB/s (%.1f ms of copies); "
           "host to card %.2f GB/s (%.1f ms of requests)\n",
           mib, wall / 1e6, total / (double)wall, total / (double)gpu_ns, gpu_ns / 1e6, total / (double)card_ns,
           card_ns / 1e6);

    /* Pass 2: each chunk to the card again, pushed back into a poisoned
     * staging slot, every word compared. */
    size_t bad = 0, checked = 0;
    for (size_t i = 0; i < nchunks; i++) {
        uint64_t tag = TAG_BASE + (uint64_t)(i % POOL_USE) * CHUNK;
        CLCHECK(clEnqueueReadBuffer(q, dbuf, CL_TRUE, i * CHUNK, CHUNK, stage_at(0), 0, NULL, NULL));
        if ((st = card_resident(tag, 0)) != 0) {
            fprintf(stderr, "gpu-phi-stage: pass 2 K_RESIDENT of chunk %zu: status %d\n", i, st);
            return 1;
        }
        memset(stage_at(1), 0xa5, CHUNK);
        if ((st = card_push(tag, 1)) != 0) {
            fprintf(stderr, "gpu-phi-stage: pass 2 K_PUSH of chunk %zu: status %d\n", i, st);
            return 1;
        }
        const uint32_t *w = (const uint32_t *)stage_at(1);
        uint32_t first = (uint32_t)(i * (CHUNK / 4));
        for (uint32_t j = 0; j < CHUNK / 4; j++) {
            if (w[j] != pat(first + j)) {
                if (bad < 4)
                    printf("chunk %zu word %u: %08x, expected %08x\n", i, j, w[j], pat(first + j));
                bad++;
            }
        }
        checked += CHUNK / 4;
    }
    printf("pass 2: %zu of %zu words differ from the GPU's pattern after GPU -> host -> card -> host\n", bad,
           checked);
    clear_req();
    post_wait(VPU_K_FLUSH);
    printf("%s\n", bad == 0 ? "PASS" : "FAIL");
    return bad != 0;
}
