# glue.S

The ggml side of the card backend: the glue ggml's C interface
requires and nothing more, what `csrc/ggml-phi.c` was, slot for slot.
It registers one device of type ACCEL named `Phi` that shares the
CPU's host buffers, accepts the `MUL_MAT`, `MUL_MAT_ID` and (opt in)
SwiGLU nodes whose weight type and shape the cards take, and runs each
in three steps: `phi_ggml_begin` (backend.S: the cards start on their
rows), the host's own rows on a private ggml CPU backend built here
from ggml's own kernels, `phi_ggml_end` (the cards' rows gathered).
Every ggml function it calls is an undefined symbol of the library,
bound when the program loads it (`imports.list`, `build.md`); every
ggml layout number comes from `ggml_layout.inc`.

## Registration

`ggml_backend_init` (what `ggml_backend_load` looks up) returns
`ggml_backend_phi_reg`: the `ggml_backend_reg` with the API version
and the registration interface table, built once by `tables_init`
(each interface pointer at the slot `ggml_layout.inc` names, the rest
null). The device count is decided once, by `phi_ggml_open`
(`phi_reg_get_device_count`): 1 when a card answers, else 0, and
llama.cpp runs on the CPU without the backend. The device's properties
(`phi_dev_get_props`): the name, the description, memory 0/0, type
ACCEL, capabilities `buffer_from_host_ptr` only. `phi_dev_init_backend`
hands out a backend from a pool of eight (`BE_SLOT`, `BE_USED`).

## What is accepted (`phi_dev_supports_op`)

- `MUL_MAT` with a weight source that is a leaf named `*weight*`, of a
  type the cards take (`card_type`: F32, F16, BF16, Q4_K, Q5_K, Q6_K,
  Q8_0, IQ4_XS, and Q4_0 since 2026-11-09), whose shape `phi_ggml_supports` accepts, not flagged
  `GGML_HINT_SRC0_IS_HADAMARD` (`T_OP_PARAMS+4`); the weight is noted
  for the shares (`phi_ggml_note_weight`, with its layer from the name
  `blk.N.`).
- `MUL_MAT_ID` likewise, over the experts.
- Either multiply only when `phi_ggml_take` (`backend.md`) says the cards
  may take it: not when a rule `prepare` applies would keep it with the
  host anyway (the batch class: `PHI_GGML_PP_ONLY`, `PHI_GGML_TG_ONLY`,
  float weights at a batch; and, once the weight has been multiplied, its
  split: the host keeping every row, the judge having taken it off at
  that class). Declined, the multiply stays inside the CPU's sub-graph and
  is not offered for the shares. ggml's scheduler asks again whenever
  llama.cpp builds a graph (each new shape: in a server, each request),
  not at a graph it reuses.
- `GLU` (SwiGLU, split) when `PHI_GGML_FFN` is on and the node is part
  of a fusable block (`ffn_enabled`; off under `PHI_GGML_OFFLOAD` with
  the message the C gave; off while `PHI_GGML_STREAM` is on, since
  `K_FFN` names resident slices).
- views, reshapes, permutes and transposes are ggml's own and pass.

The scheduler asks about every node of a graph before it computes any,
so at the first multiply the offered weights are the model.

## Computing a graph (`phi_graph_compute`)

The nodes are scanned once into `role[]`: each `MUL_MAT`, `MUL_MAT_ID`,
and each block of four (`find_quad`: gate, up, GLU, down over the same
input, when the fused path is on) gets its role, the rest are skipped.
Then in order:

- `run_mul_mat`: `phi_ggml_begin(a, type, m, k, nb_a, keep, b, n,
  nb_b)`; for each host range `phi_ggml_host_range(i)` the host's rows
  through `host_rows` (a ggml context of two tensors aliasing the
  weight's rows and the activations, one `ggml_mul_mat`, computed on
  the private CPU backend, the result rows at their place in the
  destination); `phi_ggml_end(d, nb_d)`.
- `run_mul_mat_id`: `phi_ggml_begin_id` with the mixture's columns;
  the host's rows through `host_rows_id`, over the backend's
  substituted ids (`phi_ggml_host_ids`) when there are any;
  `phi_ggml_end_id`.
- `run_id_pair`: a layer's gate and up over the same ids as one
  request (`phi_ggml_begin_id_pair`; -2 means the two go one by one);
  `host_rows_id_pair`; `phi_ggml_end_id_pair`. `id_pair` pairs two
  quantized weights of any types, or two of the same float type (the
  card runs a float request's pieces one by one with its float
  kernels); until 2026-10-09 the test was "a card type of 2 or more",
  which let a bfloat16 pair through to a request the card rejected.
- `run_ffn`: `phi_ggml_ffn_begin(&args)`; -2 declines and the four
  nodes run as they are; else the host's runs of the intermediate
  through `host_ffn` (gate and up rows, the SwiGLU and down's columns,
  in a scratch buffer), the result zeroed when the host has no run,
  then `phi_ggml_ffn_end`.
- `PHI_GGML_GRAPH=N` dumps the first N sub-graphs (`dump_graph`), one
  line per node with its op, name, sources and data pointers.

Every one of these prints the lines the C printed under
`PHI_GGML_VERBOSE` (`host rows A..B of TYPE MxK, n N: compute T ms`,
the feed-forward's host part), with `w_elapsed_ms` giving C's `%.3f`
of the elapsed milliseconds.

## A multiply issued ahead of its node (`PHI_GGML_AHEAD`, 2026-11-09)

ggml hands this backend one sub-graph at a time, and a model's graph
puts other backends' nodes between two multiplies that read the same
activations (a layer's `attn_qkv` and `attn_gate` both read `attn_norm`;
`attn_q`, `attn_v` and `attn_k` likewise): the second is begun only
after the first one's result has been copied to the GPU, worked on
there and the scheduler has come back, though its input was there all
along. With `PHI_GGML_AHEAD=1` the cards do not wait for that.

| step | where | what |
| --- | --- | --- |
| learn | `run_mul_mat`, on entry | under the weight of the multiply before this one (`ah_prev`), `ah_succ` keeps this one's weight, activations' address, rows, columns, row stride, card type and `keep`: 2048 entries of 64 bytes by the weight's address (`ah_slot`: eight places tried) |
| issue ahead | `run_mul_mat`, after `phi_ggml_end` | when the entry under this weight names activations at the address this multiply read: the activations copied to `ah_copy`, `phi_ggml_begin` for that multiply at once, its arguments kept (`ah_a`, `ah_x`, `ah_n`, `ah_nb_b`, `ah_m`, `ah_nr`), `ah_valid` set; the return to ggml leaves its request with the cards |
| take | `run_mul_mat`, on entry, `ah_valid` set | the node's weight, its activations' address, their count and stride, its rows, and the activations' bytes against `ah_copy` (`repe cmpsb`): all equal, the node skips `phi_ggml_begin` and goes to its host ranges and `phi_ggml_end` |
| drop | `ahead_drain` | any difference: `phi_ggml_end` into `ah_drain`, and the node runs as it always did; also at the top of `run_ffn`, `run_id_pair` and `run_mul_mat_id`, which ring the cards themselves |

What makes it safe:

- A wrong guess costs a multiply, never a result: the node's request is
  the one issued ahead only when every argument and every byte of its
  input is what that request was begun with.
- `phi_ggml_begin` takes the backend's lock and drops it before it
  returns (`backend.S`, `begin_common`); between it and `phi_ggml_end`
  the multiply lives in the backend's statics, as it does inside
  `run_mul_mat`, and nothing else enters the backend while a graph is
  computed.
- The window's doorbell lock stays taken from the request to its reply
  (`../common/window.md`). Its only other user is a DMA router's fault
  handler, which is asked for a card's rows when a batch is computed on
  a GPU from the host's mapping; this backend declines those batches
  whole, so no fault of that kind falls inside a pass that has a
  multiply outstanding.
- A multiply is begun ahead only when its activations are at the
  address the finished one read, so none is left outstanding at the end
  of a pass: the next pass's first multiply reads another tensor.
- It needs `PHI_GGML_OFFLOAD` (`ahead_enabled` says so and stays off
  without it): outside the offload `phi_ggml_end` times a multiply from
  its `phi_ggml_begin` and judges the tensor by it, and a request left
  running across ggml's other work would be counted against its tensor.

The copy is 1 MiB and the drop buffer 8 MiB (`AH_COPY_MAX`,
`AH_DRAIN_MAX`); a multiply whose activations or results would not fit
is not begun ahead. `ah_report` says the two counts at 256 multiplies
taken and then every 16384 (`ahead: N multiplies were with the cards
before their node arrived, M begun ahead were not what came`).

Measured on card 0 with a 27 B Q4_0 model, one server, the GPU doing
every node but the multiplies, the cards holding every row of 60
layers, 64 tokens a run, four runs each, the plain runs first:

| | plain | `PHI_GGML_AHEAD=1` |
| --- | --- | --- |
| 64 tokens | 25.979, 25.703, 25.423, 24.791 s | 24.802, 24.055, 24.061, 24.019 s |
| tokens a second | 2.425 to 2.541 | 2.540 to 2.623 |
| the card's compute time | 20.09 to 20.18 s | 20.04 to 20.09 s |
| the output at temperature 0, 24 tokens with the top five log-probabilities of each | the reference (two runs equal) | the same bytes (two runs) |

The mean falls from 25.474 to 24.234 s (5.1 %) with the card's compute
time unchanged: the time between its multiplies is what shrank, about
19 ms a token. 16384 multiplies had been taken and none dropped when
the count was last said. The two sets were not interleaved, and their
ranges touch (24.791 and 24.802 s).

## The host's rows

`host_backend` creates the private CPU backend once: `ggml_backend_dev_by_type(CPU)`,
`ggml_backend_dev_init`, `ggml_backend_set_n_threads` through the
registration's `get_proc_address` with `PHI_GGML_HOST_THREADS`
(default 12), and a threadpool of its own unless
`PHI_GGML_HOST_POOL=0`, polling `PHI_GGML_HOST_POLL` (default 0: the
pool's threads sleep between graphs; the busy wait is only asked for).
The messages (`the host's rows run on ggml's CPU backend with N
threads`, the threadpool's, the `-t` advice) are the C's.

A ggml call is made with the stack 16-byte aligned; `ggml_init` takes
its 24-byte parameter struct in memory (built on the stack); the
threadpool parameters come back through a hidden pointer.
