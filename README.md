# slabhra

**Differentiable programming for Almide** — reverse-mode automatic
differentiation, tensors, and optimizers, written entirely in Almide.

*slabhra* (Irish, [ˈsˠlˠəuɾˠə]) = **chain** — for the chain rule that every
gradient flows through.

---

## What it is

A PyTorch-shaped autograd engine that is a **language-native library, not a
framework bolted onto a host language**. You build a model as an ordinary
Almide program; `slabhra` records the computation and gives you gradients.

```
slabhra/
  src/
    tensor.almd   Tensor + reverse-mode autograd (flat tape, depth-ordered backward)
    ops.almd      ops + their vjp (gradients delegate to the same forward kernels)
    optim.almd    SGD, Adam
    nn.almd       layers / modules            (later)
  examples/
    mlp/          tiny MLP, verified against PyTorch
    dlgn/         differentiable logic-gate networks (gate-count / depth Pareto)
```

## Why it exists (and why not just use Burn / PyTorch)

Rust already has Burn; Python has PyTorch. `slabhra` is not a re-implementation
of either — it exists to do what neither structurally can, because it is part
of a **verifiable, multi-target language** rather than a library on top of one:

- **Language-native, not a backend decorator.** Burn injects autograd by
  wrapping a backend in `Autodiff<B>` and storing the graph in a global,
  mutex-guarded registry with on-the-fly graph merging and `Arc`-refcount
  liveness — most of that machinery is a *workaround* for the host language not
  owning the tensor type. Almide owns the tensor type, so `slabhra` keeps an
  **explicit, owned tape** and deletes that entire runtime.
- **Verifiable across targets.** The same model compiles to CPU-SIMD / WASM /
  WebGPU, and slabhra is built so results stay **bit-identical** across them
  (the discipline that took nn's LLM engine to a 1e-6 match with PyTorch and a
  bit-exact native↔wasm↔WGSL logic circuit).
- **Non-standard primitives are first-class.** Differentiable logic gates
  (DLGN), ternary, matmul-free — substrates PyTorch/Burn treat as exotic — are
  ordinary ops here. The first examples are gate-native, not FP-matrix-native.

## Design, borrowed and discarded (from studying Burn)

Kept from Burn's design:
- **depth = max(parents)+1**, then run backward by iterating depth-buckets in
  reverse — correct reverse-topological order with no sort at backward time.
- **vjps delegate to the forward kernels** (a backward never re-implements
  math), so autograd bookkeeping stays orthogonal to numerics.
- zero-overhead inference: no tape entry when nothing upstream needs grad.

Dropped (accidental complexity from being a host-language library):
- the global `Mutex` graph registry + graph-merging + `unsafe Send/Sync`,
- `Box<dyn Any>` type erasure for grads/state,
- refcount-driven liveness — an owned tape is freed when its scope ends.

## Verification discipline

Every op and example is checked against a **PyTorch oracle** (a `.py` next to
the Almide source) to numeric agreement, the same way nn's inference engine was
held to logits parity. "Works" means "matches the oracle," not "runs."

## Status

Bootstrapping from the `nn/research/autograd/` spikes: scalar → tensor autograd
(MLP matches PyTorch f64), DLGN gate-neuron + training + hardening, gate-count
and depth Pareto — all verified against PyTorch. Being lifted here into a real
tensor API (`Tensor`, `.backward()`, `Adam`).
