# Looped Graph Transformer Plan
## OpenMythos / Parcae port into this GNN library

This document is the concrete implementation plan for bringing an OpenMythos / Parcae-style looped transformer into your graph library.

The goal is not a vague "recurrent GNN". The goal is the same recurrent scaffold:

**Prelude -> shared recurrent core looped `T` times -> Coda**

with the same stable update rule, the same loop-index differentiation idea, the same optional sparse MoE FFN, and then a graph-specific extension for **node-wise adaptive depth** and **adaptive adjacency**.

---

## 1. What is already in your library

The important point after reading your two local files is that the core graph primitives are already present.

From `layers (6).py`:

- `scaled_query_key_softmax(...)` and `scaled_dot_product_attention(...)` already implement sparse adjacency-masked attention.
- `Attention` already projects Q/K/V, applies optional RoPE, applies sparse attention with an adjacency mask, and supports gated attention.
- `Transformer` already gives you a pre-norm residual attention + gated MLP block.
- `GatedMLP` / `build_gated_mlp(...)` already provide the FFN style we need.
- `TemporalAttention` already exists as a graph-specific temporal corrector, but it is **not** part of the exact OpenMythos / Parcae baseline.
- `GraphNetBlock` already gives an edge-aware message-passing block with optional gates and relative RoPE.

From `processors (4).py`:

- `EncodeTransformDecode` already builds a sparse graph transformer stack using the adjacency matrix.
- `EncodeProcessDecode` already builds a message-passing graph processor using repeated blocks.
- Both files already use the exact decomposition we want to extend: encoder -> processor -> decoder.

So the missing pieces are **not** "graph attention" or "gated MLP". The missing pieces are:

1. a **shared** recurrent block instead of a list of distinct blocks,
2. **stable input injection** across loops,
3. **loop-index embedding** so the tied block can behave differently at loop 1 vs loop 8,
4. optional **MoE FFN** inside the recurrent block,
5. optional **node-wise ACT / halting**,
6. and then a graph-specific **adaptive adjacency policy**.

---

## 2. Exact target architecture

### 2.1 Baseline exact port

We will build a new processor that looks like this:

```text
graph.x, graph.edge_index, graph.pos
        |
        v
node encoder / optional edge encoder
        |
        v
Prelude P         (run once, n_P layers)
        |
        v
encoded input e   (frozen and reinjected every loop)
        |
        v
Recurrent core R  (shared n_R-layer stack, looped T times)
        |
        v
Coda C            (run once, n_C layers)
        |
        v
decoder / task head
```

For the **exact port baseline**, the recurrent core is a **shared stack of graph transformer blocks** built from your existing `Transformer` class and sparse adjacency attention.

To stay aligned with your mesh-transformer paper while matching the Parcae structural convention, we will enforce the symmetric split

$$
n_P = n_R = n_C = n_{\mathrm{prc}}
$$

rather than choosing Prelude / recurrent / Coda depths independently. The width, head count, and Gated-MLP ratio stay on your graph-model ladder; the part we import from Parcae is the **P/R/C depth symmetry and recurrence schedule**, not the raw LM width table.

That means the first implementation target is:

- node-state recurrence,
- sparse adjacency mask from `edge_index`,
- optional node-coordinate RoPE from `graph.pos`,
- gated FFN from your existing `GatedMLP`,
- Parcae-style symmetric `P/R/C` depth allocation,
- no adaptive adjacency yet,
- no `TemporalAttention` yet,
- no edge-conditioned attention yet.

That first version is the cleanest "same system in my library" interpretation.

### 2.2 Graph-specific extension path

After the exact port works, we add graph-native features in a controlled order:

1. **MoE FFN inside the recurrent block**  
   Routing is per node, exactly as token routing is per token in LLMs.

2. **Adaptive per-node depth / ACT halting**  
   Easy nodes halt earlier; hard nodes keep looping.

3. **Adaptive adjacency**  
   Once node halting exists, we turn it into compute savings:
   - first by skipping updates for halted target nodes,
   - then by learning edge gates for active nodes.

4. **Optional edge-aware recurrent variant**  
   If edge attributes matter strongly for the task, we add either:
   - edge-conditioned attention bias, or
   - a looped `GraphNetBlock` variant.

---

## 3. Math we will implement

## 3.1 Parcae / OpenMythos-style recurrent update

Let:

- `x` be the node features,
- `A0` be the base adjacency,
- `p` be the node coordinates / positions if available,
- `Enc` be the node encoder,
- `P` be the Prelude,
- `R` be the shared recurrent graph block,
- `C` be the Coda,
- `Dec` be the task decoder.

We define the encoded input once:

$$
e = \mathrm{Norm}\!\left(P(\mathrm{Enc}(x), A_0, p)\right)
$$

For the **repo-faithful MVP**, we initialize

$$
h_0 = e
$$

Then for loop index $t = 0, 1, \dots, T-1$:

$$
u_t = \mathrm{Norm}\!\left(\mathrm{LoopEmbed}(h_t, t) + e\right)
$$

$$
r_t = R_\theta(u_t, A_t, p)
$$

$$
h_{t+1} = \bar{A} \odot h_t + \bar{B} \odot e + r_t
$$

where $\bar{A} \in (0,1)^d$ is a learned per-channel stable diagonal recurrence and $\bar{B} \in \mathbb{R}^d$ is the learned input injection vector.

This is the graph adaptation of the same update rule used in OpenMythos / Parcae.

---

## 3.2 Stable parameterization of the recurrent state matrix

The stability-critical piece is the diagonal recurrent coefficient.

We will parameterize it as:

$$
a = \exp\!\Big(-\exp(\log \Delta + \log A)\Big)
$$

applied elementwise across channels. This guarantees:

$$
0 < a_k < 1 \quad \forall k
$$

so the discrete diagonal transition has spectral radius strictly below 1:

$$
\rho(\bar{A}) = \max_k a_k < 1
$$

This is the exact reason the loop does not explode when depth increases.

For the graph model, $\bar{A}$ is still channel-wise and shared across nodes. So the update per node feature vector is simply:

$$
h_{i,t+1} = a \odot h_{i,t} + b \odot e_i + r_{i,t}
$$

with `a` and `b` broadcast over nodes.

---

## 3.3 Loop-index embedding

Weight tying alone makes the recurrent block too rigid: the same parameters must behave like an "early pass" and a "late refinement pass" with no signal telling them which pass they are on.

So we add a loop-index embedding to a subset of channels:

$$
\mathrm{LoopEmbed}(h_t, t) = h_t + \phi_{\text{loop}}(t)
$$

where $\phi_{\text{loop}}(t)$ is a sinusoidal or learned embedding over loop depth.

This is the recurrence-depth analogue of positional encoding.

For graphs this is independent of node coordinates. Node-coordinate RoPE still handles geometry; loop-index embedding handles **recurrent depth**.

---

## 3.4 Optional ACT / node-wise adaptive depth

For node-wise halting, each node gets a per-loop halt probability:

$$
p_i^{(t)} = \sigma(w^\top h_i^{(t)} + b)
$$

Let cumulative halting mass be:

$$
c_i^{(t)} = c_i^{(t-1)} + p_i^{(t)}
$$

and define the active mask:

$$
m_i^{(t)} = \mathbf{1}[c_i^{(t)} < \tau]
$$

with threshold $\tau \approx 0.99$ as in ACT-style halting.

Once a node halts:

- its hidden state is frozen,
- it no longer receives an update,
- but it may still remain visible to active neighbors as a source of information.

The ACT output is the weighted sum over loop states:

$$
h_i^{\text{out}} = \sum_t \omega_i^{(t)} h_i^{(t)}
$$

where the final weight uses the standard ACT "remainder" trick.

This gives us **per-node variable depth** inside a single batched graph forward pass.

---

## 3.5 Adaptive adjacency: the graph-specific extension

This is where the graph version becomes more interesting than the token version.

### Phase 1: target-active masking

If an edge is directed as $j \to i$, and node $i$ has halted, there is no reason to keep computing updates for that target.

So the first useful mask is:

$$
M_{ij}^{(t)} = m_i^{(t)}
$$

and the active adjacency becomes:

$$
A_t = A_0 \odot M^{(t)}
$$

This means:

- halted targets do **not** get updated,
- but active targets can still read from halted source nodes.

That second point matters. We should **not** naively remove every edge touching a halted node, because active nodes may still need those halted neighbors as a memory source.

### Phase 2: learned edge gating

After target-active masking works, we add a learned edge gate:

$$
g_{ij}^{(t)} = \sigma\!\Big(\phi([h_i^{(t)}, h_j^{(t)}, e_{ij}, \Delta p_{ij}])\Big)
$$

Then:

$$
\tilde{A}_t = A_0 \odot M^{(t)} \odot \mathrm{TopKOrThreshold}(g^{(t)})
$$

Possible variants:

1. **soft gate**  
   Keep all edges but multiply message / attention logits by $g_{ij}^{(t)}$.

2. **hard threshold**  
   Remove edges with $g_{ij}^{(t)} < \epsilon$.

3. **top-k per target**  
   Keep only the strongest incoming edges for each active target node.

The order should be exactly this: first target-active masking, then learned edge pruning.

---

## 3.6 Compute and memory view

With sparse adjacency attention, the recurrent core cost is naturally edge-scaled rather than dense-$N^2$.

Ignoring constant factors, a sparse loop step is roughly:

$$
\mathrm{Cost}_{\text{loop}} \propto |E_t| \cdot d
$$

where $|E_t|$ is the number of active edges at loop $t$.

If adaptive depth prunes a fraction of target nodes and edges, then:

$$
|E_t| = \beta_t |E_0|
$$

with $\beta_t \in [0,1]$ the retained-edge ratio.

So dynamic adjacency gives a direct path to:

- lower loop-time FLOPs,
- lower activation memory,
- lower inference latency.

The training-time memory still depends on whether we keep activations from all loops or truncate / checkpoint them.

---

## 4. What we will port from OpenMythos

Below is the logic we will port, rewritten in graph form.

## 4.1 Stable injection module

```python
class StableInjection(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.log_A = nn.Parameter(torch.zeros(dim))
        self.log_dt = nn.Parameter(torch.zeros(1))
        self.B = nn.Parameter(torch.full((dim,), 0.1))

    def get_A(self) -> torch.Tensor:
        return torch.exp(-torch.exp(torch.clamp(self.log_A + self.log_dt, -20, 20)))

    def forward(self, h: torch.Tensor, e: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        a = self.get_A().view(1, -1)
        b = self.B.view(1, -1)
        return a * h + b * e + update
```

This is the key Parcae-style stabilization. We should log `get_A().max()` during training because, for a diagonal matrix, that is the spectral radius.

---

## 4.2 Loop-index embedding

```python
def add_loop_index(h: torch.Tensor, t: int, loop_dim: int, theta: float = 10_000.0):
    freq = theta ** (-torch.arange(0, loop_dim, 2, device=h.device, dtype=h.dtype) / loop_dim)
    ang = t * freq
    emb = torch.cat([ang.sin(), ang.cos()], dim=0)[:loop_dim]

    full = torch.zeros(h.size(-1), device=h.device, dtype=h.dtype)
    full[:loop_dim] = emb
    return h + full.view(1, -1)
```

For batched graphs this simply broadcasts across nodes.

---

## 4.3 Node-wise MoE FFN

```python
class NodeMoEFFN(nn.Module):
    def __init__(self, dim: int, n_experts: int, top_k: int, expert_ctor, n_shared: int = 0):
        super().__init__()
        self.router = nn.Linear(dim, n_experts, bias=False)
        self.top_k = top_k
        self.experts = nn.ModuleList([expert_ctor() for _ in range(n_experts)])
        self.shared = nn.ModuleList([expert_ctor() for _ in range(n_shared)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(self.router(x), dim=-1)
        w, idx = probs.topk(self.top_k, dim=-1)
        w = w / w.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(x)
        for slot in range(self.top_k):
            chosen = idx[:, slot]
            weight = w[:, slot:slot+1]
            for eid, expert in enumerate(self.experts):
                mask = (chosen == eid)
                if mask.any():
                    out[mask] += weight[mask] * expert(x[mask])

        for expert in self.shared:
            out = out + expert(x)

        return out
```

This is conceptually identical to the OpenMythos MoE, except routing is now per node instead of per token.

---

## 4.4 Recurrent graph core

```python
class RecurrentGraphCore(nn.Module):
    def __init__(self, block, dim, max_loops, act_threshold=0.99, loop_dim=None):
        super().__init__()
        self.block = block                     # shared sparse graph transformer block or tied stack
        self.injection = StableInjection(dim)
        self.halt_head = nn.Linear(dim, 1)
        self.norm = RMSNorm(dim)
        self.max_loops = max_loops
        self.act_threshold = act_threshold
        self.loop_dim = loop_dim or (dim // 8)

    def forward(self, h, e, base_adj, pos=None, n_loops=None, adj_policy=None):
        T = n_loops or self.max_loops

        halted = torch.zeros(h.size(0), dtype=torch.bool, device=h.device)
        cum_p = torch.zeros(h.size(0), device=h.device)
        h_out = torch.zeros_like(h)

        for t in range(T):
            h_loop = add_loop_index(h, t, self.loop_dim)
            adj_t = base_adj if adj_policy is None else adj_policy(base_adj, h, halted, t)

            update = self.block(self.norm(h_loop + e), adj_t, pos=pos)
            h_new = self.injection(h, e, update)

            p = torch.sigmoid(self.halt_head(h_new)).squeeze(-1)
            still_running = ~halted
            remainder = (1.0 - cum_p).clamp(min=0.0)
            weight = torch.where(cum_p + p >= self.act_threshold, remainder, p)

            h_out = h_out + weight.unsqueeze(-1) * h_new
            cum_p = cum_p + p * still_running.float()
            new_halted = halted | (cum_p >= self.act_threshold)

            # nodes that were already halted keep their previous state;
            # active or newly halted nodes keep h_new and then freeze there
            h = torch.where(halted.unsqueeze(-1), h, h_new)
            halted = new_halted

            if halted.all():
                break

        return h_out
```

Two notes:

1. The exact update-freeze semantics may be slightly rearranged in the final implementation.
2. If we want actual compute savings from halted nodes in sparse attention, we will likely need a specialized active-query sparse path rather than simply masking after the fact.

---

## 5. Concrete classes to add to your codebase

The cleanest way to integrate this into your existing files is:

### In `layers.py`

Add:

- `StableInjection`
- `LoopIndexEmbedding` or functional `add_loop_index(...)`
- `NodeMoEFFN`
- `NodeACTHalting`
- `AdaptiveAdjacencyPolicy`
- optionally `EdgeGateMLP`

Keep reusing:

- `RMSNorm`
- `build_gated_mlp(...)`
- `Attention`
- `Transformer`

### In `processors.py`

Add:

- `LoopedEncodeTransformDecode`
- optionally `LoopedEncodeProcessDecode` for the edge-aware `GraphNetBlock` variant

### New processor structure

```python
class LoopedEncodeTransformDecode(nn.Module):
    def __init__(..., n_prc: int, ...):
        self.nodes_encoder = ...
        self.prelude = nn.ModuleList([Transformer(...) for _ in range(n_prc)])   # P
        self.recurrent_stack = nn.ModuleList([Transformer(...) for _ in range(n_prc)])  # R, tied across loops
        self.recurrent_core = RecurrentGraphCore(block=self.recurrent_stack, ...)
        self.coda = nn.ModuleList([Transformer(...) for _ in range(n_prc)])      # C
        self.decode_module = ...
```

### Important implementation decisions

1. **Exact baseline keeps `TemporalAttention` off.**  
   It is already in your library, but it is not part of the exact OpenMythos / Parcae recipe.

2. **Exact baseline uses adjacency only.**  
   Your current `Transformer` block consumes adjacency and node positions, not edge features. That is fine for the first port.

3. **If edge features are essential, do not bolt them on carelessly.**  
   Use either:
   - edge-conditioned attention bias, or
   - the `GraphNetBlock` recurrent variant.

---

## 6. What I would implement first

### Stage A - exact stable looped graph transformer

This is the MVP.

- Use `EncodeTransformDecode` as the structural template.
- Replace the list of distinct transformer blocks with a Parcae-style symmetric split:
  - Prelude `P`: `n_prc` untied blocks,
  - recurrent core `R`: `n_prc` tied blocks looped `T`,
  - Coda `C`: `n_prc` untied blocks,
  - decoder.
- Choose `n_prc` from the remapped depth table in Section 8 rather than free-form Prelude / Coda depths.
- Add stable input injection `A * h + B * e + update`.
- Add loop-index embedding.
- Add prelude normalization on `e`.
- Start with `h_0 = e`.
- Keep adjacency static.
- Keep FFN dense (your current gated MLP).
- Keep ACT off initially.

This gets us the exact recurrent port with the same central update rule.

### Stage B - Parcae training fixes

After the MVP is stable:

- add per-graph depth sampling inside a batch,
- start with the Parcae recurrence defaults:
  - `mu_rec = 8`,
  - `mu_bwd = 4`,
  - truncated Poisson sampling, but **per graph** rather than per token / sequence,
- then sweep around that default if the task needs it,
- log state norm and recurrent residual jump.

This is the stability layer from the paper.

### Stage C - MoE recurrent FFN

Once the stable loop works:

- replace the recurrent FFN with node-wise MoE,
- keep prelude and coda dense,
- log expert utilization and load balance.

This matches the OpenMythos recurrent design.

### Stage D - node-wise ACT / halting

Only after the stable dense recurrent baseline is working:

- add node-wise halting,
- freeze halted nodes,
- log per-node exit depths,
- start with **semantic halting only** (no actual compute saving yet).

### Stage E - adaptive adjacency

After ACT semantics are correct:

1. prune updates to halted target nodes,
2. then add a learned edge gate,
3. then optionally switch to top-k edge retention per target.

This is the right order. Do not begin with learned edge deletion.

---

## 7. Adaptive-depth research: what is relevant for the graph version

The LLM literature points to a few strong design conclusions.

### 7.1 Online halting is better than one-shot early depth prediction

The strongest practical lesson is that a **shallow one-shot depth predictor is not the right first design**.

The more promising direction is **online halting**:

- the model decides after each recurrent step whether a node is done,
- halted states are frozen,
- active nodes continue refining.

That fits your graph setting better than a single pre-loop routing decision, because node difficulty can change after neighborhood information starts propagating.

### 7.2 Frozen states should remain visible

Adaptive token-depth methods repeatedly converge on the same principle:

- once a token halts, its state stops changing,
- but other active tokens can still attend to it.

The graph analogue is:

- once a node halts, stop updating that node,
- but do **not** immediately delete it as a source from every neighbor's adjacency.

This is why I recommend **target-active masking first**, not "remove every edge touching halted nodes".

### 7.3 If you want predictable compute, use a budgeted router

ACT-style halting gives flexible compute but less predictable budget.

If you want strict control over runtime, use a MoD-style budget:

- keep exactly top-k active nodes per loop,
- or exactly top-k incoming edges per active target node,
- or a fixed active-node ratio.

This is a very good later ablation, especially for inference-time VRAM and latency studies.

### 7.4 Add a compute regularizer

Without any penalty, adaptive-depth models can easily collapse to "everyone keeps thinking forever".

So once ACT is introduced, add a compute penalty such as:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{depth}} \cdot \mathbb{E}[d_i]
$$

or a KL term against an exponential prior over node exit depths.

I would start with a simple expected-depth penalty and only move to a full depth-prior formulation if needed.

---

## 8. Recommended initial hyperparameters

These are now grounded in two sources:

1. **your graph-transformer paper** for width, head count, and Gated-MLP sizing,
2. **Parcae** for the recurrent split and recurrence-control hyperparameters.

So the rule is: **keep your graph widths, keep your sparse masked attention, keep your gated MLP; import Parcae's `P/R/C` symmetry and loop schedule.**

### 8.1 Width / heads / FFN to keep from your graph transformer paper

Keep the same width ladder already used in your paper:

- `d_model in {16, 32, 48, 64, 80, 128, 152, 200, 256, 512}`
- `n_heads = 2` for `d_model <= 80`
- `n_heads = 4` for `d_model >= 128`

Keep the same Gated-MLP sizing rule:

$$
d_{\text{gated}} = 3 \, d_{\text{model}}
$$

So the default family becomes:

| `d_model` | `d_gated` | `n_heads` |
|---:|---:|---:|
| 16 | 48 | 2 |
| 32 | 96 | 2 |
| 48 | 144 | 2 |
| 64 | 192 | 2 |
| 80 | 240 | 2 |
| 128 | 384 | 4 |
| 152 | 456 | 4 |
| 200 | 600 | 4 |
| 256 | 768* | 4 |
| 512 | 1536 | 4 |

\* I would use `768 = 3 × 256` unless your original source config really intended `769`.

### 8.2 Parcae-style structural rule for depth

Instead of choosing Prelude / recurrent / Coda depth independently, enforce:

$$
n_P = n_R = n_C = n_{\text{prc}}
$$

To map your original untied transformer depths into this Parcae-style split, use:

| Original untied depth from your paper | Looped graph split |
|---:|:---|
| `L = 3` | `P/R/C = 1 / 1 / 1` |
| `L = 5` | `P/R/C = 2 / 2 / 2` |
| `L = 10` | `P/R/C = 3 / 3 / 3` |
| `L = 12` | `P/R/C = 4 / 4 / 4` |
| `L = 14` or `15` | `P/R/C = 5 / 5 / 5` |

This is the main hyperparameter change relative to the previous README.

### 8.3 Canonical model tiers to actually run

For the looped graph transformer work, I would center the experiments on these tiers:

| Tier | `d_model` | `d_gated` | `n_heads` | `P/R/C` | Use |
|:--|---:|---:|---:|:---|:--|
| Debug | 16 | 48 | 2 | `1/1/1` | kernel checks, unit tests, overfit-one-batch |
| Small | 64 | 192 | 2 | `3/3/3` | first real training runs |
| Medium | 128 | 384 | 4 | `5/5/5` | main ablation tier |
| Large | 256 | 768 | 4 | `5/5/5` | scaling-law tier |
| XLarge | 512 | 1536 | 4 | `5/5/5` | max-scale experiments |

Then use `80/240/2`, `152/456/4`, and `200/600/4` as intermediate width sweeps when fitting scaling curves.

### 8.4 Recurrence hyperparameters to copy from Parcae first

These should become the **default recurrent settings** for the first graph port:

- injection: **diagonal**
- state init: **like-init**
- initial training mean loops: `mu_rec = 8`
- backward depth: `mu_bwd = 4`
- loop-count sampling: **truncated Poisson**, but sampled **per graph**
- loop-index channels: `d_model // 8`

Then sweep:

- `mu_rec in {4, 8, 12, 16}`
- `mu_bwd in {2, 4, 6, 8}` with the constraint `mu_bwd <= mu_rec`

### 8.5 ACT

- threshold: `0.99`
- start without halting regularization
- then sweep `lambda_depth`

### 8.6 MoE

- experts: `4` or `8`
- shared experts: `1`
- top-k: `2`

### 8.7 Adaptive adjacency

- first gate = target-active only
- second gate = learned scalar per edge
- third gate = top-k incoming edges per active node

---

## 9. Exact experiments we need to run

This section is the most important operationally.

## 9.1 Baselines

We need these baselines in exactly this order:

1. **Untied fixed-depth graph transformer**  
   Your current `EncodeTransformDecode`.

2. **Shared recurrent graph transformer without stable injection**  
   Same recurrent stack tied across loops, plain residual recurrence.

3. **Shared recurrent graph transformer with stable injection**  
   Adds the Parcae-style `A, B`.

4. **+ loop-index embedding**

5. **+ prelude normalization / per-graph depth sampling**

6. **+ MoE recurrent FFN**

7. **+ node ACT halting**

8. **+ target-active adjacency pruning**

9. **+ learned edge gating**

This ablation ladder matters. Otherwise we will not know which piece gave the gain.

---

## 9.2 Stability experiments

We should reproduce the spirit of the Parcae stability analysis in the graph setting.

### Sweep

- learning rate sweep over the same order of magnitude as your current stable runs,
- fixed model size,
- fixed dataset split,
- fixed recurrence settings.

### Log the following every run

- training loss
- validation loss
- `max(A)` or `rho(A)` for the diagonal injection
- recurrent state norm:
  $$
  \|h_T\|_2
  $$
- recurrent residual jump:
  $$
  \|h_T - h_{T-1}\|_2
  $$
- gradient norm
- number of loss spikes / NaN runs

### Success criterion

The stable injected model should:

- converge under settings where the naive shared recurrent model fails or spikes,
- keep `rho(A) < 1`,
- avoid runaway state norms.

---

## 9.3 End-to-end quality experiments

At a fixed parameter budget, compare:

- untied depth baseline,
- stable looped model,
- stable looped + MoE,
- stable looped + adaptive depth.

We should compare under:

1. **parameter-matched** settings,
2. **training-FLOP-matched** settings,
3. **inference-latency-matched** settings.

For graph tasks, report the task metric you already care about, plus validation loss if available.

---

## 9.4 Training scaling laws

We should reproduce the paper's training-scaling logic as closely as the graph setting allows.

### Step 1: define training compute budget

For a graph batch, define approximate training compute as:

$$
C_{\text{train}} \approx \text{steps} \times \left(F_P + \mu_{\text{rec}} F_R + F_C\right)
$$

where:

- $F_P$ = Prelude cost,
- $F_R$ = one recurrent loop cost,
- $F_C$ = Coda cost.

In the sparse case, $F_R$ should scale with active edge count.

### Step 2: sweep mean recurrence and data budget

At fixed parameter count, train models across:

- several FLOP budgets,
- several mean recurrences `mu_rec`,
- several data budgets `D`.

For graphs, `D` can be measured as:

- number of training graphs, or
- total nodes seen, or
- total edges seen.

I would log **both graph count and total nodes seen**, then fit with whichever is more predictive.

### Step 3: fit iso-FLOP parabolas

For each training-FLOP budget, fit performance vs `mu_rec`, then extract the recurrence giving the best validation loss.

That gives the graph analogue of:

$$
\mu_{\text{rec}}^\star \propto C^{\gamma_\mu}
$$

and

$$
D^\star \propto C^{\gamma_D}
$$

### Step 4: fit a parametric law

Paper-faithful version:

$$
\hat{L}_{\text{train}}(\mu_{\text{rec}}, D)
= E + X \cdot N_{\text{eff}}(\mu_{\text{rec}})^{-x} + Y \cdot D^{-y}
$$

with

$$
N_{\text{eff}}(\mu_{\text{rec}}) \approx N_P + \mu_{\text{rec}} N_R + N_C
$$

where `N_eff` is the unrolled-depth parameter-equivalent model size, not the literal learned parameter count.

If that fit is awkward for graphs, we also try the simpler alternative:

$$
\hat{L}_{\text{train}}(\mu_{\text{rec}}, D)
= E + X \cdot \mu_{\text{rec}}^{-x} + Y \cdot D^{-y}
$$

and compare which fit extrapolates better.

---

## 9.5 Test-time scaling experiments

This is mandatory.

For every trained looped model, evaluate at:

$$
T \in \{1, 2, 4, 6, 8, 12, 16, 24\}
$$

or the closest reasonable range around the trained `mu_rec`.

Then fit:

$$
L(T) = L_\infty + Z e^{-zT}
$$

We want to answer:

1. does more loop depth improve test-time quality?
2. where does it saturate?
3. does the plateau occur near the trained `mu_rec`?
4. can we extrapolate performance at unseen loop counts?

### Unified fit

If the training fit is stable, also fit the unified law:

$$
\hat{L}_{\text{unified}}(T \mid \mu_{\text{rec}}, D)
=
E + X \cdot N_{\text{eff}}(\mu_{\text{rec}})^{-x} + Y \cdot D^{-y}
+ Z \exp\!\left(-z \cdot T / \mu_{\text{rec}}\right)
$$

This gives one model for training-time and inference-time compute scaling.

---

## 9.6 VRAM and throughput experiments

You explicitly asked for VRAM work, so this needs its own benchmark suite.

### Measure

For each configuration, record:

- peak CUDA memory allocated,
- peak CUDA memory reserved,
- step time,
- throughput in graphs/s,
- throughput in nodes/s,
- throughput in edges/s,
- inference latency.

### Sweep

Sweep over:

- loop count `T`,
- graph size `(N, E)`,
- hidden size,
- MoE on/off,
- ACT on/off,
- active-node ratio,
- retained-edge ratio.

### Comparisons to report

1. fixed-depth untied baseline  
2. shared recurrent stable baseline  
3. + MoE  
4. + ACT semantic halting  
5. + target-active adjacency pruning  
6. + learned edge pruning

### What we expect

- Stable recurrence alone should reduce parameter memory pressure versus untied deeper models.
- ACT semantics alone may not reduce compute unless we implement an active-query / active-target sparse path.
- Actual VRAM and latency wins should appear once adaptive adjacency prunes active targets / edges.

### Instrumentation

Use the standard CUDA stats around forward + backward:

```python
torch.cuda.reset_peak_memory_stats()
...
peak_alloc = torch.cuda.max_memory_allocated()
peak_reserved = torch.cuda.max_memory_reserved()
```

and log them together with graph size and loop count.

---

## 9.7 MoE experiments

For the MoE phase, log:

- validation metric,
- expert utilization histogram,
- fraction of nodes sent to each expert,
- router entropy,
- top-k stability across loops,
- load balance loss or imbalance score,
- train-time memory increase,
- inference-time latency increase.

The main question is whether MoE improves quality enough to justify the routing overhead in the graph setting.

---

## 9.8 Adaptive-depth and adaptive-adjacency diagnostics

These diagnostics are essential.

### For node depth

Log:

- mean exit depth per node,
- histogram of node exit depths,
- exit depth by graph,
- exit depth vs node degree,
- exit depth vs centrality,
- exit depth vs local loss / uncertainty,
- exit depth vs recurrent residual norm.

### For adjacency adaptation

Log:

- fraction of active target nodes per loop,
- fraction of retained edges per loop,
- retained edges by node degree bucket,
- retained edges by loop index,
- task metric vs retained-edge budget.

### The key question

Does the model actually allocate more loops / edges to hard nodes, or is it just learning a trivial structural heuristic?

That is exactly the failure mode to watch for.

---

## 10. Implementation risks and how to handle them

### 10.1 Sparse attention halting does not automatically save compute

If we simply halt a node and then overwrite its state with the previous one **after** full sparse attention is computed, the semantics are correct but the compute is not reduced.

So for real savings we likely need one of these:

1. **active-target sparse attention**  
   only compute queries for active target nodes;

2. **active-target message passing**  
   easier if we use the `GraphNetBlock` path;

3. **hard edge pruning before the sparse kernel**.

This is why I recommend semantic halting first, compute-saving adjacency second.

### 10.2 Edge features are not in the current transformer path

Your current `Transformer` path uses the adjacency mask and node positions, but not `edge_attr`.

If the task depends heavily on edge features, we should not ignore that. The two clean options are:

- add an edge-conditioned attention bias,
- or use a looped `GraphNetBlock` variant.

### 10.3 DGL sparse backend matters

Your current exact sparse attention path depends on DGL sparse. The PyG fallback path is not the same thing.

So the **exact** recurrent sparse-attention experiments should target the DGL sparse backend.

### 10.4 Adaptive edge deletion can easily become too aggressive

If we learn edge gates too early, the model can remove useful long-range routes before the recurrent dynamics are stable.

So the schedule should remain:

1. stable loop,
2. ACT,
3. target-active masking,
4. learned edge pruning.

---

## 11. The exact roadmap I recommend

### Milestone 1 - exact port
Deliver a `LoopedEncodeTransformDecode` with:

- Prelude / shared recurrent block / Coda,
- stable injection,
- loop-index embedding,
- prelude norm,
- static adjacency,
- dense FFN.

### Milestone 2 - Parcae stability suite
Add:

- per-graph depth sampling,
- truncated backprop depth,
- stability logging,
- test-time loop sweep.

### Milestone 3 - MoE recurrent FFN
Add:

- node-wise top-k experts,
- expert usage logs,
- ablation vs dense FFN.

### Milestone 4 - node ACT
Add:

- online node halting,
- frozen halted nodes,
- exit-depth logging,
- expected-depth regularization sweep.

### Milestone 5 - adaptive adjacency
Add:

- target-active adjacency masking,
- retained-edge logging,
- optional learned edge gates,
- VRAM / latency study.

### Milestone 6 - graph-specific extensions
Optionally add:

- edge-conditioned attention bias,
- looped `GraphNetBlock`,
- temporal corrector as an ablation rather than a baseline.

---

## 12. Bottom line

Yes - we can build essentially the same system in your library.

The shortest correct path is:

1. port the **Prelude / Recurrent / Coda** scaffold into your existing sparse graph transformer path,
2. use the **same stable update formula**,
3. add **loop-index embedding**,
4. verify **Parcae-style stability and test-time loop scaling**,
5. then layer on **MoE**,
6. then **online node halting**,
7. then **adaptive adjacency**.

The critical design recommendation from the adaptive-depth literature is:

> do **online halting** first, not one-shot depth prediction;  
> freeze halted nodes, but keep them visible as sources;  
> then turn that semantics into actual compute savings by pruning halted targets and finally learning edge gates.

That gives you the cleanest path to a looped graph transformer that is both faithful to the original idea and genuinely useful for graph workloads.

---

## 13. Sources consulted for this plan

### Local code you attached
- `layers (6).py`
- `processors (4).py`

### External architecture / scaling references
- OpenMythos repository
- Parcae: *Scaling Laws for Stable Looped Language Models*

### Adaptive token / node compute references that informed the design
- Mixture-of-Recursions (MoR)
- AdaPonderLM
- ANIRA / *Understanding Dynamic Compute Allocation in Recurrent Transformers*
- Inner Thinking Transformer (ITT)
- Universal Transformer / ACT
- Mixture-of-Depths (MoD)

### Graph-side analogues worth checking later
- Adaptive Message Passing
- Cooperative Graph Neural Networks
