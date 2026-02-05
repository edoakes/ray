# Ray Data Backpressure: Current State and Proposed Redesign

## Current Implementation

### Operator-Level Backpressure

Backpressure in Ray Data operates at the **operator level**, not per-actor. The streaming executor evaluates whether each operator can accept new work based on aggregate metrics across all actors in that operator.

Key files:
- `python/ray/data/_internal/execution/backpressure_policy/backpressure_policy.py` — base interface
- `python/ray/data/_internal/execution/streaming_executor_state.py:558-666` — policy application
- `python/ray/data/_internal/execution/resource_manager.py` — resource tracking

### Three Concurrent Policies

1. **ConcurrencyCapBackpressurePolicy**: Dynamically adjusts operator concurrency based on output queue growth using EWMA tracking.

2. **DownstreamCapacityBackpressurePolicy**: Compares queue size to downstream processing capacity; applies backpressure when `queue_size / downstream_capacity > ratio` (default 10.0).

3. **ResourceBudgetBackpressurePolicy**: Enforces resource budget limits via `OpResourceAllocator`.

### Global Resource Budget

The resource budget is computed as a **cluster-wide aggregate** (`default_cluster_autoscaler_v2.py:303-311`):

```python
def get_total_resources(self) -> ExecutionResources:
    resources = self._autoscaling_coordinator.get_allocated_resources(...)
    total = ExecutionResources.zero()
    for res in resources:
        total = total.add(ExecutionResources.from_resource_dict(res))
    return total
```

This sums object store memory across all nodes into a single global budget.

### Shuffle Exemption

Blocking/materializing operators (shuffle, all-to-all, zip) are **exempted from backpressure** (`resource_manager.py:1014-1022`):

```python
# A materializing operator like `AllToAllOperator` waits for all its input
# operator's outputs before processing data. This often forces the input
# operator to exceed its object store memory budget. To prevent deadlock, we
# disable object store memory backpressure for the input operator.
for op in eligible_ops:
    if self._resource_manager._is_blocking_materializing_op(op):
        self._op_budgets[op] = self._op_budgets[op].copy(
            object_store_memory=float("inf")
        )
```

---

## Issues with Current Approach

### 1. Global Budget Ignores Per-Node Pressure

The object store is **per-node**, not global. Each node can spill independently when its local store fills. A global budget fails to capture skewed memory distribution:

- Node A: 95% object store utilization (about to spill)
- Node B: 5% object store utilization (nearly empty)
- Global view: 50% utilization → no backpressure triggered

Data skew is common with operations like `GroupBy` or `Repartition` that concentrate outputs by key.

### 2. Underutilization from Conservative Throttling

When global backpressure triggers due to one hot node, **all nodes are throttled**:

- Hot node at 95% → global backpressure kicks in
- Cold nodes at 10% → sit idle waiting for hot node to drain
- Cluster throughput drops despite available capacity

### 3. Shuffle Exemption Relies on Spilling

Disabling backpressure for shuffles means memory is unbounded during shuffle operations. The system relies on object store spilling as the safety valve, which:

- Degrades performance significantly
- Can still OOM if spilling can't keep up
- Provides no flow control signal to upstream

### 4. Conflates Input Consumption with Output Production

The current exemption logic assumes: "shuffle can't consume until everything is ready." This is incorrect. Shuffle operators **can consume inputs incrementally** (building hash tables), they just can't **produce outputs** until all inputs are consumed.

Consuming inputs into internal state **is progress** — the input queue drains as the operator processes. Backpressure based on input queue depth would not deadlock.

---

## Proposed Redesign: Per-Actor Bounded Queues

### Core Model

Replace operator-level global budgets with **per-actor bounded queues**:

```
upstream actor ──► [bounded queue] ──► downstream actor
```

- Each actor has a fixed-length input queue
- Queue length can be adjusted based on input/output sizes
- When queue is full, upstream backpressures
- Output is considered complete only when consumed by downstream (credit-based flow control)

### Advantages

1. **Implicit per-node pressure handling**: If a node's object store is stressed, local actors slow down (object puts get slower), their queues stop draining, and backpressure propagates upstream automatically.

2. **Natural load balancing**: Fast actors drain queues and pull more work. Slow actors fill up and self-throttle. No central coordination needed.

3. **Bounded memory per actor**: `queue_length × max_output_size` gives predictable memory bounds. Sum across actors on a node ≈ node memory bound.

4. **Simpler reasoning**: Matches bounded channel semantics from streaming systems (Flink, Go channels, Rust mpsc).

### Credit-Based Flow Control

An output should only be considered "complete" (freeing a queue slot) once the downstream actor has consumed it:

```
Producer completes → slot still held → downstream task completes → slot freed
```

This prevents producers from racing ahead while unconsumed outputs accumulate in the object store.

### Map Operators

Map operators are the simplest case and illustrate the core model well. Consider a pipeline with two map stages:

```
┌────────────┐    ┌─────────┐    ┌────────────┐    ┌─────────┐    ┌────────────┐
│  Input     │───►│ Queue A │───►│  Map Op 1  │───►│ Queue B │───►│  Map Op 2  │───► Output
│  Source    │    │ (cap=3) │    │  (actors)  │    │ (cap=3) │    │  (actors)  │
└────────────┘    └─────────┘    └────────────┘    └─────────┘    └────────────┘
```

**Normal flow:**
1. Input source produces blocks into Queue A
2. Map Op 1 actors pull from Queue A, process, push to Queue B
3. Map Op 2 actors pull from Queue B, process, emit output

**Backpressure scenario:**

If Map Op 2 is slower than Map Op 1:

```
Time T1:  Queue A: [■ ■ ·]    Queue B: [■ ■ ■]  ← Queue B full
          Map Op 1 blocks on push to Queue B

Time T2:  Queue A: [■ ■ ■]  ← Queue A fills    Queue B: [■ ■ ■]
          Input source blocks on push to Queue A

Time T3:  Map Op 2 completes one task, frees slot in Queue B
          Queue A: [■ ■ ■]    Queue B: [■ ■ ·]
          Map Op 1 unblocks, pushes to Queue B, pulls from Queue A

Time T4:  Queue A: [■ ■ ·]    Queue B: [■ ■ ■]
          Input source unblocks, pushes to Queue A
```

**Per-actor queues with multiple actors:**

With N actors per operator, each actor has its own input queue:

```
                    ┌─────────┐    ┌──────────────┐
               ┌───►│Queue 1a │───►│ Map1 Actor 1 │───┐
               │    └─────────┘    └──────────────┘   │
┌────────┐     │    ┌─────────┐    ┌──────────────┐   │    ┌─────────┐    ┌──────────────┐
│ Input  │─────┼───►│Queue 1b │───►│ Map1 Actor 2 │───┼───►│Queue 2a │───►│ Map2 Actor 1 │──► Out
│ Source │     │    └─────────┘    └──────────────┘   │    └─────────┘    └──────────────┘
└────────┘     │    ┌─────────┐    ┌──────────────┐   │    ┌─────────┐    ┌──────────────┐
               └───►│Queue 1c │───►│ Map1 Actor 3 │───┴───►│Queue 2b │───►│ Map2 Actor 2 │──► Out
                    └─────────┘    └──────────────┘        └─────────┘    └──────────────┘
```

- Input source distributes work across Map1 actor queues (round-robin or work-stealing)
- Each Map1 actor pushes to any available Map2 actor queue
- If Map2 Actor 1's queue is full but Actor 2's has space, Map1 can still make progress
- Natural load balancing: fast actors drain queues, slow actors self-throttle

**Credit-based completion:**

A Map1 actor's output slot is freed only when Map2 finishes processing it:

```
1. Map1 Actor produces output O1 → pushes to Queue 2a (slot occupied)
2. Map2 Actor 1 pulls O1 from Queue 2a → processes it
3. Map2 Actor 1 completes processing O1 → signals completion
4. Map1 Actor's slot is freed → can accept new input
```

This ensures that unconsumed outputs don't accumulate unboundedly in the object store.

### Streaming Aggregations

Input queue → internal state accumulation → output queue. Internal state is operator-specific (spill if too large), orthogonal to queue-based backpressure.

### Group-by

Same as streaming aggregation, just with keyed state. Equivalent to "build side only" of a join.

### Joins

Decompose into two streaming phases:

```
build upstream ──► [build input queue] ──► ┌─────────────┐
                                           │ join actor  │
                                           │ (hash table)│ ──► [output queue] ──► downstream
probe upstream ──► [probe input queue] ──► └─────────────┘
```

- Build phase: Input queue → hash table accumulation (no outputs yet)
- Probe phase: Input queue → lookup + emit → output queue
- Each phase independently fits the queue model
- Internal hash table state handled separately (spill if needed)

**Shuffle / Repartition**: Per-edge queues for all-to-all communication:

```
A1 ──[queue]──► B1
A1 ──[queue]──► B2    (if full, A1 blocks sending to B2 only)
A1 ──[queue]──► B3
```

Each (producer, consumer) pair has an independent bounded queue. Producers handle partial blockage by buffering or spilling records destined for blocked partitions.

### Shuffle Input Queues

Shuffles should have bounded input queues like any other operator:

```
upstream ──► [bounded input queue] ──► shuffle operator (consuming, building state)
                                              │
                                         internal state grows
                                              │
                                              ▼ (after all inputs consumed)
                                       [bounded output queue] ──► downstream
```

- Shuffle actively consumes from input queue, building internal state
- Queue drains as shuffle consumes
- If shuffle is slower than upstream, queue fills, upstream backpressures
- No deadlock because shuffle is making progress (consuming inputs)

The bounded queue limits **in-flight unprocessed data**, not total inputs the operator will eventually see. Internal state growth is the operator's responsibility (spill if needed), separate from dataflow backpressure.

---

## Summary

| Aspect | Current | Proposed |
|--------|---------|----------|
| Backpressure granularity | Operator-level | Per-actor / per-edge |
| Resource budget | Global cluster sum | Implicit per-node via queue bounds |
| Shuffle handling | Exempted, rely on spilling | Bounded input queues, active consumption |
| Flow control | Output queue size | Credit-based (consumed = complete) |
| Node pressure | Not tracked | Implicit via local actor slowdown |
| Memory bounds | Global budget fraction | Per-actor queue × output size |
