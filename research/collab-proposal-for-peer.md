## A proposal for the Burgers' QNN project — joint paper angle

Quick context on where this came from: I dug into the "Quantum-Train" literature your architecture is based on (Liu et al. 2024, arXiv:2402.16465 — the paper that generates classical NN weights from a quantum circuit's measurements). Two things stood out that change what our paper could claim.

### What I found
1. The paper's argument for *why quantum is necessary* (not just convenient) is a real complexity-theory result — but the authors themselves admit it has a hole: for circuits that are 1D or "geometrically local," a classical tensor network (matrix product state) can do the same compression job just as well. Their proof only holds outside that case.
2. Separately, there's other work (Cerezo et al. 2023) showing that circuit designs specifically chosen to *avoid* barren plateaus — shallow, small-angle-initialized, the kind of ansatz that's actually usable in practice — often turn out to be classically simulable anyway, for a related but different reason.

Put together: nobody has actually checked whether a QT-style weight generator for a real PDE task (not toy classification) falls into one of these "actually just classical" categories, or whether it's genuinely doing something a classical generator can't. That question is currently unasked and unanswered for exactly the kind of architecture you're building.

### What I'm proposing
Not a separate paper — a joint one, split by what each of us is best positioned to do:
- **Your side:** get the QT-PINN actually trained and working on 2D Burgers'. You already own this.
- **My side:** build a matched-parameter-count classical baseline (a plain low-rank random projection, and a tensor-network/MPS generator), classify our circuit design against the "is this classically simulable" criteria first, predict what should happen, then run the head-to-head comparison — same PDE, same PINN backbone, same training budget.

The paper becomes: "here's the architecture and it trains (your part), and here's a rigorous answer to the question every reviewer will ask — does the quantum part actually matter (my part)." Either outcome is a real result: if quantum wins, that's a genuinely notable finding for a real task, not a toy one. If it doesn't, we get to honestly reframe the contribution as "quantum-inspired compression," which is still a legitimate and useful thing to publish — and better to know that now than after a reviewer points it out.

### Three concrete things I need from you to make the comparison fair
1. **Confirm the readout scheme.** Your design (per the Gemini chat) reads out per-qubit ⟨Z_i⟩ (N numbers). The actual Quantum-Train paper reads out the full 2^N basis-state distribution. These aren't the same thing — worth deciding which one we're testing, or testing both.
2. **Circuit connectivity.** For the "classically simulable or not" classification to be interesting, we want an ansatz that's *not* 1D/geometrically-local (i.e. keep the circular/non-adjacent entangling gates, don't simplify to nearest-neighbor-only) — otherwise the answer is trivially "yes, classically replicable" before we even run anything.
3. **Repo structure.** Suggest the joint experiment code lives in your repo (`qnn-burgers-eq` or whatever it ends up called) — happy to open a PR/branch with the classical-baseline + comparison code once your training pipeline is far enough along, or we build in parallel and merge.

Rough timeline on my end: ~3.5 weeks total (I've fracture-checked this and have a phased plan), most of it front-loaded on a small warm-up exercise plus building the classical baselines — shouldn't block your training work at all.
