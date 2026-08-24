# What you’re seeing (vortex merger)

**Color:** deep blue = little/no rotation, yellow = strong vorticity (spinning fluid).
Orange dots = tracer particles carried by the flow; grey lines = streamlines.

## Physics in 4 beats

1. **t ≈ 0–4 — Four co-rotating vortices**  
   Same-sign Gaussian vortices sit in the four quadrants. Because they all spin the *same* way, each one’s velocity field pushes on the others (unlike Taylor–Green, where opposite signs cancel).

2. **t ≈ 4–10 — Orbiting & stretching**  
   Mutual induction makes them circle around their common center and pull inward. Cores deform and filaments stretch.

3. **t ≈ 10–18 — Merger**  
   The four cores coalesce into **one** large central vortex (2D inverse energy cascade / vortex merger). This is the visually important part — the new gif spends most of its time here.

4. **t ≈ 18–40 — Single decaying swirl**  
   Topology is done: one blob remains and viscosity slowly weakens it (yellow fades toward blue). Nothing new merges.

## Columns

| DNS | Classical PINN | Quantum-trained PINN |
|-----|----------------|----------------------|
| Spectral Navier–Stokes reference | Wide RFF MLP fit to DNS | Smaller deployed MLP from QT |

Left panel is the “truth”; the other two are neural predictions of the same field.
