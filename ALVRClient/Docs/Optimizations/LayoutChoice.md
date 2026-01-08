# Layered vs Dedicated Layout

The compositor layout choice can affect allocation behavior and foveation support. Layered is generally preferred when available.

How to try:
- Set UserDefaults `ALVRForceLayout` to `layered` or `dedicated`.
- Compare `RenderStats` drawable sizes and the configuration log for the selected layout.

Expected outcome:
- If one layout yields larger drawables, layout selection is a contributing factor.
- If sizes are identical, layout is not the bottleneck.
