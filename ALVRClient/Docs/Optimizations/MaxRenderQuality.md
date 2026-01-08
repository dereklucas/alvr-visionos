# Max Render Quality Override

If visionOS 26 reduces drawable allocation size, forcing a higher max render quality can lift the OS allocation ceiling.

How to try:
- Set UserDefaults `ALVRMaxRenderQuality` to a Double (for example 1.0).
- Observe the `RenderStats` log for drawable sizes and the `maxRenderQuality` startup/change logs.

Expected outcome:
- If drawable sizes increase after setting this value, the previous half-res look was likely gated by max render quality.
- If sizes stay fixed, a different limiter is in play (layout, foveation, or stream negotiation).
