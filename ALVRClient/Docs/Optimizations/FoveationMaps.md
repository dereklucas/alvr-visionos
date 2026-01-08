# Rasterization Rate Map Usage

When foveation is enabled, applying the drawable's rasterization rate map to the render pass descriptor preserves the intended resolution distribution.

How to try:
- Toggle UserDefaults `ALVREnableFoveationMaps` between true and false.
- Check `RenderStats` `vrr=` and ratio values, and compare perceived sharpness.

Expected outcome:
- If disabling maps reduces clarity or changes the resolution ratio, missing map application was a factor.
- If no change, focus on layout or render quality instead.
