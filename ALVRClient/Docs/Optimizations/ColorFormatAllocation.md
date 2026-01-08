# Color Format and Allocation Size

The compositor may allocate smaller drawables for higher-bandwidth formats. Forcing a lighter format can help validate whether allocation size is constrained by format choice.

How to try:
- Set UserDefaults `ALVRForceColorFormat` to `bgra8UnormSrgb` or `rgba16Float`.
- Compare `RenderStats` drawable sizes and the configuration log for the chosen color format.

Expected outcome:
- If switching to `bgra8UnormSrgb` yields larger drawables, format bandwidth was likely limiting allocation size.
- If there is no change, focus on render quality or layout.
