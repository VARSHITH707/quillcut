# Podcast (delegated)

Clips come from OpenShorts (`data/repos/openshorts`, API on port 8000): face-tracked
crops, speaker stacking, word captions, local Qwen moment picking. Quillcut submits the video
with `clip_min_seconds=30`, files the results in this format's `projects/`, and publishes to
`outputs\<date>\podcast\` when a rights record exists. Can become native later without
touching shared code.
