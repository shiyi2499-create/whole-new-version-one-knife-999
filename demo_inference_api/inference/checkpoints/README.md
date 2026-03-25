# Checkpoint Notes

Large checkpoint binaries are not duplicated inside this API bundle.

`CHECKPOINT_MANIFEST.json` records:
- the canonical source path inside the main research repo
- the filename expected if a standalone `checkpoint_dir` is assembled for the demo
- the loading method and model class notes

For demo integration, the easiest path is:
1. copy the listed checkpoint files into one directory
2. keep the filenames from the manifest
3. call `load_all_models(checkpoint_dir)`
