# Reflection environment

The studio that metal and varnished wood reflect in the viewport
(`src/viewport/EnvironmentImage.cpp`) is baked from:

- **Studio Small 08** by Sergej Majboroda, Poly Haven -
  https://polyhaven.com/a/studio_small_08 (2k `.hdr`)
- Licence: CC0 1.0 (public domain) - no attribution required; credited anyway.

To rebake (or swap in another panorama), download the `.hdr` and run
`tools/bake_environment.py bake <file>.hdr` (needs numpy and Pillow).
