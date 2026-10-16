#pragma once

namespace materializr {

// The photographed studio metals reflect: a JPEG atlas of the panorama
// pre-blurred for six roughness levels, embedded so no platform has to
// package an asset file. Layout and encoding are documented in
// tools/bake_environment.py, which generates EnvironmentImage.cpp.
extern const unsigned char kEnvironmentJpeg[];
extern const unsigned long kEnvironmentJpegSize;

} // namespace materializr
