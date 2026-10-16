#pragma once
#include <glm/glm.hpp>
#include <string>
#include <vector>

// What a body is made of: a density for its weight and a finish for how it
// draws. A body stores only the material's id (BodyEntry::material, "" = none),
// so the catalogue is free to retune a density or a finish without touching a
// saved project. Ids are persisted - never rename one, only add.
//
// Header-light on purpose (glm + std only): Document, ProjectIO and the
// renderer all reach this, and none of them should pull OCCT or ImGui
// through it.

namespace materializr {

// Order is the order the picker shows the groups in.
enum class MaterialCategory { Printing, Metal, Other };

// How the renderer draws the material beyond its colour and roughness.
enum class MaterialLook {
    Plain,     // plastics, resin, composites
    Metal,     // reflects a studio environment tinted by its colour
    Wood,      // solid timber: strong procedural grain under a varnish
    Plywood,   // a veneer face: softer, lower-contrast grain under a varnish
};

struct Material {
    const char* id;          // persisted in .mzr - never rename
    const char* name;        // shown; abbreviations (PLA, PETG) are not translated
    MaterialCategory category;
    double density;          // g/cm^3, for the mass readout
    glm::vec3 color;         // the colour the material brings, see keepsColor
    // Filament and sheet plastics come in every colour, so picking PLA leaves
    // the body's own colour alone. A metal or a wood IS its colour, so picking
    // it repaints the body (which can still be re-tinted afterwards).
    bool keepsColor;
    float roughness;         // 0 = mirror-smooth, 1 = fully matte
    MaterialLook look;
    // Printed on an FDM machine: the mass readout asks for infill and walls
    // instead of assuming a solid part.
    bool fdm;
};

// Every material the picker offers, grouped by category in display order.
const std::vector<Material>& materialCatalogue();

// nullptr for "" (no material) and for an id this build does not know - a file
// from a newer version keeps its id on the body and simply draws as "none".
const Material* findMaterial(const std::string& id);

const char* materialCategoryName(MaterialCategory c); // English; tr() at the call site

// How a body draws: the inputs to ShapeRenderer's mesh shader. The default is
// the look every body had before materials existed, so "no material" changes
// nothing on screen.
struct SurfaceFinish {
    float specStrength = 0.5f;
    float shininess = 32.0f;
    float metallic = 0.0f;   // 0 = dielectric, 1 = metal
    float roughness = 0.45f; // blurs a metal's reflection
    float clearcoat = 0.0f;  // varnish: a white glossy layer over the base
    float grain = 0.0f;      // wood-grain contrast, 0 = none
};
SurfaceFinish surfaceFinish(const Material* m);

// Grams of a solid of `volumeMm3` in material `m`.
inline double massGrams(double volumeMm3, const Material& m) {
    return volumeMm3 / 1000.0 * m.density;   // mm^3 -> cm^3
}

// How a slicer fills an FDM part. The defaults are what Orca, Bambu Studio
// and PrusaSlicer ship with.
struct PrintSettings {
    int infillPercent = 15;
    int walls = 2;
};
constexpr double kPrintLineWidthMm = 0.45;   // a 0.4 mm nozzle's wall line

// The weight a slicer would report, near enough: a shell of `walls` lines is
// solid all over the surface (walls on the sides, the same depth of skin top
// and bottom), and the interior is filled at the infill percentage. The shell
// is surface area x thickness, capped at the whole volume, so it overcounts
// on ribs thinner than two shells - there the part is solid anyway. Supports,
// brims and purge are not part of the part and are not counted.
inline double printedMassGrams(double volumeMm3, double areaMm2, const Material& m,
                               const PrintSettings& p) {
    const double shellMm = p.walls * kPrintLineWidthMm;
    double shell = areaMm2 * shellMm;
    if (shell > volumeMm3) shell = volumeMm3;
    const double fill = p.infillPercent / 100.0;
    return massGrams(shell + fill * (volumeMm3 - shell), m);
}

} // namespace materializr
