#include "Material.h"

#include <cmath>

namespace materializr {

const std::vector<Material>& materialCatalogue()
{
    using C = MaterialCategory;
    using L = MaterialLook;
    // Densities are typical datasheet values for the bulk material. A spool's
    // label beats this table for one brand (fillers and blends move it a few
    // percent). An FDM print is not solid, so its readout estimates from
    // infill and walls instead (printedMassGrams).
    static const std::vector<Material> kCatalogue = {
        //  id            name               cat         g/cm3  colour                  keeps  rough  look        fdm
        { "pla",       "PLA",              C::Printing, 1.24, {0.80f, 0.80f, 0.82f}, true,  0.45f, L::Plain,   true  },
        { "petg",      "PETG",             C::Printing, 1.27, {0.80f, 0.80f, 0.82f}, true,  0.25f, L::Plain,   true  },
        { "abs",       "ABS",              C::Printing, 1.04, {0.80f, 0.80f, 0.82f}, true,  0.60f, L::Plain,   true  },
        { "asa",       "ASA",              C::Printing, 1.07, {0.80f, 0.80f, 0.82f}, true,  0.70f, L::Plain,   true  },
        { "tpu",       "TPU",              C::Printing, 1.21, {0.80f, 0.80f, 0.82f}, true,  0.75f, L::Plain,   true  },
        { "nylon",     "Nylon (PA)",       C::Printing, 1.14, {0.80f, 0.80f, 0.82f}, true,  0.80f, L::Plain,   true  },
        { "pc",        "PC",               C::Printing, 1.20, {0.80f, 0.80f, 0.82f}, true,  0.25f, L::Plain,   true  },
        { "hips",      "HIPS",             C::Printing, 1.04, {0.80f, 0.80f, 0.82f}, true,  0.60f, L::Plain,   true  },
        { "pp",        "PP",               C::Printing, 0.90, {0.80f, 0.80f, 0.82f}, true,  0.55f, L::Plain,   true  },
        { "pla_cf",    "PLA-CF",           C::Printing, 1.29, {0.18f, 0.18f, 0.19f}, false, 0.85f, L::Plain,   true  },
        { "pa_cf",     "PA-CF",            C::Printing, 1.20, {0.16f, 0.16f, 0.17f}, false, 0.85f, L::Plain,   true  },
        { "peek",      "PEEK",             C::Printing, 1.30, {0.76f, 0.66f, 0.50f}, false, 0.55f, L::Plain,   true  },
        // Resin prints solid (no infill pattern), so it weighs as a solid.
        { "resin",     "Resin (SLA)",      C::Printing, 1.18, {0.80f, 0.80f, 0.82f}, true,  0.30f, L::Plain,   false },

        { "aluminum",  "Aluminium",        C::Metal,    2.70, {0.80f, 0.81f, 0.83f}, false, 0.45f, L::Metal,   false },
        { "steel",     "Steel",            C::Metal,    7.85, {0.56f, 0.57f, 0.58f}, false, 0.35f, L::Metal,   false },
        { "stainless", "Stainless steel",  C::Metal,    8.00, {0.70f, 0.70f, 0.72f}, false, 0.25f, L::Metal,   false },
        { "brass",     "Brass",            C::Metal,    8.50, {0.80f, 0.66f, 0.35f}, false, 0.30f, L::Metal,   false },
        { "copper",    "Copper",           C::Metal,    8.96, {0.85f, 0.50f, 0.32f}, false, 0.30f, L::Metal,   false },
        { "titanium",  "Titanium",         C::Metal,    4.43, {0.62f, 0.60f, 0.57f}, false, 0.40f, L::Metal,   false },

        // Woods: roughness is the finish's (satin oil, not gloss varnish), the
        // grain carries the rest.
        { "plywood",   "Plywood",          C::Other,    0.68, {0.84f, 0.70f, 0.50f}, false, 0.65f, L::Plywood, false },
        { "hardwood",  "Hardwood",         C::Other,    0.75, {0.62f, 0.40f, 0.23f}, false, 0.55f, L::Wood,    false },
        { "acrylic",   "Acrylic",          C::Other,    1.18, {0.80f, 0.80f, 0.82f}, true,  0.10f, L::Plain,   false },
        { "pom",       "POM (Delrin)",     C::Other,    1.41, {0.92f, 0.92f, 0.90f}, true,  0.45f, L::Plain,   false },
        { "carbon",    "Carbon fibre",     C::Other,    1.55, {0.12f, 0.12f, 0.13f}, false, 0.30f, L::Plain,   false },
        { "foam",      "Foam board",       C::Other,    0.04, {0.90f, 0.90f, 0.88f}, true,  0.95f, L::Plain,   false },
    };
    return kCatalogue;
}

const Material* findMaterial(const std::string& id)
{
    if (id.empty()) return nullptr;
    for (const Material& m : materialCatalogue())
        if (id == m.id) return &m;
    return nullptr;
}

const char* materialCategoryName(MaterialCategory c)
{
    switch (c) {
        case MaterialCategory::Printing: return "3D printing";
        case MaterialCategory::Metal:    return "Metal";
        case MaterialCategory::Other:    return "Other";
    }
    return "Other";
}

SurfaceFinish surfaceFinish(const Material* m)
{
    SurfaceFinish f;
    if (!m) return f;
    // Roughness widens the highlight and dims it: the light a smooth surface
    // focuses into a small hot spot, a matte one smears across the face. At
    // roughness ~0.45 this lands on the default look (32 / 0.5), so PLA reads
    // like a body with no material at all.
    const float smooth = 1.0f - m->roughness;
    f.shininess = std::pow(2.0f, 1.0f + 7.0f * smooth);
    f.specStrength = 0.1f + 0.75f * smooth;
    f.roughness = m->roughness;
    switch (m->look) {
        case MaterialLook::Plain:
            break;
        case MaterialLook::Metal:
            // A polished metal's highlight is far hotter and tighter than any
            // plastic's; most of its look is the reflected environment.
            f.metallic = 1.0f;
            f.shininess = std::pow(2.0f, 2.0f + 8.0f * smooth);
            f.specStrength = 0.4f + 1.2f * smooth;
            break;
        case MaterialLook::Wood:
            f.grain = 1.0f;
            f.clearcoat = 0.25f;
            break;
        case MaterialLook::Plywood:
            f.grain = 0.6f;
            f.clearcoat = 0.15f;
            break;
    }
    return f;
}

} // namespace materializr
