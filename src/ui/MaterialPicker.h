#pragma once
#include <glm/glm.hpp>
#include <string>

// The body appearance controls, shared by every place that edits a body's look
// (Items panel swatch, im-touch tree swatch, Properties panel). They only
// report a pick; the caller writes it through setBodyMaterialAndMark /
// setBodyColorAndMark (core/ItemsPanelActions.h) so the colour rule and the
// dirty marking live in one place.

namespace materializr {

struct AppearancePick {
    bool materialPicked = false;
    std::string materialId;      // valid when materialPicked; "" = None
    bool colorPicked = false;
    glm::vec3 color{0.0f};       // valid when colorPicked
};

// Contents of a body swatch popup (the caller owns BeginPopup / EndPopup):
// material chips grouped by category, then the hue wheel. `width` sizes the
// wheel, and the chips wrap to it.
AppearancePick bodyAppearancePicker(const std::string& materialId,
                                    const glm::vec3& color, float width);

// A combo over the same catalogue. True when the user picked; `out` holds the
// pick ("" = None).
bool materialCombo(const char* label, const std::string& materialId, std::string& out);

} // namespace materializr
