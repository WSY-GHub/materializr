#include "MaterialPicker.h"

#include "../core/Material.h"
#include "../i18n.h"
#include "LengthField.h"
#include "UiTheme.h"

#include <imgui.h>
#include <cstdio>

namespace materializr {

namespace {

constexpr MaterialCategory kCategories[] = {
    MaterialCategory::Printing, MaterialCategory::Metal, MaterialCategory::Other,
};

void densityTooltip(const Material& m) {
    if (!ImGui::IsItemHovered()) return;
    char d[32];
    std::snprintf(d, sizeof d, "%.2f g/cm\xC2\xB3", m.density);
    ImGui::SetTooltip("%s", trFormat("Density: %s", std::string(d)).c_str());
}

// One chip. Wraps onto a new line when it would run past `right`.
bool chip(const char* label, bool selected, float left, float right, bool& first) {
    const ImGuiStyle& st = ImGui::GetStyle();
    const float w = ImGui::CalcTextSize(label, nullptr, true).x + 2.0f * st.FramePadding.x;
    if (!first) {
        const float next = ImGui::GetItemRectMax().x + st.ItemSpacing.x;
        if (next + w <= right) ImGui::SameLine();
        else ImGui::SetCursorScreenPos(ImVec2(left, ImGui::GetCursorScreenPos().y));
    }
    first = false;
    if (selected) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
    }
    const bool hit = ImGui::Button(label);
    if (selected) ImGui::PopStyleColor(2);
    return hit;
}

// A collapsible group header, closed by default so the list stays short on a
// small screen. Each time the popup or combo opens, only the group holding the
// current material starts open; a closed header still names that material.
bool categoryHeader(MaterialCategory cat, const Material* cur) {
    const bool holdsCurrent = cur && cur->category == cat;
    if (ImGui::IsWindowAppearing()) ImGui::SetNextItemOpen(holdsCurrent);
    char label[160];
    if (holdsCurrent)
        std::snprintf(label, sizeof label, "%s  -  %s###cat%d", tr(materialCategoryName(cat)),
                      tr(cur->name), static_cast<int>(cat));
    else
        std::snprintf(label, sizeof label, "%s###cat%d", tr(materialCategoryName(cat)),
                      static_cast<int>(cat));
    return ImGui::CollapsingHeader(label);
}

} // namespace

AppearancePick bodyAppearancePicker(const std::string& materialId,
                                    const glm::vec3& color, float width) {
    AppearancePick pick;
    const float left = ImGui::GetCursorScreenPos().x;
    const float right = left + width;

    ImGui::TextColored(accentText(), "%s", tr("Material"));
    bool first = true;
    ImGui::PushID("materials");
    if (chip(tr("None"), materialId.empty(), left, right, first)) {
        pick.materialPicked = true;
        pick.materialId.clear();
    }
    const Material* cur = findMaterial(materialId);
    for (MaterialCategory cat : kCategories) {
        if (!categoryHeader(cat, cur)) continue;
        first = true;
        for (const Material& m : materialCatalogue()) {
            if (m.category != cat) continue;
            ImGui::PushID(m.id);
            if (chip(tr(m.name), materialId == m.id, left, right, first)) {
                pick.materialPicked = true;
                pick.materialId = m.id;
            }
            densityTooltip(m);
            ImGui::PopID();
        }
    }
    ImGui::PopID();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextColored(accentText(), "%s", tr("Colour"));
    glm::vec3 c = color;
    ImGui::SetNextItemWidth(width);
    if (ImGui::ColorPicker3("##pick", &c.x,
                            ImGuiColorEditFlags_NoSidePreview |
                            ImGuiColorEditFlags_PickerHueWheel)) {
        pick.colorPicked = true;
        pick.color = c;
    }
    return pick;
}

bool materialCombo(const char* label, const std::string& materialId, std::string& out) {
    const Material* cur = findMaterial(materialId);
    // An id this build does not know (a newer file) still shows as itself.
    const std::string preview = cur ? tr(cur->name)
                              : materialId.empty() ? tr("None") : materialId;
    bool picked = false;
    if (ImGui::BeginCombo(label, preview.c_str(), ImGuiComboFlags_HeightLargest)) {
        if (ImGui::Selectable(tr("None"), materialId.empty())) {
            out.clear();
            picked = true;
        }
        for (MaterialCategory cat : kCategories) {
            if (!categoryHeader(cat, cur)) continue;
            for (const Material& m : materialCatalogue()) {
                if (m.category != cat) continue;
                ImGui::PushID(m.id);
                if (ImGui::Selectable(tr(m.name), materialId == m.id)) {
                    out = m.id;
                    picked = true;
                }
                densityTooltip(m);
                ImGui::PopID();
            }
        }
        ImGui::EndCombo();
    }
    return picked;
}

} // namespace materializr
