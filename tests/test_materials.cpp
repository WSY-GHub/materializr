// Body materials: the catalogue, the colour rule a pick applies, the mass
// arithmetic, and the .mzr round trip.
//
// A body stores only a material id (BodyEntry::material), so what has to hold
// is that the id survives save/load - including an id this build does not
// know, which a file from a newer version can carry - and that a body with no
// material stays exactly as it was before materials existed.

#include "core/Document.h"
#include "core/ItemsPanelActions.h"
#include "core/Material.h"
#include "io/ProjectIO.h"

#include <gtest/gtest.h>
#include <BRepPrimAPI_MakeBox.hxx>
#include <cstdio>
#include <set>
#include <string>
#include <vector>
#include "test_tmp_path.h"

using namespace materializr;

TEST(Materials, CatalogueIdsAreUniqueAndDensitiesReal) {
    std::set<std::string> ids;
    for (const Material& m : materialCatalogue()) {
        EXPECT_TRUE(ids.insert(m.id).second) << "duplicate material id " << m.id;
        EXPECT_GT(m.density, 0.0) << m.id;
        // Ids are written into a whitespace-separated line.
        EXPECT_EQ(std::string(m.id).find_first_of(" \t\n"), std::string::npos) << m.id;
        EXPECT_EQ(findMaterial(m.id), &m);
    }
}

TEST(Materials, CommonPrintingPlasticsArePresent) {
    for (const char* id : {"pla", "petg", "abs", "asa", "tpu", "nylon"}) {
        const Material* m = findMaterial(id);
        ASSERT_NE(m, nullptr) << id;
        EXPECT_EQ(m->category, MaterialCategory::Printing) << id;
    }
}

TEST(Materials, NoneAndUnknownFindNothing) {
    EXPECT_EQ(findMaterial(""), nullptr);
    EXPECT_EQ(findMaterial("unobtainium"), nullptr);
}

// No material must draw exactly as every body drew before materials existed.
TEST(Materials, NoMaterialKeepsTheOldLook) {
    const SurfaceFinish f = surfaceFinish(nullptr);
    EXPECT_FLOAT_EQ(f.specStrength, 0.5f);
    EXPECT_FLOAT_EQ(f.shininess, 32.0f);
    EXPECT_FLOAT_EQ(f.metallic, 0.0f);
    EXPECT_FLOAT_EQ(f.clearcoat, 0.0f);
    EXPECT_FLOAT_EQ(f.grain, 0.0f);
    EXPECT_GT(surfaceFinish(findMaterial("steel")).metallic, 0.5f);
    EXPECT_GT(surfaceFinish(findMaterial("hardwood")).grain, 0.0f);
    EXPECT_GT(surfaceFinish(findMaterial("plywood")).clearcoat, 0.0f);
    // A plastic draws no grain, no varnish, no metal.
    const SurfaceFinish pla = surfaceFinish(findMaterial("pla"));
    EXPECT_FLOAT_EQ(pla.grain + pla.clearcoat + pla.metallic, 0.0f);
}

// FDM plastics ask for infill; resin, metal and wood weigh as solids.
TEST(Materials, OnlyFdmMaterialsTakeInfill) {
    for (const char* id : {"pla", "petg", "abs", "asa", "tpu", "nylon", "pla_cf"})
        EXPECT_TRUE(findMaterial(id)->fdm) << id;
    for (const char* id : {"resin", "steel", "plywood", "acrylic"})
        EXPECT_FALSE(findMaterial(id)->fdm) << id;
}

// The slicer-style estimate: solid shell over the surface, infill inside.
TEST(Materials, PrintedMassCountsShellThenInfill) {
    const Material& pla = *findMaterial("pla");
    // A 40 mm cube: 64000 mm^3, 9600 mm^2. Two 0.45 mm walls = a 0.9 mm
    // shell, 8640 mm^3 solid; the rest at 15 %.
    const double v = 64000.0, a = 9600.0;
    const PrintSettings def;
    const double shell = a * 2 * kPrintLineWidthMm;
    EXPECT_NEAR(printedMassGrams(v, a, pla, def),
                massGrams(shell + 0.15 * (v - shell), pla), 1e-9);
    // 100 % infill is the solid part; 0 % is the shell alone.
    EXPECT_NEAR(printedMassGrams(v, a, pla, {100, 2}), massGrams(v, pla), 1e-9);
    EXPECT_NEAR(printedMassGrams(v, a, pla, {0, 2}), massGrams(shell, pla), 1e-9);
    // More walls weigh more, and a shell can never outweigh the solid.
    EXPECT_GT(printedMassGrams(v, a, pla, {15, 4}), printedMassGrams(v, a, pla, def));
    EXPECT_NEAR(printedMassGrams(v, a, pla, {0, 20}), massGrams(v, pla), 1e-9);
    // Lighter than solid for any partial infill.
    EXPECT_LT(printedMassGrams(v, a, pla, def), massGrams(v, pla));
}

TEST(Materials, MassIsVolumeTimesDensity) {
    const Material* pla = findMaterial("pla");
    ASSERT_NE(pla, nullptr);
    // A 10 mm cube is 1 cm^3.
    EXPECT_NEAR(massGrams(1000.0, *pla), pla->density, 1e-12);
    const Material* steel = findMaterial("steel");
    ASSERT_NE(steel, nullptr);
    EXPECT_NEAR(massGrams(1.0e6, *steel), 1000.0 * steel->density, 1e-9);
}

// A plastic keeps the body's colour (filament comes in every colour); a metal
// brings its own. Either way the body is marked so its mesh re-bakes.
TEST(Materials, PickAppliesTheColourRule) {
    Document doc;
    const int id = doc.addBody(BRepPrimAPI_MakeBox(10, 10, 10).Shape(), "b");
    const glm::vec3 red(0.9f, 0.1f, 0.1f);
    doc.setBodyColor(id, red);
    std::vector<int> marked;
    auto mark = [&](int b) { marked.push_back(b); };

    setBodyMaterialAndMark(doc, id, "petg", mark);
    EXPECT_EQ(doc.getBodyMaterial(id), "petg");
    EXPECT_EQ(doc.getBodyColor(id), red);
    EXPECT_EQ(marked, std::vector<int>{id});

    setBodyMaterialAndMark(doc, id, "brass", mark);
    EXPECT_EQ(doc.getBodyMaterial(id), "brass");
    EXPECT_EQ(doc.getBodyColor(id), findMaterial("brass")->color);

    // Clearing leaves the colour where the last pick put it.
    setBodyMaterialAndMark(doc, id, "", mark);
    EXPECT_EQ(doc.getBodyMaterial(id), "");
    EXPECT_EQ(doc.getBodyColor(id), findMaterial("brass")->color);
    EXPECT_EQ(marked.size(), 3u);
}

TEST(Materials, RoundTripsThroughAProjectFile) {
    Document src;
    const int a = src.addBody(BRepPrimAPI_MakeBox(10, 10, 10).Shape(), "a");
    const int b = src.addBody(BRepPrimAPI_MakeBox(5, 5, 5).Shape(), "b");
    const int c = src.addBody(BRepPrimAPI_MakeBox(2, 2, 2).Shape(), "c");
    src.setBodyMaterial(a, "asa");
    src.setBodyPrint(a, {7, 3});
    src.setBodyMaterial(c, "future_material_v9");   // unknown to this build
    (void)b;                                         // no material

    const std::string path = mzrtest::tmpPath("mz_test_materials.mzr");
    ASSERT_TRUE(ProjectIO::save(path, src).success);
    Document back;
    auto lr = ProjectIO::load(path, back);
    std::remove(path.c_str());
    ASSERT_TRUE(lr.success) << lr.errorMessage;

    EXPECT_EQ(back.getBodyMaterial(a), "asa");
    EXPECT_EQ(back.getBodyPrint(a).infillPercent, 7);
    EXPECT_EQ(back.getBodyPrint(a).walls, 3);
    EXPECT_EQ(back.getBodyPrint(b).infillPercent, PrintSettings{}.infillPercent);
    EXPECT_EQ(back.getBodyMaterial(b), "");
    EXPECT_EQ(back.getBodyMaterial(c), "future_material_v9");
}
