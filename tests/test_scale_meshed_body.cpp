// Regression: scaling a body that is on screen must not crash the next save.
//
// Steve, 2026-10-18: "it did commit the scale but as soon as it did and the
// object changed size, the program just closed out". The crash-recovery
// snapshot that runs right after the commit died in:
//
//   ProjectIO::save -> BinTools::Write -> BinTools_SurfaceSet::WriteSurface
//   SIGSEGV 'segmentation violation' detected. Address 0
//
// The Scale panel and gizmo always reach TransformOp through setScaleXYZ, which
// ran BRepBuilderAPI_GTransform even with Uniform ticked. GTransform NURBS-
// converts its input first, and on a body that still carries the viewport's
// triangulation, OCCT 7.9 hands back every already-B-spline face (his fillets)
// with a NULL surface. The scaled body looked fine; the next write crashed.
//
// A box converted to B-splines and then meshed reproduces it exactly.

#include "core/Document.h"
#include "core/History.h"
#include "modeling/BatchTransformOp.h"
#include "modeling/MeshFreeGTransform.h"
#include "modeling/TransformOp.h"

#include <gtest/gtest.h>
#include <BRepAdaptor_Surface.hxx>
#include <BRepBndLib.hxx>
#include <BRepBuilderAPI_NurbsConvert.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <BinTools.hxx>
#include <Bnd_Box.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <gp_GTrsf.hxx>
#include <gp_Mat.hxx>
#include <gp_XYZ.hxx>
#include <memory>
#include <sstream>

namespace {

// A 10 mm box whose faces are all B-spline surfaces, meshed as the viewport
// would have meshed it.
TopoDS_Shape meshedNurbsBox() {
    TopoDS_Shape s = BRepBuilderAPI_NurbsConvert(
        BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Shape()).Shape();
    BRepMesh_IncrementalMesh(s, 0.05, Standard_False, 0.3, Standard_True);
    return s;
}

TopoDS_Shape meshedBox() {
    TopoDS_Shape s = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Shape();
    BRepMesh_IncrementalMesh(s, 0.05, Standard_False, 0.3, Standard_True);
    return s;
}

bool writes(const TopoDS_Shape& s) {
    std::ostringstream out(std::ios::binary);
    BinTools::Write(s, out, Standard_False, Standard_False,
                    BinTools_FormatVersion_CURRENT);
    return !out.str().empty();
}

void extents(const TopoDS_Shape& s, double& dx, double& dy, double& dz) {
    Bnd_Box bb;
    BRepBndLib::Add(s, bb);
    double x0, y0, z0, x1, y1, z1;
    bb.Get(x0, y0, z0, x1, y1, z1);
    dx = x1 - x0; dy = y1 - y0; dz = z1 - z0;
}

int planeCount(const TopoDS_Shape& s) {
    int n = 0;
    for (TopExp_Explorer ex(s, TopAbs_FACE); ex.More(); ex.Next())
        if (BRepAdaptor_Surface(TopoDS::Face(ex.Current())).GetType() == GeomAbs_Plane) ++n;
    return n;
}

} // namespace

TEST(ScaleMeshedBody, NonUniformPanelScaleSaves) {
    Document doc;
    const int id = doc.addBody(meshedNurbsBox());

    auto op = std::make_unique<TransformOp>();
    op->setBodyId(id);
    op->setType(TransformType::Scale);
    op->setCenter(0, 0, 0);
    op->setScaleXYZ(0.8, 1.3, 0.8);
    History h;
    ASSERT_TRUE(h.pushOperation(std::move(op), doc));

    const TopoDS_Shape& out = doc.getBody(id);
    EXPECT_FALSE(materializr::MeshFreeGTransform::hasNullSurface(out));
    EXPECT_TRUE(writes(out)) << "the scaled body cannot be saved";
    double dx, dy, dz;
    extents(out, dx, dy, dz);
    EXPECT_NEAR(dx, 8.0, 1e-3);
    EXPECT_NEAR(dy, 13.0, 1e-3);
    EXPECT_NEAR(dz, 8.0, 1e-3);
}

// Uniform ticked: equal factors are an exact similarity, so the panel scale
// must not NURBS-convert at all - the box keeps its six planes.
TEST(ScaleMeshedBody, UniformPanelScaleStaysAnalytic) {
    Document doc;
    const int id = doc.addBody(meshedBox());

    auto op = std::make_unique<TransformOp>();
    op->setBodyId(id);
    op->setType(TransformType::Scale);
    op->setCenter(5, 5, 5);
    op->setScaleXYZ(0.8, 0.8, 0.8);
    History h;
    ASSERT_TRUE(h.pushOperation(std::move(op), doc));

    const TopoDS_Shape& out = doc.getBody(id);
    EXPECT_EQ(planeCount(out), 6) << "a uniform scale went through GTransform";
    EXPECT_TRUE(writes(out));
    double dx, dy, dz;
    extents(out, dx, dy, dz);
    EXPECT_NEAR(dx, 8.0, 1e-3);
    EXPECT_NEAR(dy, 8.0, 1e-3);
    EXPECT_NEAR(dz, 8.0, 1e-3);
}

TEST(ScaleMeshedBody, BatchUniformScaleStaysAnalytic) {
    Document doc;
    const int a = doc.addBody(meshedBox());
    const int b = doc.addBody(meshedNurbsBox());

    // How the gizmo builds it: a diagonal vectorial part (reads back gp_Other).
    gp_GTrsf g;
    g.SetVectorialPart(gp_Mat(0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5));
    g.SetTranslationPart(gp_XYZ(2.5, 2.5, 2.5));   // pivot (5,5,5)
    auto op = std::make_unique<BatchTransformOp>();
    op->setBodies({a, b});
    op->setTransform(g);
    History h;
    ASSERT_TRUE(h.pushOperation(std::move(op), doc));

    EXPECT_EQ(planeCount(doc.getBody(a)), 6);
    EXPECT_TRUE(writes(doc.getBody(a)));
    EXPECT_TRUE(writes(doc.getBody(b)));
    double dx, dy, dz;
    extents(doc.getBody(a), dx, dy, dz);
    EXPECT_NEAR(dx, 5.0, 1e-3);
}

TEST(ScaleMeshedBody, BatchNonUniformScaleSaves) {
    Document doc;
    const int id = doc.addBody(meshedNurbsBox());

    gp_GTrsf g;
    g.SetVectorialPart(gp_Mat(2.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0));
    auto op = std::make_unique<BatchTransformOp>();
    op->setBodies({id});
    op->setTransform(g);
    History h;
    ASSERT_TRUE(h.pushOperation(std::move(op), doc));

    const TopoDS_Shape& out = doc.getBody(id);
    EXPECT_FALSE(materializr::MeshFreeGTransform::hasNullSurface(out));
    EXPECT_TRUE(writes(out));
    double dx, dy, dz;
    extents(out, dx, dy, dz);
    EXPECT_NEAR(dx, 20.0, 1e-3);
}
