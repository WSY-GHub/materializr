#pragma once
// BRepBuilderAPI_GTransform for a document body, done on a mesh-free copy.
//
// GTransform runs BRepBuilderAPI_NurbsConvert on its input first. When that
// input still carries the viewport's triangulation (every body on screen does,
// BRepMesh writes it into the shared TShapes), OCCT 7.9's converter hands back
// faces and pcurve records whose surface handle is NULL. The result looks fine
// on screen, but the next BinTools::Write - the crash-recovery snapshot right
// after the commit - dereferences that null and takes the process down (Steve,
// 2026-10-18: "as soon as it did and the object changed size, the program just
// closed out"). The same shape with its mesh stripped converts cleanly, so the
// transform runs on a copy without one; the live body keeps its mesh.
//
// ModifiedShape() maps a sub-shape of the ORIGINAL body through both steps, so
// face-lineage carry-over works exactly as it did against the plain builder.

#include <BRepBuilderAPI_Copy.hxx>
#include <BRepBuilderAPI_GTransform.hxx>
#include <BRepTools.hxx>
#include <BRep_Tool.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Shape.hxx>
#include <gp_GTrsf.hxx>

#include <memory>

namespace materializr {

class MeshFreeGTransform {
public:
    MeshFreeGTransform(const TopoDS_Shape& shape, const gp_GTrsf& g) {
        m_copy = std::make_unique<BRepBuilderAPI_Copy>(
            shape, /*copyGeom=*/Standard_True, /*copyMesh=*/Standard_False);
        if (!m_copy->IsDone()) return;
        TopoDS_Shape clean = m_copy->Shape();
        BRepTools::Clean(clean);   // the copy's own TShapes; the live body is untouched
        m_gtf = std::make_unique<BRepBuilderAPI_GTransform>(clean, g, Standard_True);
        if (!m_gtf->IsDone() || m_gtf->Shape().IsNull()) return;
        m_done = !hasNullSurface(m_gtf->Shape());
    }

    bool IsDone() const { return m_done; }
    const TopoDS_Shape& Shape() const { return m_gtf->Shape(); }

    // Image of a sub-shape of the original body (null when there is none).
    TopoDS_Shape ModifiedShape(const TopoDS_Shape& s) const {
        return m_gtf->ModifiedShape(m_copy->ModifiedShape(s));
    }

    // Belt and braces: if the converter still leaves a face without a surface,
    // refuse the result instead of letting it reach the document and the next
    // save.
    static bool hasNullSurface(const TopoDS_Shape& shape) {
        for (TopExp_Explorer fx(shape, TopAbs_FACE); fx.More(); fx.Next()) {
            TopLoc_Location loc;
            if (BRep_Tool::Surface(TopoDS::Face(fx.Current()), loc).IsNull())
                return true;
        }
        return false;
    }

private:
    std::unique_ptr<BRepBuilderAPI_Copy> m_copy;
    std::unique_ptr<BRepBuilderAPI_GTransform> m_gtf;
    bool m_done = false;
};

} // namespace materializr
