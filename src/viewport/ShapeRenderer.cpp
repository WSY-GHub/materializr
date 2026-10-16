#include "gl_common.h"

#include "ShapeRenderer.h"

#include <glm/gtc/type_ptr.hpp>

#include <TopoDS_Shape.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include "core/MeshParams.h"
#include "io/MeshDiskCache.h"
#include "io/ImageDecode.h"
#include "EnvironmentImage.h"
#include <chrono>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Face.hxx>
#include <BRep_Tool.hxx>
#include <BRepTools.hxx>
#include <Poly_Triangulation.hxx>
#include <TopLoc_Location.hxx>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <glm/gtc/matrix_transform.hpp>

namespace materializr {

// The studio metals reflect: a bright sky over a grey floor (world Y is up)
// with a crisp horizon, and two softboxes. Sharp light/dark edges are what make
// a surface read as metal; roughness blurs them. Shared by the mesh shader
// (the fallback before a reflection probe exists) and the probe's background,
// so the two always agree.
#define MZR_STUDIO_GLSL R"(
vec3 studio(vec3 R, float rough) {
    float blur = mix(0.02, 0.5, rough);
    vec3 sky = mix(vec3(0.70, 0.74, 0.80), vec3(1.25), smoothstep(0.0, 0.9, R.y));
    vec3 floorc = mix(vec3(0.22), vec3(0.42), smoothstep(-1.0, -0.05, R.y));
    vec3 c = mix(floorc, sky, smoothstep(-blur, blur, R.y));
    float a = dot(R, normalize(vec3(0.55, 0.62, 0.55)));
    float b = dot(R, normalize(vec3(-0.75, 0.35, -0.35)));
    c += vec3(1.8) * smoothstep(cos(0.30 + blur), cos(0.30), a);
    c += vec3(1.1) * smoothstep(cos(0.22 + blur), cos(0.22), b);
    return c;
}

// The photographed studio (EnvironmentImage.h): an equirect panorama whose mip
// levels are pre-blurred for roughness, so level = roughness^2 * 5 matches the
// probe's mapping. Stored compressed into [0, kEnvironmentRange] and gamma
// encoded - keep the range in step with RANGE in tools/bake_environment.py.
uniform sampler2D u_photoEnv;
uniform bool u_usePhotoEnv;
const float kEnvironmentRange = 4.0;
vec3 photoStudio(vec3 R, float rough) {
    vec2 uv = vec2(atan(R.z, R.x) * 0.15915494 + 0.5,
                   acos(clamp(R.y, -1.0, 1.0)) * 0.31830989);
    vec3 t = textureLod(u_photoEnv, uv, rough * rough * 5.0).rgb;
    return kEnvironmentRange * pow(t, vec3(2.2));
}
vec3 environment(vec3 R, float rough) {
    return u_usePhotoEnv ? photoStudio(R, rough) : studio(R, rough);
}
)"

// Embedded mesh shader sources
static const char* s_meshVertSource = R"(
#version 330 core
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

// Paired with the identical declaration in EdgeRenderer's vertex shader: both
// programs must produce bit-identical gl_Position for the same vertex, or
// edges z-fight their own faces (see the comment there).
invariant gl_Position;

out vec3 v_worldPos;
out vec3 v_worldNormal;
out vec3 v_objPos;

void main() {
    vec4 worldPos = u_model * vec4(a_position, 1.0);
    v_worldPos = worldPos.xyz;
    v_objPos = a_position;
    mat3 normalMatrix = transpose(inverse(mat3(u_model)));
    v_worldNormal = normalize(normalMatrix * a_normal);
    gl_Position = u_projection * u_view * worldPos;
}
)";

static const char* s_meshFragSource = R"(
#version 330 core
in vec3 v_worldPos;
in vec3 v_worldNormal;
in vec3 v_objPos;

uniform vec3 u_viewPos;
uniform vec3 u_lightDir;       // key light, direction TO the light (normalized)
uniform vec3 u_fillDir;        // fill light, direction TO the light (normalized)
uniform vec3 u_objectColor;
uniform float u_specStrength;  // highlight brightness (material finish)
uniform float u_shininess;     // highlight tightness (material finish)
uniform float u_metallic;      // 0 = plastic/wood, 1 = metal
uniform float u_roughness;     // blurs a metal's reflection
uniform float u_clearcoat;     // varnish over the base (wood)
uniform float u_grain;         // wood-grain contrast, 0 = none
uniform vec3 u_grainCenter;    // object-space centre of the mesh
uniform vec3 u_grainAxis;      // along the grain: the mesh's longest extent
uniform vec3 u_grainSide;      // toward the pith: the mesh's shortest extent
uniform float u_grainScale;    // growth-ring spacing, model units
uniform samplerCube u_envMap;  // reflection probe: the scene around the metals
uniform bool u_hasEnv;         // false until a probe exists (and while drawing one)
uniform float u_envMaxLod;     // probe's smallest mip: the roughest blur
uniform bool u_selected;
uniform float u_ambient;       // base illumination 0..1 (softens shadows)
uniform bool u_headlight;      // key light tracks the camera when true
uniform float u_fillStrength;  // fill light contribution (0 disables it)
uniform bool u_previewCut;     // tint red: this volume will be subtracted
uniform bool u_sectionEnabled; // section view: clip away one side of a plane
uniform vec3 u_sectionPoint;   // a point on the section plane (world)
uniform vec3 u_sectionNormal;  // plane normal; the normal side is removed

out vec4 fragColor;

float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

float valueNoise(vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash13(i),                    hash13(i + vec3(1.0, 0.0, 0.0)), f.x),
                   mix(hash13(i + vec3(0.0, 1.0, 0.0)), hash13(i + vec3(1.0, 1.0, 0.0)), f.x), f.y),
               mix(mix(hash13(i + vec3(0.0, 0.0, 1.0)), hash13(i + vec3(1.0, 0.0, 1.0)), f.x),
                   mix(hash13(i + vec3(0.0, 1.0, 1.0)), hash13(i + vec3(1.0, 1.0, 1.0)), f.x), f.y), f.z);
}

// Brightness factor of solid timber at an object-space point. The growth rings
// are cylinders around a pith that lies well off the part, below its thinnest
// side, so they cross the faces as gently curving bands - a flat-sawn board.
// Noise warps them along the grain, and fine fibre streaks run with it.
float woodShade(vec3 p) {
    vec3 q = (p - u_grainCenter) / u_grainScale;
    vec3 third = cross(u_grainAxis, u_grainSide);
    float along = dot(q, u_grainAxis);
    float side = dot(q, u_grainSide);
    float across = dot(q, third);
    float warp = valueNoise(vec3(along * 0.04, side * 0.25, across * 0.25)) * 1.5
               + valueNoise(vec3(along * 0.12, side * 0.9, across * 0.9)) * 0.15;
    float ring = fract(length(vec2(side + 10.0, across)) + warp);
    // Pale earlywood, then a darker latewood band that fades back out.
    float late = smoothstep(0.45, 0.8, ring) * (1.0 - smoothstep(0.86, 1.0, ring));
    float fibre = valueNoise(vec3(along * 0.5, side * 12.0, across * 12.0));
    return 1.08 - 0.42 * late + 0.14 * (fibre - 0.5);
}

)" MZR_STUDIO_GLSL R"(
void main() {
    if (u_sectionEnabled &&
        dot(v_worldPos - u_sectionPoint, u_sectionNormal) > 0.0) discard;
    vec3 normal = normalize(v_worldNormal);
    // A section cut exposes interior back faces; flip their normals so the
    // inside is lit instead of rendering black.
    if (!gl_FrontFacing) normal = -normal;
    vec3 viewDir = normalize(u_viewPos - v_worldPos);
    // Headlight: the key light comes from the camera, so the face the user is
    // looking at is always lit and large cast shadows disappear.
    vec3 keyDir = u_headlight ? viewDir : normalize(u_lightDir);

    vec3 base = u_objectColor;
    if (u_grain > 0.0) base *= mix(1.0, woodShade(v_objPos), u_grain);

    float keyDiff = max(dot(normal, keyDir), 0.0);
    float fillDiff = max(dot(normal, normalize(u_fillDir)), 0.0) * u_fillStrength;
    vec3 halfwayDir = normalize(keyDir + viewDir);
    float nh = max(dot(normal, halfwayDir), 0.0);
    float spec = pow(nh, u_shininess);

    // Plastic, wood: what every body drew before materials existed.
    vec3 result = u_ambient * base + (keyDiff + fillDiff) * base
                + u_specStrength * spec * vec3(1.0);

    vec3 R = reflect(-viewDir, normal);
    float nv = max(dot(normal, viewDir), 0.0);
    float fresnel = pow(1.0 - nv, 5.0);

    // Varnish: a thin clear layer - a tight white highlight and a faint
    // reflection that strengthens toward grazing angles.
    if (u_clearcoat > 0.0) {
        vec3 coat = 0.9 * pow(nh, 160.0) * vec3(1.0)
                  + (0.05 + 0.6 * fresnel) * environment(R, 0.15);
        result += u_clearcoat * coat;
    }

    // A metal scatters almost no light diffusely - its look is the studio it
    // reflects, tinted by its colour (white again at grazing angles).
    if (u_metallic > 0.0) {
        vec3 tint = mix(base, vec3(1.0), fresnel);
        vec3 env = u_hasEnv
            // Squared: a polished metal samples near the sharp top mip, and
            // only a properly rough one blurs into the low-res tail.
            ? textureLod(u_envMap, R, u_roughness * u_roughness * u_envMaxLod).rgb
            : environment(R, u_roughness);
        vec3 metal = 0.25 * (u_ambient + keyDiff + fillDiff) * base
                   + 0.8 * env * tint
                   + u_specStrength * spec * base;
        result = mix(result, metal, u_metallic);
    }

    if (u_previewCut) {
        result = mix(result, vec3(0.9, 0.1, 0.1), 0.55);
    }

    if (u_selected) {
        result = mix(result, vec3(0.3, 0.5, 1.0), 0.3);
    }

    fragColor = vec4(result, 1.0);
}
)";

// Reflection-probe background: the studio, plus the ground grid where a ray
// from the probe meets the world floor (y = 0), so a metal part standing on the
// grid shows it in its sides.
static const char* s_probeVertSource = R"(
#version 330 core
out vec2 v_ndc;
void main() {
    vec2 pos = vec2(float((gl_VertexID & 1) * 2 - 1), float((gl_VertexID >> 1) * 2 - 1));
    v_ndc = pos;
    gl_Position = vec4(pos, 0.0, 1.0);
}
)";

static const char* s_probeFragSource = R"(
#version 330 core
in vec2 v_ndc;
uniform mat4 u_invViewProj;
uniform vec3 u_probePos;
uniform float u_gridFade;   // distance over which the reflected grid fades out
out vec4 fragColor;
)" MZR_STUDIO_GLSL R"(
float gridLine(vec2 g) {
    vec2 w = max(fwidth(g), vec2(1e-4));
    vec2 l = abs(fract(g - 0.5) - 0.5) / w;
    return 1.0 - min(min(l.x, l.y), 1.0);
}
void main() {
    vec4 w = u_invViewProj * vec4(v_ndc, 1.0, 1.0);
    vec3 d = normalize(w.xyz / w.w - u_probePos);
    vec3 c = environment(d, 0.0);
    float t = (abs(d.y) > 1e-4) ? -u_probePos.y / d.y : -1.0;
    if (t > 0.0) {
        vec2 p = (u_probePos + d * t).xz;
        float lines = max(0.35 * gridLine(p / 10.0), 0.7 * gridLine(p / 100.0));
        c = mix(c, vec3(0.80), lines * exp(-t / u_gridFade));
    }
    fragColor = vec4(c, 1.0);
}
)";

// Embedded outline shader sources
static const char* s_outlineVertSource = R"(
#version 330 core
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_outlineWidth;

void main() {
    vec3 expandedPos = a_position + normalize(a_normal) * u_outlineWidth;
    gl_Position = u_projection * u_view * u_model * vec4(expandedPos, 1.0);
}
)";

static const char* s_outlineFragSource = R"(
#version 330 core
uniform vec4 u_outlineColor;
out vec4 fragColor;

void main() {
    fragColor = u_outlineColor;
}
)";

ShapeRenderer::ShapeRenderer() {}

ShapeRenderer::~ShapeRenderer()
{
    clear();
    if (m_meshProgram) {
        glDeleteProgram(m_meshProgram);
        m_meshProgram = 0;
    }
    if (m_outlineProgram) {
        glDeleteProgram(m_outlineProgram);
        m_outlineProgram = 0;
    }
    if (m_probeBgProgram) glDeleteProgram(m_probeBgProgram);
    if (m_probeFbo) glDeleteFramebuffers(1, &m_probeFbo);
    if (m_probeDepth) glDeleteRenderbuffers(1, &m_probeDepth);
    if (m_probeTex) glDeleteTextures(1, &m_probeTex);
    if (m_probeVao) glDeleteVertexArrays(1, &m_probeVao);
    if (m_photoTex) glDeleteTextures(1, &m_photoTex);
}

// Unpack the embedded atlas (layout in tools/bake_environment.py) into one
// equirect texture whose mip chain is the pre-blurred levels. The levels are
// uploaded as they are, never regenerated: a box-filtered mip of the sharp
// panorama would be the wrong blur.
void ShapeRenderer::loadPhotoEnvironment()
{
    DecodedImage img;
    if (!decodeImage(kEnvironmentJpeg, kEnvironmentJpegSize, img) ||
        img.width != 1024 || img.height != 768) {
        std::fprintf(stderr, "[Render] studio environment failed to decode - using the drawn studio\n");
        return;
    }
    glGenTextures(1, &m_photoTex);
    glBindTexture(GL_TEXTURE_2D, m_photoTex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    std::vector<unsigned char> level;
    int x = 0;
    for (int lvl = 0; lvl < 6; ++lvl) {
        const int w = 1024 >> lvl, h = 512 >> lvl;
        const int ox = lvl == 0 ? 0 : x, oy = lvl == 0 ? 0 : 512;
        level.resize(static_cast<size_t>(w) * h * 4);
        for (int r = 0; r < h; ++r)
            std::copy_n(&img.rgba[(static_cast<size_t>(oy + r) * img.width + ox) * 4],
                        static_cast<size_t>(w) * 4, &level[static_cast<size_t>(r) * w * 4]);
        glTexImage2D(GL_TEXTURE_2D, lvl, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, level.data());
        if (lvl > 0) x += w;
    }
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 5);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);         // round the panorama
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);  // not over the poles
    glBindTexture(GL_TEXTURE_2D, 0);
}

bool ShapeRenderer::initialize()
{
    // Compile mesh shader
    unsigned int meshVert = 0, meshFrag = 0;
    if (!compileShader(meshVert, GL_VERTEX_SHADER, s_meshVertSource)) return false;
    if (!compileShader(meshFrag, GL_FRAGMENT_SHADER, s_meshFragSource)) {
        glDeleteShader(meshVert);
        return false;
    }
    m_meshProgram = glCreateProgram();
    if (!linkProgram(m_meshProgram, meshVert, meshFrag)) {
        glDeleteShader(meshVert);
        glDeleteShader(meshFrag);
        return false;
    }
    glDeleteShader(meshVert);
    glDeleteShader(meshFrag);

    // Cache mesh uniform locations
    m_meshLoc_model = glGetUniformLocation(m_meshProgram, "u_model");
    m_meshLoc_view = glGetUniformLocation(m_meshProgram, "u_view");
    m_meshLoc_projection = glGetUniformLocation(m_meshProgram, "u_projection");
    m_meshLoc_viewPos = glGetUniformLocation(m_meshProgram, "u_viewPos");
    m_meshLoc_lightDir = glGetUniformLocation(m_meshProgram, "u_lightDir");
    m_meshLoc_fillDir = glGetUniformLocation(m_meshProgram, "u_fillDir");
    m_meshLoc_objectColor = glGetUniformLocation(m_meshProgram, "u_objectColor");
    m_meshLoc_specStrength = glGetUniformLocation(m_meshProgram, "u_specStrength");
    m_meshLoc_shininess = glGetUniformLocation(m_meshProgram, "u_shininess");
    m_meshLoc_metallic = glGetUniformLocation(m_meshProgram, "u_metallic");
    m_meshLoc_roughness = glGetUniformLocation(m_meshProgram, "u_roughness");
    m_meshLoc_clearcoat = glGetUniformLocation(m_meshProgram, "u_clearcoat");
    m_meshLoc_grain = glGetUniformLocation(m_meshProgram, "u_grain");
    m_meshLoc_grainCenter = glGetUniformLocation(m_meshProgram, "u_grainCenter");
    m_meshLoc_grainAxis = glGetUniformLocation(m_meshProgram, "u_grainAxis");
    m_meshLoc_grainSide = glGetUniformLocation(m_meshProgram, "u_grainSide");
    m_meshLoc_grainScale = glGetUniformLocation(m_meshProgram, "u_grainScale");
    m_meshLoc_envMap = glGetUniformLocation(m_meshProgram, "u_envMap");
    m_meshLoc_hasEnv = glGetUniformLocation(m_meshProgram, "u_hasEnv");
    m_meshLoc_envMaxLod = glGetUniformLocation(m_meshProgram, "u_envMaxLod");

    // Reflection-probe background. Optional: without it metals fall back to
    // the analytic studio, so a failure here is not a failure to initialise.
    {
        unsigned int pv = 0, pf = 0;
        if (compileShader(pv, GL_VERTEX_SHADER, s_probeVertSource)) {
            if (compileShader(pf, GL_FRAGMENT_SHADER, s_probeFragSource)) {
                m_probeBgProgram = glCreateProgram();
                if (!linkProgram(m_probeBgProgram, pv, pf)) {
                    glDeleteProgram(m_probeBgProgram);
                    m_probeBgProgram = 0;
                }
                glDeleteShader(pf);
            }
            glDeleteShader(pv);
        }
        if (m_probeBgProgram) {
            m_probeLoc_invViewProj = glGetUniformLocation(m_probeBgProgram, "u_invViewProj");
            m_probeLoc_probePos = glGetUniformLocation(m_probeBgProgram, "u_probePos");
            m_probeLoc_gridFade = glGetUniformLocation(m_probeBgProgram, "u_gridFade");
            m_probeLoc_photoEnv = glGetUniformLocation(m_probeBgProgram, "u_photoEnv");
            m_probeLoc_usePhotoEnv = glGetUniformLocation(m_probeBgProgram, "u_usePhotoEnv");
        }
    }
    m_meshLoc_photoEnv = glGetUniformLocation(m_meshProgram, "u_photoEnv");
    m_meshLoc_usePhotoEnv = glGetUniformLocation(m_meshProgram, "u_usePhotoEnv");
    loadPhotoEnvironment();
    m_meshLoc_selected = glGetUniformLocation(m_meshProgram, "u_selected");
    m_meshLoc_ambient = glGetUniformLocation(m_meshProgram, "u_ambient");
    m_meshLoc_headlight = glGetUniformLocation(m_meshProgram, "u_headlight");
    m_meshLoc_fillStrength = glGetUniformLocation(m_meshProgram, "u_fillStrength");
    m_meshLoc_previewCut = glGetUniformLocation(m_meshProgram, "u_previewCut");
    m_meshLoc_sectionEnabled = glGetUniformLocation(m_meshProgram, "u_sectionEnabled");
    m_meshLoc_sectionPoint = glGetUniformLocation(m_meshProgram, "u_sectionPoint");
    m_meshLoc_sectionNormal = glGetUniformLocation(m_meshProgram, "u_sectionNormal");

    // Compile outline shader
    unsigned int outlineVert = 0, outlineFrag = 0;
    if (!compileShader(outlineVert, GL_VERTEX_SHADER, s_outlineVertSource)) return false;
    if (!compileShader(outlineFrag, GL_FRAGMENT_SHADER, s_outlineFragSource)) {
        glDeleteShader(outlineVert);
        return false;
    }
    m_outlineProgram = glCreateProgram();
    if (!linkProgram(m_outlineProgram, outlineVert, outlineFrag)) {
        glDeleteShader(outlineVert);
        glDeleteShader(outlineFrag);
        return false;
    }
    glDeleteShader(outlineVert);
    glDeleteShader(outlineFrag);

    // Cache outline uniform locations
    m_outlineLoc_model = glGetUniformLocation(m_outlineProgram, "u_model");
    m_outlineLoc_view = glGetUniformLocation(m_outlineProgram, "u_view");
    m_outlineLoc_projection = glGetUniformLocation(m_outlineProgram, "u_projection");
    m_outlineLoc_outlineColor = glGetUniformLocation(m_outlineProgram, "u_outlineColor");
    m_outlineLoc_outlineWidth = glGetUniformLocation(m_outlineProgram, "u_outlineWidth");

    return true;
}

void ShapeRenderer::notePreMeshed(const TopoDS_Shape& shape,
                                  float requestedDeflection,
                                  float requestedAngularDeflection) {
    if (!shape.IsNull())
        m_meshedAt[shape.TShape().get()] =
            materializr::makeMeshTag(shape, requestedDeflection, requestedAngularDeflection);
}

void ShapeRenderer::forgetPreMeshed(const TopoDS_Shape& shape) {
    if (shape.IsNull()) return;
    m_meshedAt.erase(shape.TShape().get());
    m_meshedAtPrev.erase(shape.TShape().get());
}

int ShapeRenderer::tessellate(const TopoDS_Shape& shape, float deflection,
                              float angularDeflection)
{
    // Skip the mesher when this exact TShape was already meshed for the
    // requested (linear, angular) deflection - by an earlier tessellate (a full rebuild
    // re-runs setBodyMesh for every body, not just the ones the last
    // operation touched) or by a worker thread that pre-meshed a heavy result
    // (a swept thread's 35-turn helicoid took ~10 s on the main thread; see
    // notePreMeshed). The tag lives in m_meshedAt because OCCT stores only the
    // ACHIEVED deflection on each Poly_Triangulation (0 on a plane, ~0.007 on
    // a small cylinder asked for 0.1), never the requested value; comparing
    // that against the request could never match, so the old pre-mesh check
    // re-meshed every time. The quality slider still takes effect in BOTH
    // directions: a different value never matches, so a finer-than-requested
    // mesh cannot survive a quality lowering (that was a real bug once).
    // The tag also records which faces the mesher left bare, so a body with
    // a face it cannot triangulate is not Cleaned and re-meshed on every
    // rebuild, while a fresh shape at a recycled address (every face bare)
    // is never trusted - see meshTagCovers().
    const void* key = shape.TShape().get();
    auto tag = m_meshedAt.find(key);
    if (tag == m_meshedAt.end()) {
        auto prev = m_meshedAtPrev.find(key);
        if (prev != m_meshedAtPrev.end()) tag = m_meshedAt.insert(*prev).first;
    }
    const bool preMeshed = tag != m_meshedAt.end() &&
                           materializr::meshTagCovers(tag->second, deflection,
                                                      angularDeflection, shape);
    if (!preMeshed) {
        // Drop any cached triangulation first. BRepMesh_IncrementalMesh only
        // ever refines an existing mesh, so without this a previously finer
        // tessellation would be kept when the user *lowers* the quality.
        // Cleaning forces the mesh to be rebuilt at exactly the requested
        // deflection in either direction.
        BRepTools::Clean(shape);

        // Perform tessellation. The angular deflection subdivides curved
        // surfaces by normal angle, so rounded edges/holes get more facets
        // (smoother) while flat faces stay cheap. Run in parallel to absorb
        // the extra triangles.
        const auto t0 = std::chrono::steady_clock::now();
#ifdef MZR_PARALLEL_MESH_TESTING
        ++m_meshCallsForTest;
#endif
        // Opening a project: the same bytes as last time mesh the same way,
        // so take last time's mesh if it was worth keeping (MeshDiskCache).
        const bool fromDisk =
            materializr::meshcache::load(shape, deflection, angularDeflection);
        if (!fromDisk) {
        // Ruled walls first, exactly: the general pass keeps them, and spent
        // 16 s on autumn.mzr's lettering wall at Ultra (meshing it wrong).
        materializr::premeshRuledWalls(shape, deflection, angularDeflection);
        BRepMesh_IncrementalMesh meshGen(
            shape, materializr::meshParams(deflection, angularDeflection, true));

        // Delabella (chosen above for speed on many-holed faces) can leave a
        // face bare - confirmed on a real boolean-result body where two large
        // faces came back with zero triangles despite the shape being fully
        // BRepCheck-valid, no free edges, properly closed. The rest of the
        // pipeline has no idea: the shape is fine, only this one mesher choice
        // failed on it. Retry just the bare faces with Watson (OCCT's default,
        // slower but far more robust) instead of leaving a hole in the body -
        // it only costs the handful of faces that actually need it, not the
        // whole shape, so Delabella's perf win everywhere else is unaffected.
        int watsonRetried = 0, watsonRecovered = 0;
        for (TopExp_Explorer fx(shape, TopAbs_FACE); fx.More(); fx.Next()) {
            const TopoDS_Face& bareFace = TopoDS::Face(fx.Current());
            TopLoc_Location loc;
            if (!BRep_Tool::Triangulation(bareFace, loc).IsNull()) continue;
            ++watsonRetried;
            materializr::meshFaceWatsonBounded(bareFace, deflection, angularDeflection,
                                               materializr::kBareFaceBudgetSec);
            // Watson at the requested quality isn't always enough either - a
            // real planar face has been seen failing BOTH meshers at fine
            // (High/Ultra) deflection while meshing fine at anything coarser.
            // Back off in a few steps rather than leave a hole on screen.
            if (!BRep_Tool::Triangulation(bareFace, loc).IsNull() ||
                materializr::meshBareFaceEscalating(bareFace, deflection, angularDeflection))
                ++watsonRecovered;
        }
        if (watsonRetried > 0) {
            std::fprintf(stderr,
                "[ShapeRenderer] Delabella left %d face(s) bare - Watson "
                "fallback recovered %d\n", watsonRetried, watsonRecovered);
        }
        // A face can also come back meshed but wrong - see repairInaccurateFaces.
        materializr::repairInaccurateFaces(shape, deflection, angularDeflection);
        }

        m_lastMeshMs = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - t0).count();
        if (!fromDisk && m_lastMeshMs >= materializr::meshcache::kStoreMinMs)
            materializr::meshcache::store(shape, deflection, angularDeflection);
        m_meshedAt[key] = materializr::makeMeshTag(shape, deflection, angularDeflection);
    }

    // Collect all triangle vertices (position + normal)
    std::vector<float> vertices;

    for (TopExp_Explorer explorer(shape, TopAbs_FACE); explorer.More(); explorer.Next()) {
        const TopoDS_Face& face = TopoDS::Face(explorer.Current());
        TopLoc_Location location;
        Handle(Poly_Triangulation) triangulation = BRep_Tool::Triangulation(face, location);

        if (triangulation.IsNull()) continue;

        const gp_Trsf& trsf = location.Transformation();
        bool hasTransform = !location.IsIdentity();

        int nbTriangles = triangulation->NbTriangles();
        const bool hasNormals = triangulation->HasNormals();

        // The mesher stores no normals, and one flat normal per triangle
        // draws every curved face as visible facets - which a metal's
        // reflection magnifies into a disco ball. Average the triangles'
        // area-weighted normals at each node instead. A BRep face is smooth
        // inside, so this rounds only what is really round; edges between
        // faces stay sharp because each face has its own nodes, and an STL
        // import (one face per triangle) stays faceted.
        std::vector<gp_Vec> nodeNormals;
        if (!hasNormals) {
            nodeNormals.assign(triangulation->NbNodes() + 1, gp_Vec(0, 0, 0));
            for (int i = 1; i <= nbTriangles; ++i) {
                int a, b, c;
                triangulation->Triangle(i).Get(a, b, c);
                if (face.Orientation() == TopAbs_REVERSED) std::swap(a, b);
                const gp_Pnt pa = triangulation->Node(a);
                const gp_Vec n = gp_Vec(pa, triangulation->Node(b))
                                     .Crossed(gp_Vec(pa, triangulation->Node(c)));
                nodeNormals[a] += n;
                nodeNormals[b] += n;
                nodeNormals[c] += n;
            }
            for (gp_Vec& n : nodeNormals) {
                if (n.Magnitude() <= 1e-12) continue;
                n.Normalize();
                if (hasTransform) n.Transform(trsf);
            }
        }

        for (int i = 1; i <= nbTriangles; ++i) {
            const Poly_Triangle& tri = triangulation->Triangle(i);

            int n1, n2, n3;
            tri.Get(n1, n2, n3);

            // Handle face orientation
            if (face.Orientation() == TopAbs_REVERSED) {
                std::swap(n1, n2);
            }

            // Get vertices
            gp_Pnt p1 = triangulation->Node(n1);
            gp_Pnt p2 = triangulation->Node(n2);
            gp_Pnt p3 = triangulation->Node(n3);

            // Apply location transformation if present
            if (hasTransform) {
                p1.Transform(trsf);
                p2.Transform(trsf);
                p3.Transform(trsf);
            }

            // Compute face normal from triangle vertices
            gp_Vec v1(p1, p2);
            gp_Vec v2(p1, p3);
            gp_Vec normal = v1.Crossed(v2);
            if (normal.Magnitude() > 1e-10) {
                normal.Normalize();
            } else {
                normal = gp_Vec(0, 1, 0);
            }

            auto addVertex = [&](const gp_Pnt& p, int nodeIdx) {
                vertices.push_back(static_cast<float>(p.X()));
                vertices.push_back(static_cast<float>(p.Y()));
                vertices.push_back(static_cast<float>(p.Z()));

                if (hasNormals) {
                    gp_Dir n = triangulation->Normal(nodeIdx);
                    if (hasTransform) {
                        n.Transform(trsf);
                    }
                    if (face.Orientation() == TopAbs_REVERSED) {
                        vertices.push_back(static_cast<float>(-n.X()));
                        vertices.push_back(static_cast<float>(-n.Y()));
                        vertices.push_back(static_cast<float>(-n.Z()));
                    } else {
                        vertices.push_back(static_cast<float>(n.X()));
                        vertices.push_back(static_cast<float>(n.Y()));
                        vertices.push_back(static_cast<float>(n.Z()));
                    }
                } else {
                    // The smoothed node normal, unless the node sits on a
                    // fold the face itself has (over ~60 degrees off this
                    // triangle) - keep that crease crisp.
                    gp_Vec n = normal;
                    const gp_Vec& sn = nodeNormals[nodeIdx];
                    if (sn.Magnitude() > 0.5 && sn.Dot(normal) > 0.5) n = sn;
                    vertices.push_back(static_cast<float>(n.X()));
                    vertices.push_back(static_cast<float>(n.Y()));
                    vertices.push_back(static_cast<float>(n.Z()));
                }
            };

            addVertex(p1, n1);
            addVertex(p2, n2);
            addVertex(p3, n3);
        }
    }

    const int slot = appendVertices(vertices);
    if (slot >= 0) {
        m_meshes[slot].shape = shape;
        m_meshes[slot].deflection = deflection;
        m_meshes[slot].angularDeflection = angularDeflection;
    }
    return slot;
}

int ShapeRenderer::appendVertices(const std::vector<float>& vertices)
{
    if (vertices.empty()) {
        return -1;
    }

    // Upload to GPU
    MeshData mesh;
    mesh.vertexCount = static_cast<int>(vertices.size() / 6); // 6 floats per vertex
    mesh.bbMin = mesh.bbMax = glm::vec3(vertices[0], vertices[1], vertices[2]);
    for (size_t i = 0; i + 2 < vertices.size(); i += 6) {
        const glm::vec3 p(vertices[i], vertices[i + 1], vertices[i + 2]);
        mesh.bbMin = glm::min(mesh.bbMin, p);
        mesh.bbMax = glm::max(mesh.bbMax, p);
    }

    glGenVertexArrays(1, &mesh.vao);
    glGenBuffers(1, &mesh.vbo);

    glBindVertexArray(mesh.vao);
    glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 vertices.size() * sizeof(float),
                 vertices.data(),
                 GL_STATIC_DRAW);

    // Position attribute (location = 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Normal attribute (location = 1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                          (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

    int index = static_cast<int>(m_meshes.size());
    m_meshes.push_back(mesh);
    return index;
}

bool ShapeRenderer::isPreMeshed(const TopoDS_Shape& shape, float deflection,
                                float angularDeflection) const
{
    // Same rule as tessellate(): the tag for this TShape, in either map, must
    // still cover the shape at exactly these parameters.
    const void* key = shape.TShape().get();
    auto tag = m_meshedAt.find(key);
    if (tag == m_meshedAt.end()) {
        tag = m_meshedAtPrev.find(key);
        if (tag == m_meshedAtPrev.end()) return false;
    }
    return materializr::meshTagCovers(tag->second, deflection, angularDeflection, shape);
}

bool ShapeRenderer::reclaimStale(int bodyId)
{
    for (size_t i = 0; i < m_retired.size(); ++i) {
        if (m_retired[i].bodyId != bodyId || !m_retired[i].vao) continue;
        const int slot = static_cast<int>(m_meshes.size());
        m_meshes.push_back(m_retired[i]);
        m_retired[i] = m_retired.back();
        m_retired.pop_back();
        m_bodyToSlot[bodyId] = slot;
        return true;
    }
    return false;
}

bool ShapeRenderer::hasMeshFor(int bodyId) const
{
    auto it = m_bodyToSlot.find(bodyId);
    if (it != m_bodyToSlot.end() && it->second >= 0 &&
        it->second < static_cast<int>(m_meshes.size()) && m_meshes[it->second].vao)
        return true;
    for (const MeshData& r : m_retired)
        if (r.bodyId == bodyId && r.vao) return true;
    return false;
}

int ShapeRenderer::setBodyMesh(int bodyId, const TopoDS_Shape& shape,
                               float deflection, float angularDeflection)
{
    m_lastMeshMs = -1.0;
    // Reclaim the slot retireAll() kept for this exact shape at this quality,
    // if any: no mesher, no vertex collection, no upload. It comes back as a
    // fresh slot (default colour / flags / matrix), exactly like a new
    // tessellation would; the caller re-applies colour and previews.
    int appendedSlot = -1;
    for (size_t i = 0; i < m_retired.size(); ++i) {
        const MeshData& r = m_retired[i];
        if (r.vao && r.deflection == deflection &&
            r.angularDeflection == angularDeflection && r.shape.IsEqual(shape)) {
            MeshData fresh;
            fresh.vao = r.vao;
            fresh.vbo = r.vbo;
            fresh.vertexCount = r.vertexCount;
            fresh.bbMin = r.bbMin;
            fresh.bbMax = r.bbMax;
            fresh.shape = r.shape;
            fresh.deflection = r.deflection;
            fresh.angularDeflection = r.angularDeflection;
            appendedSlot = static_cast<int>(m_meshes.size());
            m_meshes.push_back(fresh);
            m_retired[i] = m_retired.back();
            m_retired.pop_back();
            // The tag stays authoritative for this TShape even though
            // tessellate() did not run; recount what is bare on the live shape.
            m_meshedAt[shape.TShape().get()] =
                materializr::makeMeshTag(shape, deflection, angularDeflection);
            break;
        }
    }
    // Otherwise tessellate first so we don't free the old slot's GL resources
    // unless the new tessellation actually succeeds.
    if (appendedSlot < 0)
        appendedSlot = tessellate(shape, deflection, angularDeflection);
    if (appendedSlot < 0) {
        // Failed - leave the existing slot (if any) in place. LOUDLY: a
        // kept stale slot means the screen shows geometry the document no
        // longer has (phantom bodies, "unclickable" faces).
        auto it = m_bodyToSlot.find(bodyId);
        std::fprintf(stderr,
                     "[Mesh] tessellation FAILED for body %d - %s\n", bodyId,
                     (it == m_bodyToSlot.end())
                         ? "no previous mesh, body will not render"
                         : "KEEPING STALE MESH (render != document!)");
        return (it == m_bodyToSlot.end()) ? -1 : it->second;
    }
    return placeSlot(bodyId, appendedSlot);
}

int ShapeRenderer::setBodyVertices(int bodyId, const std::vector<float>& vertices)
{
    m_lastMeshMs = -1.0;
    const int appendedSlot = appendVertices(vertices);
    if (appendedSlot < 0) {
        auto it = m_bodyToSlot.find(bodyId);
        return (it == m_bodyToSlot.end()) ? -1 : it->second;
    }
    return placeSlot(bodyId, appendedSlot);
}

int ShapeRenderer::placeSlot(int bodyId, int appendedSlot)
{
    auto it = m_bodyToSlot.find(bodyId);
    if (it == m_bodyToSlot.end()) {
        // New body; the appended slot is its home.
        m_meshes[appendedSlot].bodyId = bodyId;
        m_bodyToSlot[bodyId] = appendedSlot;
        return appendedSlot;
    }
    // Existing slot - relocate the new mesh's GL data into it so callers
    // that cached the slot index keep working (setColor / setSelected use
    // index lookups). Carry over cosmetic state.
    int oldSlot = it->second;
    if (oldSlot < 0 || oldSlot >= static_cast<int>(m_meshes.size())) {
        // Stale mapping - treat as fresh insert.
        m_meshes[appendedSlot].bodyId = bodyId;
        m_bodyToSlot[bodyId] = appendedSlot;
        return appendedSlot;
    }
    MeshData& old = m_meshes[oldSlot];
    if (old.vao) glDeleteVertexArrays(1, &old.vao);
    if (old.vbo) glDeleteBuffers(1, &old.vbo);
    glm::vec3 keepColor   = old.color;
    SurfaceFinish keepFinish = old.finish;
    bool keepSelected     = old.selected;
    bool keepSubPreview   = old.subtractPreview;
    glm::mat4 keepModel   = old.modelMatrix;
    old = m_meshes[appendedSlot];
    old.bodyId          = bodyId;
    old.color           = keepColor;
    old.finish          = keepFinish;
    old.selected        = keepSelected;
    old.subtractPreview = keepSubPreview;
    old.modelMatrix     = keepModel;
    // The appended slot was always the tail (tessellate's push_back); pop it
    // so render() doesn't visit a stale duplicate.
    m_meshes.pop_back();
    return oldSlot;
}

void ShapeRenderer::removeBody(int bodyId)
{
    auto it = m_bodyToSlot.find(bodyId);
    if (it == m_bodyToSlot.end()) return;
    int slot = it->second;
    m_bodyToSlot.erase(it);
    if (slot < 0 || slot >= static_cast<int>(m_meshes.size())) return;
    MeshData& m = m_meshes[slot];
    if (m.vao) glDeleteVertexArrays(1, &m.vao);
    if (m.vbo) glDeleteBuffers(1, &m.vbo);
    m.vao = 0; m.vbo = 0; m.vertexCount = 0; m.bodyId = -1;
    m.shape.Nullify(); // do not pin a deleted body's geometry through m_retired
    // Slot stays in the vector so other slots' indices don't shift.
    // render() skips slots with vertexCount==0.
}

int ShapeRenderer::findSlotByBody(int bodyId) const
{
    auto it = m_bodyToSlot.find(bodyId);
    return (it == m_bodyToSlot.end()) ? -1 : it->second;
}

void ShapeRenderer::render(const glm::mat4& view, const glm::mat4& projection,
                           const glm::vec3& viewPos)
{
    if (!m_meshProgram || m_meshes.empty()) return;

    updateReflectionProbe();

    glEnable(GL_DEPTH_TEST);

    // Section-view clip uniforms. Set before BOTH passes - the outline pass
    // below also runs the mesh program for its stencil fill, and it must
    // clip identically or the selection glow paints over the removed half.
    glUseProgram(m_meshProgram);
    glUniform1i(m_meshLoc_sectionEnabled, m_sectionEnabled ? 1 : 0);
    glUniform3fv(m_meshLoc_sectionPoint, 1, glm::value_ptr(m_sectionPoint));
    glUniform3fv(m_meshLoc_sectionNormal, 1, glm::value_ptr(m_sectionNormal));

    // First pass: render all selected meshes with stencil write
    // Then render the outline for selected meshes
    // Finally render all meshes normally

    // --- Outline pass (stencil technique) ---
    // Selected meshes get the blue selection outline; subtract previews get a
    // red one so it reads as "this volume will be removed".
    static const glm::vec4 kSubtractOutline(0.95f, 0.12f, 0.12f, 1.0f);
    for (const auto& mesh : m_meshes) {
        if (mesh.vertexCount == 0) continue; // vacant slot (removeBody)
        if (mesh.selected) renderMeshOutline(mesh, view, projection, m_outlineColor);
        else if (mesh.subtractPreview) renderMeshOutline(mesh, view, projection, kSubtractOutline);
    }

    // --- Main render pass ---
    glUseProgram(m_meshProgram);
    glUniformMatrix4fv(m_meshLoc_view, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(m_meshLoc_projection, 1, GL_FALSE, glm::value_ptr(projection));
    glUniform3fv(m_meshLoc_viewPos, 1, glm::value_ptr(viewPos));
    glUniform3fv(m_meshLoc_lightDir, 1, glm::value_ptr(m_lightDir));
    glUniform3fv(m_meshLoc_fillDir, 1, glm::value_ptr(m_fillDir));
    glUniform1f(m_meshLoc_ambient, m_lighting.ambient);
    glUniform1i(m_meshLoc_headlight, m_lighting.headlight ? 1 : 0);
    glUniform1f(m_meshLoc_fillStrength, m_lighting.fill ? m_lighting.fillStrength : 0.0f);
    glActiveTexture(GL_TEXTURE0 + kProbeTextureUnit);
    glBindTexture(GL_TEXTURE_CUBE_MAP, m_probeActive ? m_probeTex : 0);
    glActiveTexture(GL_TEXTURE0);
    glUniform1i(m_meshLoc_envMap, kProbeTextureUnit);
    glUniform1i(m_meshLoc_hasEnv, m_probeActive ? 1 : 0);
    glUniform1f(m_meshLoc_envMaxLod, m_probeMaxLod);
    glActiveTexture(GL_TEXTURE0 + kPhotoTextureUnit);
    glBindTexture(GL_TEXTURE_2D, m_photoTex);
    glActiveTexture(GL_TEXTURE0);
    glUniform1i(m_meshLoc_photoEnv, kPhotoTextureUnit);
    glUniform1i(m_meshLoc_usePhotoEnv, usePhotoEnvironment() ? 1 : 0);

    // Coincident-face tie-break, third generation. Overlapping bodies can
    // share exactly coplanar faces (Extrude From leaves the new body's base on
    // the source face) and shimmer as they z-fight. 878b7ca resolved that with
    // a per-body glPolygonOffset(rank, 2*rank) - but rank grew with BODY COUNT,
    // and the slope term multiplies each face's per-pixel depth gradient, so on
    // a 68-body assembly old bodies drew centimetres behind their true position
    // (worse zoomed out, where the per-pixel gradient grows). Their GL_LINES
    // edges - which polygon offset cannot move - stayed put, so whole bodies
    // degenerated into edges floating over displaced or invisible faces (the
    // corvus ghost bug). Clamping the rank only shrank the wedge: faces still
    // visibly recessed close up, and interior edges still popped through thin
    // skins far out. Any face-only offset drives that wedge; the offset had to
    // go entirely.
    //
    // Instead: bodies draw in slot order (creation order, oldest first) with
    // GL_LEQUAL, so wherever two faces produce EQUAL depth the newest body
    // deterministically wins by drawing last - the exact semantics the offset
    // was built for, with zero geometric displacement. The invariant vertex
    // shaders (shared with EdgeRenderer) make coplanar faces genuinely hit
    // equal depths. Residual risk: differently-tessellated coplanar faces can
    // still disagree by a few depth quanta at extreme grazing angles and
    // sparkle there - minor and localized, where the offset's failure mode was
    // structural and scene-wide.
    glDepthFunc(GL_LEQUAL);
    const int slotCount = static_cast<int>(m_meshes.size());
    for (int slot = 0; slot < slotCount; ++slot) {
        const auto& mesh = m_meshes[slot];
        if (mesh.vertexCount == 0) continue; // vacant slot
        glUniformMatrix4fv(m_meshLoc_model, 1, GL_FALSE, glm::value_ptr(mesh.modelMatrix));
        glUniform3fv(m_meshLoc_objectColor, 1, glm::value_ptr(mesh.color));
        const SurfaceFinish& fin = mesh.finish;
        glUniform1f(m_meshLoc_specStrength, fin.specStrength);
        glUniform1f(m_meshLoc_shininess, fin.shininess);
        glUniform1f(m_meshLoc_metallic, fin.metallic);
        glUniform1f(m_meshLoc_roughness, fin.roughness);
        glUniform1f(m_meshLoc_clearcoat, fin.clearcoat);
        glUniform1f(m_meshLoc_grain, fin.grain);
        if (fin.grain > 0.0f) {
            // The grain runs along the longest side; the pith sits off the
            // shortest, the way a board is sawn. Ring spacing follows the
            // part's size within real-timber bounds, so a small part still
            // shows a few rings and a long one doesn't turn into pinstripes.
            const glm::vec3 ext = mesh.bbMax - mesh.bbMin;
            int lo = 0, hi = 0;
            for (int k = 1; k < 3; ++k) {
                if (ext[k] > ext[hi]) hi = k;
                if (ext[k] < ext[lo]) lo = k;
            }
            if (lo == hi) lo = (hi + 1) % 3;
            glm::vec3 axis(0.0f), side(0.0f);
            axis[hi] = 1.0f;
            side[lo] = 1.0f;
            const glm::vec3 center = 0.5f * (mesh.bbMin + mesh.bbMax);
            glUniform3fv(m_meshLoc_grainCenter, 1, glm::value_ptr(center));
            glUniform3fv(m_meshLoc_grainAxis, 1, glm::value_ptr(axis));
            glUniform3fv(m_meshLoc_grainSide, 1, glm::value_ptr(side));
            glUniform1f(m_meshLoc_grainScale, glm::clamp(ext[hi] / 30.0f, 0.6f, 3.5f));
        }
        glUniform1i(m_meshLoc_selected, mesh.selected ? 1 : 0);
        glUniform1i(m_meshLoc_previewCut, mesh.subtractPreview ? 1 : 0);

        glBindVertexArray(mesh.vao);
        glDrawArrays(GL_TRIANGLES, 0, mesh.vertexCount);
    }
    glDepthFunc(GL_LESS);   // restore the default for every later pass

    glActiveTexture(GL_TEXTURE0 + kProbeTextureUnit);
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    glActiveTexture(GL_TEXTURE0 + kPhotoTextureUnit);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(0);
    glUseProgram(0);
}

// The scene as a metal part would see it: a cubemap of every non-metal body,
// the floor grid and the studio, taken from the middle of the metal bodies.
// Metals are left out of it - a probe sitting inside a metal part would see
// only that part's inside - so they reflect everything except each other.
// One probe for the scene, so the reflection has no parallax; on parts the
// size of a desk's worth of CAD that reads as real.
//
// Rebuilt only when what it shows changes (a signature over the drawn
// slots), at most every kProbeMinInterval seconds: a drag that moves a body
// every frame refreshes it about ten times a second rather than costing six
// extra scene passes per frame. reflectionPending() tells the caller to keep
// frames coming until the throttled rebuild lands.
void ShapeRenderer::updateReflectionProbe()
{
    m_probePending = false;
    if (!m_probeBgProgram || m_probeFailed) { m_probeActive = false; return; }

    std::uint64_t sig = 1469598103934665603ull;
    auto hashIn = [&sig](const void* p, size_t n) {
        const unsigned char* b = static_cast<const unsigned char*>(p);
        for (size_t i = 0; i < n; ++i) { sig ^= b[i]; sig *= 1099511628211ull; }
    };
    glm::vec3 metalSum(0.0f), lo(1e30f), hi(-1e30f);
    int metals = 0;
    for (const MeshData& m : m_meshes) {
        if (m.vertexCount == 0) continue;
        const glm::vec3 half = 0.5f * (m.bbMax - m.bbMin);
        const glm::vec3 c = glm::vec3(m.modelMatrix * glm::vec4(m.bbMin + half, 1.0f));
        lo = glm::min(lo, c - half);
        hi = glm::max(hi, c + half);
        if (m.finish.metallic > 0.0f) {
            metalSum += c;
            ++metals;
            continue;
        }
        hashIn(&m.vao, sizeof m.vao);
        hashIn(&m.vertexCount, sizeof m.vertexCount);
        hashIn(&m.color, sizeof m.color);
        hashIn(&m.modelMatrix, sizeof m.modelMatrix);
        hashIn(&m.finish, sizeof m.finish);
    }
    if (metals == 0) { m_probeActive = false; return; }
    const glm::vec3 center = metalSum / static_cast<float>(metals);
    hashIn(&center, sizeof center);
    const bool photo = usePhotoEnvironment();
    hashIn(&photo, sizeof photo);

    if (m_probeActive && sig == m_probeSig) return;
    const double now = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    if (m_probeActive && now - m_probeBuiltAt < kProbeMinInterval) {
        m_probePending = true;   // keep the stale one a moment longer
        return;
    }

    if (!m_probeTex) {
        glGenTextures(1, &m_probeTex);
        glBindTexture(GL_TEXTURE_CUBE_MAP, m_probeTex);
        for (int f = 0; f < 6; ++f)
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + f, 0, GL_RGBA8, kProbeSize, kProbeSize,
                         0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
        int levels = 0;
        for (int s2 = kProbeSize; s2 > 1; s2 /= 2) ++levels;
        m_probeMaxLod = static_cast<float>(levels);
#ifdef GL_TEXTURE_CUBE_MAP_SEAMLESS
        // Desktop GL filters each face alone unless told otherwise, which
        // shows the cube's seams in a rough reflection. GLES 3 is always
        // seamless.
        glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
#endif
        glGenRenderbuffers(1, &m_probeDepth);
        glBindRenderbuffer(GL_RENDERBUFFER, m_probeDepth);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, kProbeSize, kProbeSize);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
        glGenFramebuffers(1, &m_probeFbo);
        glGenVertexArrays(1, &m_probeVao);
    }

    GLint prevFbo = 0;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prevFbo);
    GLint prevViewport[4];
    glGetIntegerv(GL_VIEWPORT, prevViewport);
    const GLboolean cullWas = glIsEnabled(GL_CULL_FACE);
    const GLboolean scissorWas = glIsEnabled(GL_SCISSOR_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_SCISSOR_TEST);

    glBindFramebuffer(GL_FRAMEBUFFER, m_probeFbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_probeDepth);
    glViewport(0, 0, kProbeSize, kProbeSize);

    const float diag = std::max(glm::length(hi - lo), 1.0f);
    const glm::mat4 proj = glm::perspective(glm::radians(90.0f), 1.0f,
                                            diag * 1e-4f, diag * 8.0f);
    // The standard cubemap face frames: looking down each axis with the up
    // vector the GL face layout expects.
    static const glm::vec3 kDirs[6] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0},
                                       {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    static const glm::vec3 kUps[6] = {{0, -1, 0}, {0, -1, 0}, {0, 0, 1},
                                      {0, 0, -1}, {0, -1, 0}, {0, -1, 0}};
    glActiveTexture(GL_TEXTURE0 + kPhotoTextureUnit);
    glBindTexture(GL_TEXTURE_2D, m_photoTex);
    glActiveTexture(GL_TEXTURE0);
    bool complete = true;
    for (int f = 0; f < 6 && complete; ++f) {
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_CUBE_MAP_POSITIVE_X + f, m_probeTex, 0);
        if (f == 0 && glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            complete = false;
            break;
        }
        const glm::mat4 view = glm::lookAt(center, center + kDirs[f], kUps[f]);

        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
        glUseProgram(m_probeBgProgram);
        const glm::mat4 inv = glm::inverse(proj * view);
        glUniformMatrix4fv(m_probeLoc_invViewProj, 1, GL_FALSE, glm::value_ptr(inv));
        glUniform3fv(m_probeLoc_probePos, 1, glm::value_ptr(center));
        glUniform1f(m_probeLoc_gridFade, std::max(diag * 2.0f, 200.0f));
        glUniform1i(m_probeLoc_photoEnv, kPhotoTextureUnit);
        glUniform1i(m_probeLoc_usePhotoEnv, photo ? 1 : 0);
        glBindVertexArray(m_probeVao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glDepthMask(GL_TRUE);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glClear(GL_DEPTH_BUFFER_BIT);

        glUseProgram(m_meshProgram);
        glUniformMatrix4fv(m_meshLoc_view, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(m_meshLoc_projection, 1, GL_FALSE, glm::value_ptr(proj));
        glUniform3fv(m_meshLoc_viewPos, 1, glm::value_ptr(center));
        glUniform3fv(m_meshLoc_lightDir, 1, glm::value_ptr(m_lightDir));
        glUniform3fv(m_meshLoc_fillDir, 1, glm::value_ptr(m_fillDir));
        glUniform1f(m_meshLoc_ambient, m_lighting.ambient);
        glUniform1i(m_meshLoc_headlight, 0);   // no camera here; light it like the studio
        glUniform1f(m_meshLoc_fillStrength, m_lighting.fill ? m_lighting.fillStrength : 0.0f);
        glUniform1i(m_meshLoc_sectionEnabled, 0);
        glUniform1i(m_meshLoc_hasEnv, 0);
        glUniform1i(m_meshLoc_photoEnv, kPhotoTextureUnit);
        glUniform1i(m_meshLoc_usePhotoEnv, photo ? 1 : 0);
        glUniform1i(m_meshLoc_selected, 0);
        glUniform1i(m_meshLoc_previewCut, 0);
        for (const MeshData& m : m_meshes) {
            if (m.vertexCount == 0 || m.finish.metallic > 0.0f) continue;
            const SurfaceFinish& fin = m.finish;
            glUniformMatrix4fv(m_meshLoc_model, 1, GL_FALSE, glm::value_ptr(m.modelMatrix));
            glUniform3fv(m_meshLoc_objectColor, 1, glm::value_ptr(m.color));
            glUniform1f(m_meshLoc_specStrength, fin.specStrength);
            glUniform1f(m_meshLoc_shininess, fin.shininess);
            glUniform1f(m_meshLoc_metallic, 0.0f);
            glUniform1f(m_meshLoc_roughness, fin.roughness);
            glUniform1f(m_meshLoc_clearcoat, fin.clearcoat);
            glUniform1f(m_meshLoc_grain, 0.0f);   // too fine to survive a reflection
            glBindVertexArray(m.vao);
            glDrawArrays(GL_TRIANGLES, 0, m.vertexCount);
        }
    }
    glBindVertexArray(0);
    glActiveTexture(GL_TEXTURE0 + kPhotoTextureUnit);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);

    glBindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(prevFbo));
    glViewport(prevViewport[0], prevViewport[1], prevViewport[2], prevViewport[3]);
    if (cullWas) glEnable(GL_CULL_FACE);
    if (scissorWas) glEnable(GL_SCISSOR_TEST);

    if (!complete) {
        // An implementation that cannot render to a cube face: keep the
        // analytic studio for good rather than retrying every frame.
        std::fprintf(stderr, "[Render] reflection probe framebuffer incomplete - using the studio fallback\n");
        m_probeFailed = true;
        m_probeActive = false;
        return;
    }
    glBindTexture(GL_TEXTURE_CUBE_MAP, m_probeTex);
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    m_probeSig = sig;
    m_probeBuiltAt = now;
    m_probeActive = true;
}

void ShapeRenderer::renderMeshOutline(const MeshData& mesh, const glm::mat4& view,
                                      const glm::mat4& projection, const glm::vec4& color)
{
    // Stencil technique:
    // 1. Draw the mesh into stencil buffer (marking pixels with 1)
    // 2. Draw the expanded outline where stencil != 1

    glEnable(GL_STENCIL_TEST);

    // Step 1: Fill stencil with 1 where the mesh is drawn
    glStencilFunc(GL_ALWAYS, 1, 0xFF);
    glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
    glStencilMask(0xFF);
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    glDepthMask(GL_FALSE);

    glUseProgram(m_meshProgram);
    glUniformMatrix4fv(m_meshLoc_model, 1, GL_FALSE, glm::value_ptr(mesh.modelMatrix));
    glUniformMatrix4fv(m_meshLoc_view, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(m_meshLoc_projection, 1, GL_FALSE, glm::value_ptr(projection));

    glBindVertexArray(mesh.vao);
    glDrawArrays(GL_TRIANGLES, 0, mesh.vertexCount);

    // Step 2: Draw outline where stencil != 1
    glStencilFunc(GL_NOTEQUAL, 1, 0xFF);
    glStencilMask(0x00);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthMask(GL_TRUE);

    glUseProgram(m_outlineProgram);
    glUniformMatrix4fv(m_outlineLoc_model, 1, GL_FALSE, glm::value_ptr(mesh.modelMatrix));
    glUniformMatrix4fv(m_outlineLoc_view, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(m_outlineLoc_projection, 1, GL_FALSE, glm::value_ptr(projection));
    glUniform4fv(m_outlineLoc_outlineColor, 1, glm::value_ptr(color));
    glUniform1f(m_outlineLoc_outlineWidth, m_outlineWidth);

    glDrawArrays(GL_TRIANGLES, 0, mesh.vertexCount);
    glBindVertexArray(0);

    // Reset stencil state
    glStencilMask(0xFF);
    glStencilFunc(GL_ALWAYS, 0, 0xFF);
    glDisable(GL_STENCIL_TEST);
    glUseProgram(0);
}

void ShapeRenderer::setModelMatrix(int meshIndex, const glm::mat4& model)
{
    if (meshIndex >= 0 && meshIndex < static_cast<int>(m_meshes.size())) {
        m_meshes[meshIndex].modelMatrix = model;
    }
}

void ShapeRenderer::setColor(int meshIndex, glm::vec3 color)
{
    if (meshIndex >= 0 && meshIndex < static_cast<int>(m_meshes.size())) {
        m_meshes[meshIndex].color = color;
    }
}

void ShapeRenderer::setFinish(int meshIndex, const SurfaceFinish& finish)
{
    if (meshIndex >= 0 && meshIndex < static_cast<int>(m_meshes.size())) {
        m_meshes[meshIndex].finish = finish;
    }
}

void ShapeRenderer::setSelected(int meshIndex, bool selected)
{
    if (meshIndex >= 0 && meshIndex < static_cast<int>(m_meshes.size())) {
        m_meshes[meshIndex].selected = selected;
    }
}

void ShapeRenderer::setSubtractPreview(int meshIndex, bool subtractPreview)
{
    if (meshIndex >= 0 && meshIndex < static_cast<int>(m_meshes.size())) {
        m_meshes[meshIndex].subtractPreview = subtractPreview;
    }
}

void ShapeRenderer::retireAll()
{
    // Keep the buffers: the full rebuild that follows may hand the same
    // shapes straight back (see setBodyMesh). Anything retired last time
    // and never reclaimed dies here.
    freeRetired();
    m_retired.swap(m_meshes);
    m_meshes.clear();
    m_bodyToSlot.clear();
    m_meshedAtPrev.swap(m_meshedAt);
    m_meshedAt.clear();
}

void ShapeRenderer::clear()
{
    retireAll();
    freeRetired();
}

void ShapeRenderer::freeRetired()
{
    for (auto& mesh : m_retired) {
        if (mesh.vao) glDeleteVertexArrays(1, &mesh.vao);
        if (mesh.vbo) glDeleteBuffers(1, &mesh.vbo);
    }
    m_retired.clear();
}

void ShapeRenderer::debugDumpSlots() const
{
    std::fprintf(stderr, "  [Slots] %zu mesh slots:\n", m_meshes.size());
    for (size_t i = 0; i < m_meshes.size(); ++i) {
        const auto& m = m_meshes[i];
        std::fprintf(stderr,
                     "    slot %zu: body=%d verts=%d sel=%d subPrev=%d\n",
                     i, m.bodyId, m.vertexCount, m.selected ? 1 : 0,
                     m.subtractPreview ? 1 : 0);
    }
}

glm::vec3 ShapeRenderer::bodyColor(int index)
{
    static const glm::vec3 palette[] = {
        {0.60f, 0.65f, 0.75f},  // steel blue
        {0.75f, 0.55f, 0.40f},  // copper
        {0.50f, 0.70f, 0.50f},  // sage green
        {0.70f, 0.50f, 0.65f},  // mauve
        {0.65f, 0.65f, 0.45f},  // olive
        {0.45f, 0.60f, 0.70f},  // teal
        {0.72f, 0.58f, 0.55f},  // salmon
        {0.55f, 0.55f, 0.70f},  // lavender
    };
    static const int paletteSize = static_cast<int>(sizeof(palette) / sizeof(palette[0]));

    int i = index % paletteSize;
    if (i < 0) i += paletteSize;
    return palette[i];
}

bool ShapeRenderer::compileShader(unsigned int& shader, unsigned int type, const char* source)
{
    shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    int success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::fprintf(stderr, "ShapeRenderer shader compilation failed: %s\n", infoLog);
        glDeleteShader(shader);
        shader = 0;
        return false;
    }
    return true;
}

bool ShapeRenderer::linkProgram(unsigned int program, unsigned int vertShader,
                                unsigned int fragShader)
{
    glAttachShader(program, vertShader);
    glAttachShader(program, fragShader);
    glLinkProgram(program);

    int success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::fprintf(stderr, "ShapeRenderer shader linking failed: %s\n", infoLog);
        return false;
    }
    return true;
}

} // namespace materializr
