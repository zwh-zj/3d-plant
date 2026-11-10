#include <cstdio>
#include <vector>
#include <algorithm>
#include <math.h>
using namespace std;

// ===================== tunable parameters =====================
const float AMBIENT = 0.35f;       // ambient light; keep dark regions visible
const float DIFFUSE = 0.65f;       // diffuse strength; shading contrast
const float LIGHT_DIR[3] = {0.3f, 0.3f, 0.9f};  // light from upper-front
const bool ENABLE_AA = true;       // edge antialiasing on sphere silhouette
// ========================================================

struct PointPattern {
    int dx, dy;     // pixel offset
    float dz;       // z depth offset
    float lambert;  // precomputed Lambert term
    float alpha;    // AA alpha
};

extern "C" {

void render_ball(int h, int w, unsigned char * show, int n, int * xyzs,
                 float * c0, float * c1, float * c2, int r) {
    r = max(r, 1);
    vector<int> depth(h * w, -2100000000);
    vector<PointPattern> pattern;

    // ---------- 1. precompute sphere splat (lighting + AA) ----------
    // normalize light direction
    float l_len = sqrtf(LIGHT_DIR[0]*LIGHT_DIR[0] + LIGHT_DIR[1]*LIGHT_DIR[1] + LIGHT_DIR[2]*LIGHT_DIR[2]);
    float lx = LIGHT_DIR[0] / l_len;
    float ly = LIGHT_DIR[1] / l_len;
    float lz = LIGHT_DIR[2] / l_len;

    float r_f = (float)r;
    float r_sq = r_f * r_f;

    for (int dx = -r; dx <= r; dx++) {
        for (int dy = -r; dy <= r; dy++) {
            float dist = sqrtf((float)(dx*dx + dy*dy));
            if (dist > r_f) continue;

            float dz = sqrtf(r_sq - dx*dx - dy*dy);
            PointPattern p;
            p.dx = dx;
            p.dy = dy;
            p.dz = dz;

            // Lambert: surface normal dot light
            float nx = dx / r_f;
            float ny = dy / r_f;
            float nz = dz / r_f;
            float diffuse = max(0.0f, nx*lx + ny*ly + nz*lz);
            p.lambert = AMBIENT + diffuse * DIFFUSE;

            // AA: 1px linear fade at the silhouette
            if (ENABLE_AA && dist > r_f - 1.0f) {
                p.alpha = r_f - dist;
            } else {
                p.alpha = 1.0f;
            }

            pattern.push_back(p);
        }
    }

    // ---------- 2. global depth range (depth fade for depth cue) ----------
    float zmin = 0.0f, zmax = 0.0f;
    for (int i = 0; i < n; i++) {
        float z = (float)xyzs[i*3 + 2];
        if (i == 0) {
            zmin = z - r_f;
            zmax = z + r_f;
        } else {
            zmin = min(zmin, z - r_f);
            zmax = max(zmax, z + r_f);
        }
    }
    float z_range = zmax - zmin;
    if (z_range < 1e-6f) z_range = 1.0f; // avoid divide-by-zero

    // ---------- 3. per-point splat (depth test + blend) ----------
    for (int i = 0; i < n; i++) {
        int x = xyzs[i*3 + 0];
        int y = xyzs[i*3 + 1];
        int z = xyzs[i*3 + 2];

        // input: c0=R, c1=G, c2=B (fixes original R/G swap)
        float r_col = c0[i];
        float g_col = c1[i];
        float b_col = c2[i];

        for (int j = 0; j < (int)pattern.size(); j++) {
            int x2 = x + pattern[j].dx;
            int y2 = y + pattern[j].dy;
            float z2 = z + pattern[j].dz;

            // bounds check
            if (x2 < 0 || x2 >= h || y2 < 0 || y2 >= w) continue;

            int idx = x2 * w + y2;
            // depth test: nearer points occlude farther ones
            if (depth[idx] >= (int)z2) continue;
            depth[idx] = (int)z2;

            // brightness: Lambert * depth fade
            float depth_factor = (z2 - zmin) / z_range * 0.7f + 0.3f;
            float brightness = pattern[j].lambert * depth_factor;

            // foreground color in OpenCV BGR order
            float fb = b_col * brightness;
            float fg = g_col * brightness;
            float fr = r_col * brightness;

            int addr = idx * 3;
            float alpha = pattern[j].alpha;

            if (alpha >= 0.99f) {
                // opaque: write and clamp
                show[addr + 0] = (unsigned char)min(255.0f, max(0.0f, fb)); // B
                show[addr + 1] = (unsigned char)min(255.0f, max(0.0f, fg)); // G
                show[addr + 2] = (unsigned char)min(255.0f, max(0.0f, fr)); // R
            } else {
                // silhouette: alpha-blend with background
                float inv_alpha = 1.0f - alpha;
                float orig_b = (float)show[addr + 0];
                float orig_g = (float)show[addr + 1];
                float orig_r = (float)show[addr + 2];

                show[addr + 0] = (unsigned char)min(255.0f, max(0.0f, fb*alpha + orig_b*inv_alpha));
                show[addr + 1] = (unsigned char)min(255.0f, max(0.0f, fg*alpha + orig_g*inv_alpha));
                show[addr + 2] = (unsigned char)min(255.0f, max(0.0f, fr*alpha + orig_r*inv_alpha));
            }
        }
    }
}

} // extern "C"