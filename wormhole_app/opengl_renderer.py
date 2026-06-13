from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from .config import RenderConfig


VERTEX_SHADER = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""


FRAGMENT_SHADER = """
#version 330
uniform sampler2D entrance_tex;
uniform sampler2D exit_tex;
uniform vec2 resolution;
uniform int frame_index;
uniform int frame_count;
uniform int geodesic_steps;
uniform int aa_samples;
uniform int high_order_filter;
uniform float rho;
uniform float a_param;
uniform float mass_param;
uniform float lensing_width;
uniform float camera_distance;
uniform float celestial_distance;
uniform float fov_degrees;
uniform float turn_fraction;
in vec2 v_uv;
out vec4 fragColor;

const float PI = 3.141592653589793;

float smoothstep01(float e0, float e1, float x) {
    float t = clamp((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

float shape_radius(float l_coord) {
    float abs_l = abs(l_coord);
    float x = 2.0 * (abs_l - a_param) / (PI * max(lensing_width, 1e-6));
    float exterior = rho + lensing_width * (x * atan(x) - 0.5 * log(1.0 + x * x));
    return abs_l < a_param ? rho : exterior;
}

float integrate_phi(float start_l, float end_l, float impact) {
    int count = max(geodesic_steps, 8);
    float total = 0.0;
    float span_l = end_l - start_l;
    float abs_step = abs(span_l) / float(count);
    for (int i = 0; i < 4096; i++) {
        if (i >= count) break;
        float t = (float(i) + 0.5) / float(count);
        float l_mid = start_l + span_l * t;
        float r_mid = max(shape_radius(l_mid), rho);
        float radial_sq = max(1.0 - (impact / max(r_mid, 1e-6)) * (impact / max(r_mid, 1e-6)), 1e-7);
        total += impact / max(r_mid * r_mid * sqrt(radial_sq), 1e-6) * abs_step;
    }
    return total;
}

float turning_abs_l(float impact, float camera_abs_l) {
    float low = a_param;
    float high = max(camera_abs_l, a_param);
    for (int i = 0; i < 32; i++) {
        float mid = (low + high) * 0.5;
        float r_mid = shape_radius(mid);
        if (r_mid >= impact) high = mid;
        else low = mid;
    }
    return high;
}

vec2 pano_uv(vec3 dir) {
    dir = normalize(dir);
    float u = atan(dir.x, dir.z) / (2.0 * PI) + 0.5;
    float v = 0.5 - asin(clamp(dir.y, -1.0, 1.0)) / PI;
    return vec2(u, 1.0 - v);
}

vec3 sample_pano(sampler2D tex, vec3 dir) {
    return texture(tex, pano_uv(dir)).rgb;
}

vec3 sample_pano_filtered(sampler2D tex, vec3 dir, float phi) {
    vec2 uv = pano_uv(dir);
    float weight = smoothstep01(PI * 1.4, PI * 4.5, abs(phi));
    if (high_order_filter == 0 || weight <= 0.001) {
        return texture(tex, uv).rgb;
    }
    vec3 sharp = texture(tex, uv).rgb;
    vec3 blurred = textureLod(tex, uv, 3.0 + 3.0 * weight).rgb;
    return mix(sharp, blurred, weight) * (1.0 - 0.28 * weight);
}

mat3 look_at(vec3 position, vec3 target) {
    vec3 forward = normalize(target - position);
    vec3 world_up = vec3(0.0, 1.0, 0.0);
    if (abs(dot(forward, world_up)) > 0.98) world_up = vec3(1.0, 0.0, 0.0);
    vec3 right = normalize(cross(world_up, forward));
    vec3 up = cross(forward, right);
    return mat3(right, up, forward);
}

void camera_pose(out vec3 position, out vec3 target) {
    float t = float(frame_index) / max(float(frame_count - 1), 1.0);
    float turn_start = 1.0 - turn_fraction;
    if (t < turn_start) {
        float q = smoothstep01(0.0, turn_start, t);
        float z = -camera_distance * (1.0 - q) + (a_param + camera_distance * 0.42) * q;
        position = vec3(0.0, 0.0, z);
        target = vec3(0.0, 0.0, z + 5.0);
    } else {
        float q = smoothstep01(turn_start, 1.0, t);
        float z = a_param + camera_distance * 0.42;
        float orbit_radius = rho * 0.36;
        position = vec3(sin(q * PI) * orbit_radius, sin(q * PI * 0.75) * orbit_radius * 0.35, z);
        target = vec3(0.0, 0.0, a_param - (1.5 + camera_distance * q));
    }
}

vec3 slerp_forward(vec3 a, vec3 b, float t) {
    a = normalize(a);
    b = normalize(b);
    float d = clamp(dot(a, b), -1.0, 1.0);
    if (d < -0.999) {
        vec3 axis = normalize(cross(a, vec3(0.0, 1.0, 0.0)));
        if (length(axis) < 1e-5) axis = normalize(cross(a, vec3(1.0, 0.0, 0.0)));
        float angle = PI * t;
        return normalize(a * cos(angle) + cross(axis, a) * sin(angle));
    }
    if (d > 0.999) return normalize(mix(a, b, t));
    float theta = acos(d);
    float st = sin(theta);
    return normalize((sin((1.0 - t) * theta) * a + sin(t * theta) * b) / st);
}

void camera_pose_continuous(out vec3 position, out mat3 rot) {
    float t = float(frame_index) / max(float(frame_count - 1), 1.0);
    float turn_start = 1.0 - turn_fraction;
    float exit_z = a_param + camera_distance * 0.42;
    if (t < turn_start) {
        float q = smoothstep01(0.0, turn_start, t);
        float z = -camera_distance * (1.0 - q) + exit_z * q;
        position = vec3(0.0, 0.0, z);
        vec3 forward = vec3(0.0, 0.0, 1.0);
        vec3 right = vec3(1.0, 0.0, 0.0);
        vec3 up = vec3(0.0, 1.0, 0.0);
        rot = mat3(right, up, forward);
    } else {
        float q = smoothstep01(turn_start, 1.0, t);
        float orbit_radius = rho * 0.36;
        position = vec3(sin(q * PI) * orbit_radius, 0.0, exit_z);
        float angle = PI * q;
        vec3 forward = normalize(vec3(sin(angle), 0.0, cos(angle)));
        vec3 up = vec3(0.0, 1.0, 0.0);
        vec3 right = normalize(cross(up, forward));
        rot = mat3(right, up, forward);
    }
}

vec3 trace_color(vec3 ray, float camera_l) {
    float sphere_l = max(celestial_distance, camera_distance + 2.0 * a_param + 1.0);
    float r0 = shape_radius(camera_l);
    float mu0 = clamp(ray.z, -1.0, 1.0);
    float impact = clamp(r0 * sqrt(max(0.0, 1.0 - mu0 * mu0)), 0.0, r0);
    vec2 transverse = ray.xy;
    vec2 transverse_unit = length(transverse) > 1e-8 ? normalize(transverse) : vec2(1.0, 0.0);
    float direction = mu0 >= 0.0 ? 1.0 : -1.0;
    float end_l = 0.0;
    float phi = 0.0;

    if (abs(camera_l) >= a_param) {
        float camera_side = camera_l >= 0.0 ? 1.0 : -1.0;
        bool moving_toward = direction * camera_side < 0.0;
        bool passes = impact < rho * (1.0 - 1e-5);
        if (!moving_toward) {
            end_l = camera_side * sphere_l;
            phi = integrate_phi(camera_l, end_l, impact);
        } else if (passes) {
            end_l = -camera_side * sphere_l;
            phi = integrate_phi(camera_l, end_l, impact);
        } else {
            float turn_l = camera_side * turning_abs_l(impact, abs(camera_l));
            end_l = camera_side * sphere_l;
            phi = integrate_phi(camera_l, turn_l, impact) + integrate_phi(turn_l, end_l, impact);
        }
    } else {
        end_l = direction * sphere_l;
        phi = integrate_phi(camera_l, end_l, impact);
    }

    bool exit_side = end_l >= 0.0;
    vec3 sample_dir = normalize(vec3(transverse_unit * sin(phi), (exit_side ? 1.0 : -1.0) * cos(phi)));
    return exit_side ? sample_pano_filtered(exit_tex, sample_dir, phi) : sample_pano_filtered(entrance_tex, sample_dir, phi);
}

void main() {
    vec3 position;
    mat3 rot;
    camera_pose_continuous(position, rot);
    float aspect = resolution.x / resolution.y;
    float fov = radians(fov_degrees);
    vec3 color = vec3(0.0);

    int samples = aa_samples <= 1 ? 1 : (aa_samples <= 4 ? 4 : 9);
    for (int s = 0; s < 9; s++) {
        if (s >= samples) break;
        vec2 off = vec2(0.0);
        if (samples == 4) {
            off = vec2((s == 0 || s == 2) ? -0.25 : 0.25, (s < 2) ? -0.25 : 0.25);
        } else if (samples == 9) {
            off = vec2(float(s % 3) - 1.0, float(s / 3) - 1.0) * 0.33;
        }
        vec2 p = ((gl_FragCoord.xy + off) / resolution) * 2.0 - 1.0;
        p.x *= aspect;
        vec3 local_ray = normalize(vec3(p.x, p.y, 1.0 / tan(fov * 0.5)));
        vec3 ray = normalize(rot * local_ray);
        color += trace_color(ray, position.z);
    }
    color /= float(samples);
    vec2 centered = (gl_FragCoord.xy - resolution * 0.5) / resolution.y;
    float radial = length(centered);
    float vignette = 1.0 - clamp((radial - 0.42) / 0.38, 0.0, 1.0) * 0.38;
    fragColor = vec4(clamp(color * vignette, 0.0, 1.0), 1.0);
}
"""


def is_opengl_available() -> tuple[bool, str]:
    try:
        import moderngl  # type: ignore

        ctx = moderngl.create_standalone_context()
        return True, ctx.info.get("GL_RENDERER", "OpenGL")
    except Exception as exc:
        return False, str(exc)


class OpenGLFrameRenderer:
    def __init__(self, cfg: RenderConfig) -> None:
        import moderngl  # type: ignore

        self.moderngl = moderngl
        self.cfg = cfg
        self.ctx = moderngl.create_standalone_context()
        self.program = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        self.vbo = self.ctx.buffer(quad.tobytes())
        self.vao = self.ctx.vertex_array(self.program, [(self.vbo, "2f", "in_pos")])
        self.color_tex = self.ctx.texture((cfg.width, cfg.height), 4)
        self.fbo = self.ctx.framebuffer(color_attachments=[self.color_tex])
        self.entrance_tex = self._load_texture(cfg.entrance_texture, (24, 36, 82))
        self.exit_tex = self._load_texture(cfg.exit_texture, (78, 38, 70))
        self.entrance_tex.use(0)
        self.exit_tex.use(1)
        self.program["entrance_tex"].value = 0
        self.program["exit_tex"].value = 1

    def _load_texture(self, path: Path, fallback_tint: tuple[int, int, int]):
        if path.exists():
            image = Image.open(path).convert("RGB")
        else:
            arr = np.zeros((1024, 2048, 3), dtype=np.uint8)
            arr[..., 0] = fallback_tint[0]
            arr[..., 1] = fallback_tint[1]
            arr[..., 2] = fallback_tint[2]
            image = Image.fromarray(arr, "RGB")
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        tex = self.ctx.texture(image.size, 3, image.tobytes())
        tex.filter = (self.moderngl.LINEAR, self.moderngl.LINEAR)
        tex.repeat_x = True
        tex.repeat_y = False
        tex.build_mipmaps()
        tex.filter = (self.moderngl.LINEAR_MIPMAP_LINEAR, self.moderngl.LINEAR)
        return tex

    def render_frame(self, frame: int) -> Image.Image:
        cfg = self.cfg
        self.fbo.use()
        self.ctx.viewport = (0, 0, cfg.width, cfg.height)
        self.program["resolution"].value = (float(cfg.width), float(cfg.height))
        self.program["frame_index"].value = int(frame)
        self.program["frame_count"].value = int(cfg.frames)
        self.program["geodesic_steps"].value = int(min(max(cfg.geodesic_steps, 8), 4096))
        self.program["aa_samples"].value = int(cfg.antialias_samples)
        self.program["high_order_filter"].value = 1 if cfg.high_order_filter else 0
        self.program["rho"].value = float(max(cfg.rho, 0.05))
        self.program["a_param"].value = float(max(cfg.a, 0.001))
        if "mass_param" in self.program:
            self.program["mass_param"].value = float(max(cfg.mass_parameter, 0.001))
        self.program["lensing_width"].value = float(max(cfg.lensing_width, 0.001))
        self.program["camera_distance"].value = float(max(cfg.camera_distance, 0.1))
        self.program["celestial_distance"].value = float(max(cfg.celestial_distance, 5.0))
        self.program["fov_degrees"].value = float(cfg.fov_degrees)
        self.program["turn_fraction"].value = float(cfg.turn_fraction)
        self.vao.render(self.moderngl.TRIANGLE_STRIP)
        data = self.fbo.read(components=3, alignment=1)
        image = Image.frombytes("RGB", (cfg.width, cfg.height), data)
        return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
