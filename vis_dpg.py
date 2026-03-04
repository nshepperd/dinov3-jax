"""
Interactive DINOv3 Feature Similarity Visualizer using DearPyGui

Click anywhere on the image to see which patches have similar features.
Ctrl+V to paste an image from clipboard.
"""

import os

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"

import io
import re
import subprocess
import urllib.request
from dataclasses import dataclass

import dearpygui.dearpygui as dpg
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from PIL import Image

# JAX DINOv3 imports
from dinov3_jax import Dinov3VitModel, load_dinov3
from dinov3_jax.utils.pjit import pjit

# Configuration
PATCH_SIZE = 16
IMAGE_SIZE = 768  # Display size (height)
FEATURE_SCALE = 4.0  # Scale factor for feature extraction input (2.0 = 2x resolution feature map)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Paths - adjust these
IMAGE_PATH = ""
MODEL_PATH = "/data/models/dinov3-vitl16-pretrain-lvd1689m"


def preprocess_image(image: Image.Image, image_size: int, patch_size: int, feature_scale: float = 1.0):
    """Preprocess a PIL image for DINOv3.

    Returns display image and feature extraction tensor (possibly at different resolution).
    """
    w, h = image.size

    # Display image: sized to image_size height
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    display_h = h_patches * patch_size
    display_w = w_patches * patch_size
    image_display = image.resize((display_w, display_h), Image.Resampling.BILINEAR)

    # Feature extraction image: scaled up for higher-res feature map
    feat_h_patches = int(h_patches * feature_scale)
    feat_w_patches = int(w_patches * feature_scale)
    feat_h = feat_h_patches * patch_size
    feat_w = feat_w_patches * patch_size
    image_feat = image.resize((feat_w, feat_h), Image.Resampling.BILINEAR)

    # Convert feature image to tensor
    img_array = np.array(image_feat).astype(np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)  # CHW
    for c in range(3):
        img_array[c] = (img_array[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    img_tensor = jnp.array(img_array[np.newaxis, ...])

    return image_display, img_tensor


def load_and_preprocess_image(path: str, image_size: int, patch_size: int, feature_scale: float = 1.0):
    """Load image from file and preprocess for DINOv3."""
    image = Image.open(path).convert("RGB")
    return preprocess_image(image, image_size, patch_size, feature_scale)


def get_clipboard_text():
    """Get text from system clipboard."""
    for cmd in [
        ['wl-paste', '--no-newline'],
        ['xclip', '-selection', 'clipboard', '-o'],
    ]:
        try:
            print(f"[clipboard] trying text: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, timeout=5)
            print(f"[clipboard]   rc={result.returncode}, stdout={len(result.stdout)}B, stderr={result.stderr.decode(errors='ignore').strip()!r}")
            if result.returncode == 0 and result.stdout:
                text = result.stdout.decode('utf-8', errors='ignore').strip()
                print(f"[clipboard]   got text: {text[:200]!r}")
                return text
        except FileNotFoundError:
            print(f"[clipboard]   {cmd[0]} not found")
        except subprocess.TimeoutExpired:
            print(f"[clipboard]   timeout")
    return None


def get_clipboard_image():
    """Get image from system clipboard (supports Wayland and X11).

    Tries image data first, then falls back to checking if clipboard text is a URL.
    """
    # Try image data from clipboard
    for cmd in [
        ['wl-paste', '--type', 'image/png'],
        ['xclip', '-selection', 'clipboard', '-t', 'image/png', '-o'],
    ]:
        try:
            print(f"[clipboard] trying image: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, timeout=5)
            print(f"[clipboard]   rc={result.returncode}, stdout={len(result.stdout)}B, stderr={result.stderr.decode(errors='ignore').strip()!r}")
            if result.returncode == 0 and result.stdout:
                try:
                    img = Image.open(io.BytesIO(result.stdout)).convert("RGB")
                    print(f"[clipboard]   got image: {img.size}")
                    return img
                except Exception as e:
                    print(f"[clipboard]   failed to decode image: {e}")
        except FileNotFoundError:
            print(f"[clipboard]   {cmd[0]} not found")
        except subprocess.TimeoutExpired:
            print(f"[clipboard]   timeout")

    # Try clipboard text as URL
    print("[clipboard] no image data, trying text as URL...")
    text = get_clipboard_text()
    if text and re.match(r'https?://', text):
        print(f"[clipboard] fetching URL: {text[:200]}")
        try:
            req = urllib.request.Request(text, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
                print(f"[clipboard]   got {len(data)}B, content-type={resp.headers.get('Content-Type')}")
                img = Image.open(io.BytesIO(data)).convert("RGB")
                print(f"[clipboard]   decoded image: {img.size}")
                return img
        except Exception as e:
            print(f"[clipboard]   failed to fetch/decode URL: {e}")
    elif text:
        print(f"[clipboard] text is not a URL: {text[:200]!r}")

    return None


def extract_features(model, image_tensor, embed_dim):
    """Extract features from the model."""
    n_layers = len(model.layer)

    @pjit
    def fwd(model: Dinov3VitModel, image_tensor: Array) -> Array:
        features = model.get_intermediate_layers(
            image_tensor.astype(jnp.float32),
            n=n_layers,
            reshape=True,
            norm=True
        )
        x = features[-1] if isinstance(features, tuple) else features
        x = x.squeeze(0)
        if len(x.shape) == 3 and x.shape[0] == embed_dim:
            x = x.transpose(1, 2, 0)  # HWC
        return x

    return fwd(model, image_tensor)


def apply_colormap(values, cmap_name='viridis'):
    """Apply matplotlib colormap to normalized values."""
    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap(cmap_name)
    rgba = cmap(values)
    return (rgba[..., :3] * 255).astype(np.uint8)


class SimilarityVisualizer:
    def __init__(self, image: Image.Image, features: np.ndarray):
        self.image = image
        self.features = features
        self.H, self.W, self.C = features.shape
        self.img_w, self.img_h = image.size

        # Precompute normalized features for fast similarity
        features_flat = features.reshape(-1, self.C)
        self.features_norm = (features_flat - features_flat.mean(axis=1, keepdims=True)) / (np.linalg.norm(features_flat, axis=1, keepdims=True) + 1e-8)

        # JIT compile similarity computation
        @jax.jit
        def compute_similarity(features_norm, target_idx):
            target_feature = features_norm[target_idx]
            similarities = features_norm @ target_feature
            return similarities

        @jax.jit
        def compute_similarity_vec(features_norm, target_vec):
            target_vec = target_vec - target_vec.mean()
            target_vec = target_vec / (jnp.linalg.norm(target_vec) + 1e-8)
            similarities = features_norm @ target_vec
            return similarities

        self.compute_similarity = compute_similarity
        self.compute_similarity_vec = compute_similarity_vec
        self.features_norm_jax = jnp.array(self.features_norm)
        self.features_jax = jnp.array(features.reshape(-1, self.C))

        # Convert original image to float array for blending
        self.image_array = np.array(image).astype(np.float32) / 255.0

        # Current state
        self.current_row = self.H // 2
        self.current_col = self.W // 2
        self.alpha = 0.5
        self.painted_tiles = set()  # set of (row, col) for paint mode

    def get_similarity_map(self, row: int, col: int) -> np.ndarray:
        """Compute similarity map for a given patch."""
        target_idx = row * self.W + col
        similarities = self.compute_similarity(self.features_norm_jax, target_idx)
        return np.array(similarities).reshape(self.H, self.W)

    def get_similarity_map_painted(self) -> np.ndarray:
        """Compute similarity map for averaged painted tiles."""
        indices = [r * self.W + c for r, c in self.painted_tiles]
        avg_feature = self.features_jax[jnp.array(indices)].mean(axis=0)
        similarities = self.compute_similarity_vec(self.features_norm_jax, avg_feature)
        return np.array(similarities).reshape(self.H, self.W)

    def render_frame(self, row: int, col: int) -> tuple[np.ndarray, tuple[float, float, float]]:
        """Render the blended visualization."""
        # Compute similarity
        if self.painted_tiles:
            sim_map = self.get_similarity_map_painted()
        else:
            sim_map = self.get_similarity_map(row, col)

        # Normalize to [0, 1] for colormap
        sim_min, sim_max = sim_map.min(), sim_map.max()
        sim_norm = (sim_map - sim_min) / (sim_max - sim_min + 1e-8)

        # Apply colormap
        sim_rgb = apply_colormap(sim_norm, 'viridis')

        # Resize similarity map to image size
        sim_pil = Image.fromarray(sim_rgb).resize((self.img_w, self.img_h), Image.Resampling.NEAREST)
        sim_array = np.array(sim_pil).astype(np.float32) / 255.0

        # Blend with original
        blended = self.image_array * (1 - self.alpha) + sim_array * self.alpha

        px_per_patch_x = self.img_w / self.W
        px_per_patch_y = self.img_h / self.H

        if self.painted_tiles:
            # Draw red outline on each painted tile
            for pr, pc in self.painted_tiles:
                x0 = int(pc * px_per_patch_x)
                y0 = int(pr * px_per_patch_y)
                x1 = int((pc + 1) * px_per_patch_x)
                y1 = int((pr + 1) * px_per_patch_y)
                # Top and bottom edges
                for px_ in range(max(0, x0), min(self.img_w, x1)):
                    for t in range(2):
                        py_ = y0 + t
                        if 0 <= py_ < self.img_h:
                            blended[py_, px_] = [1.0, 0.0, 0.0]
                        py_ = y1 - 1 - t
                        if 0 <= py_ < self.img_h:
                            blended[py_, px_] = [1.0, 0.0, 0.0]
                # Left and right edges
                for py_ in range(max(0, y0), min(self.img_h, y1)):
                    for t in range(2):
                        px_ = x0 + t
                        if 0 <= px_ < self.img_w:
                            blended[py_, px_] = [1.0, 0.0, 0.0]
                        px_ = x1 - 1 - t
                        if 0 <= px_ < self.img_w:
                            blended[py_, px_] = [1.0, 0.0, 0.0]
        else:
            # Draw crosshair at target location
            cx = int(col * px_per_patch_x + px_per_patch_x / 2)
            cy = int(row * px_per_patch_y + px_per_patch_y / 2)
            cross_size = 10
            cross_thickness = 2

            # Red crosshair
            for dy in range(-cross_size, cross_size + 1):
                for dx in range(-cross_thickness // 2, cross_thickness // 2 + 1):
                    py, px = cy + dy, cx + dx
                    if 0 <= py < self.img_h and 0 <= px < self.img_w:
                        blended[py, px] = [1.0, 0.0, 0.0]
            for dx in range(-cross_size, cross_size + 1):
                for dy in range(-cross_thickness // 2, cross_thickness // 2 + 1):
                    py, px = cy + dy, cx + dx
                    if 0 <= py < self.img_h and 0 <= px < self.img_w:
                        blended[py, px] = [1.0, 0.0, 0.0]

        # Add alpha channel for DearPyGui (RGBA)
        rgba = np.ones((self.img_h, self.img_w, 4), dtype=np.float32)
        rgba[..., :3] = blended

        return rgba.flatten(), (sim_min, sim_max, sim_map.mean())


def main():
    print("Loading model...")
    model = load_dinov3(MODEL_PATH, dtype=jnp.float32)

    # Try loading initial image (optional)
    vis = None
    img_w, img_h = 512, 512  # default placeholder size
    if IMAGE_PATH and os.path.isfile(IMAGE_PATH):
        print("Loading image...")
        image, image_tensor = load_and_preprocess_image(IMAGE_PATH, IMAGE_SIZE, PATCH_SIZE, FEATURE_SCALE)
        print(f"Display size: {image.size}")
        print(f"Feature tensor shape: {image_tensor.shape}")

        print("Extracting features...")
        features = extract_features(model, image_tensor, embed_dim=model.config.hidden_size)
        features_np = np.array(features)
        print(f"Feature grid: {features_np.shape} ({features_np.shape[1]}x{features_np.shape[0]} patches)")
        vis = SimilarityVisualizer(image, features_np)
        img_w, img_h = image.size
    else:
        print("No starting image — paste one with Ctrl+V")

    @dataclass
    class State:
        vis: SimilarityVisualizer | None
        img_w: int
        img_h: int
        tex_id: int = 0
        vp_w: int = 0
        vp_h: int = 0

    CONTROLS_PAD_Y = 100  # vertical space for controls above plot
    CONTROLS_PAD_X = 30   # horizontal padding
    state = State(vis=vis, img_w=img_w, img_h=img_h)

    # Setup DearPyGui
    dpg.create_context()

    # Create texture
    with dpg.texture_registry(tag="tex_registry"):
        if vis:
            initial_frame, stats = vis.render_frame(vis.current_row, vis.current_col)
        else:
            # Grey placeholder
            initial_frame = np.full(img_w * img_h * 4, 0.2, dtype=np.float32)
            initial_frame[3::4] = 1.0  # alpha
        dpg.add_raw_texture(
            width=img_w,
            height=img_h,
            default_value=initial_frame,
            format=dpg.mvFormat_Float_rgba,
            tag=f"main_texture_{state.tex_id}"
        )

    def update_display():
        v = state.vis
        if v is None:
            return
        frame, stats = v.render_frame(v.current_row, v.current_col)
        dpg.set_value(f"main_texture_{state.tex_id}", frame)
        if v.painted_tiles:
            dpg.set_value("stats_text",
                f"Paint: {len(v.painted_tiles)} tiles | "
                f"Similarity: [{stats[0]:.3f}, {stats[1]:.3f}] mean={stats[2]:.3f}")
        else:
            dpg.set_value("stats_text",
                f"Patch: ({v.current_row}, {v.current_col}) | "
                f"Similarity: [{stats[0]:.3f}, {stats[1]:.3f}] mean={stats[2]:.3f}")

    def resize_plot_to_fit():
        """Resize the plot to fit the current viewport, preserving aspect ratio."""
        vp_w = dpg.get_viewport_client_width()
        vp_h = dpg.get_viewport_client_height()
        if vp_w == state.vp_w and vp_h == state.vp_h:
            return
        state.vp_w = vp_w
        state.vp_h = vp_h

        iw, ih = state.img_w, state.img_h
        if iw == 0 or ih == 0:
            return

        avail_w = max(100, vp_w - CONTROLS_PAD_X)
        avail_h = max(100, vp_h - CONTROLS_PAD_Y)
        aspect = iw / ih

        if avail_w / avail_h > aspect:
            plot_h = avail_h
            plot_w = int(plot_h * aspect)
        else:
            plot_w = avail_w
            plot_h = int(plot_w / aspect)

        dpg.configure_item("main_plot", width=plot_w, height=plot_h)

    def handle_mouse_input():
        """Update if left mouse button is down and mouse is over the plot.

        Ctrl+drag: paint tiles (features averaged for similarity target).
        Drag without Ctrl: clear paint buffer, use single tile as target.
        """
        if state.vis is None or not dpg.is_mouse_button_down(0):
            return

        mouse_pos = dpg.get_plot_mouse_pos()
        x, y = mouse_pos
        v = state.vis
        iw, ih = state.img_w, state.img_h

        # Check if mouse is within plot bounds
        if x < 0 or x >= iw or y < 0 or y >= ih:
            return

        # Convert to patch coordinates using feature grid dimensions
        col = int(x / iw * v.W)
        row = int((ih - y) / ih * v.H)

        # Clamp to valid range
        row = max(0, min(v.H - 1, row))
        col = max(0, min(v.W - 1, col))

        ctrl_held = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)

        if ctrl_held:
            # Paint mode: add tile to paint buffer
            tile = (row, col)
            if tile not in v.painted_tiles:
                v.painted_tiles.add(tile)
                update_display()
        else:
            # Normal mode: clear paint buffer, set single target
            if v.painted_tiles:
                v.painted_tiles.clear()
            if row != v.current_row or col != v.current_col:
                v.current_row = row
                v.current_col = col
            update_display()

    def on_alpha_change(sender, app_data):
        if state.vis is not None:
            state.vis.alpha = app_data
            update_display()

    def handle_paste():
        dpg.set_value("stats_text", "Reading clipboard...")
        dpg.render_dearpygui_frame()

        clip_img = get_clipboard_image()
        if clip_img is None:
            dpg.set_value("stats_text", "No image found in clipboard")
            return

        dpg.set_value("stats_text", f"Processing pasted image ({clip_img.size[0]}x{clip_img.size[1]})...")
        dpg.render_dearpygui_frame()

        img, tensor = preprocess_image(clip_img, IMAGE_SIZE, PATCH_SIZE, FEATURE_SCALE)
        feats = extract_features(model, tensor, embed_dim=model.config.hidden_size)
        feats_np = np.array(feats)

        new_vis = SimilarityVisualizer(img, feats_np)
        if state.vis is not None:
            new_vis.alpha = state.vis.alpha  # preserve alpha setting
        new_w, new_h = img.size
        state.vis = new_vis
        state.img_w = new_w
        state.img_h = new_h

        # Recreate texture with new dimensions (DPG doesn't release aliases on delete)
        old_tag = f"main_texture_{state.tex_id}"
        state.tex_id += 1
        new_tag = f"main_texture_{state.tex_id}"

        dpg.delete_item("image_series")
        dpg.delete_item(old_tag)

        initial_frame, _ = new_vis.render_frame(new_vis.current_row, new_vis.current_col)
        dpg.add_raw_texture(
            width=new_w,
            height=new_h,
            default_value=initial_frame,
            format=dpg.mvFormat_Float_rgba,
            tag=new_tag,
            parent="tex_registry",
        )
        dpg.add_image_series(
            new_tag,
            bounds_min=[0, 0],
            bounds_max=[new_w, new_h],
            tag="image_series",
            parent="y_axis",
        )

        # Update plot axis limits for new image coordinate space
        dpg.set_axis_limits("x_axis", 0, new_w)
        dpg.set_axis_limits("y_axis", new_h, 0)

        # Force resize recalculation on next frame
        state.vp_w = 0
        state.vp_h = 0

        update_display()
        print(f"Pasted image: {clip_img.size} -> display {new_w}x{new_h}, features {feats_np.shape}")

    # Create window
    with dpg.window(label="DINOv3 Feature Similarity", tag="main_window"):
        dpg.add_text("Click to explore | Ctrl+Click to paint tiles | Ctrl+V to paste", color=(200, 200, 200))
        dpg.add_text("", tag="stats_text")

        dpg.add_slider_float(
            label="Overlay Alpha",
            default_value=0.5,
            min_value=0.0,
            max_value=1.0,
            callback=on_alpha_change
        )

        dpg.add_separator()

        # Use a plot to display the image (allows mouse position tracking)
        with dpg.plot(
            label="",
            width=-1,
            height=-1,
            no_title=True,
            no_menus=True,
            no_box_select=True,
            equal_aspects=True,
            tag="main_plot"
        ):
            dpg.add_plot_axis(dpg.mvXAxis, no_tick_labels=True, no_tick_marks=True, tag="x_axis")
            dpg.set_axis_limits("x_axis", 0, img_w)

            with dpg.plot_axis(dpg.mvYAxis, no_tick_labels=True, no_tick_marks=True, tag="y_axis"):
                dpg.set_axis_limits("y_axis", img_h, 0)  # Flip Y axis
                dpg.add_image_series(
                    f"main_texture_{state.tex_id}",
                    bounds_min=[0, 0],
                    bounds_max=[img_w, img_h],
                    tag="image_series"
                )

    # Initial stats update
    if vis:
        update_display()
    else:
        dpg.set_value("stats_text", "Paste an image with Ctrl+V")

    init_vp_w = min(img_w + CONTROLS_PAD_X, 1600)
    init_vp_h = min(img_h + CONTROLS_PAD_Y, 1000)
    dpg.create_viewport(title="DINOv3 Feature Similarity Visualizer", width=init_vp_w, height=init_vp_h)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)

    # Manual render loop to check mouse state and keyboard each frame
    v_was_down = False
    while dpg.is_dearpygui_running():
        resize_plot_to_fit()
        handle_mouse_input()

        # Detect Ctrl+V keypress (edge-triggered)
        v_is_down = dpg.is_key_down(dpg.mvKey_V)
        if v_is_down and not v_was_down:
            if dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl):
                handle_paste()
        v_was_down = v_is_down

        dpg.render_dearpygui_frame()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
