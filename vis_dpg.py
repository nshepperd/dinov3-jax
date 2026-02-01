"""
Interactive DINOv3 Feature Similarity Visualizer using DearPyGui

Click anywhere on the image to see which patches have similar features.
"""

import os
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import dearpygui.dearpygui as dpg

# JAX DINOv3 imports
from dinov3_jax import dinov3_vitl16

# Configuration
PATCH_SIZE = 16
IMAGE_SIZE = 768*3  # Smaller for interactive use

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Paths - adjust these
IMAGE_PATH = "/home/em/Dev/neural/minihf/yonaka/data/2024-11-01_15.10.05.png"
WEIGHTS_PATH = "/data/models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"


def load_and_preprocess_image(path: str, image_size: int, patch_size: int):
    """Load and preprocess image for DINOv3."""
    image = Image.open(path).convert("RGB")

    # Resize to dimensions divisible by patch size
    w, h = image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    new_h = h_patches * patch_size
    new_w = w_patches * patch_size
    image_resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Convert to tensor
    img_array = np.array(image_resized).astype(np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)  # CHW
    for c in range(3):
        img_array[c] = (img_array[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    img_tensor = jnp.array(img_array[np.newaxis, ...])

    return image_resized, img_tensor


def extract_features(model, image_tensor, embed_dim):
    """Extract features from the model."""
    @jax.jit
    def fwd(model, image_tensor):
        features = model.get_intermediate_layers(
            image_tensor.astype(jnp.float32),
            n=24,  # All layers for ViT-L
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
        self.features_norm = features_flat / (np.linalg.norm(features_flat, axis=1, keepdims=True) + 1e-8)

        # JIT compile similarity computation
        @jax.jit
        def compute_similarity(features_norm, target_idx):
            target_feature = features_norm[target_idx]
            similarities = features_norm @ target_feature
            return similarities

        self.compute_similarity = compute_similarity
        self.features_norm_jax = jnp.array(self.features_norm)

        # Convert original image to float array for blending
        self.image_array = np.array(image).astype(np.float32) / 255.0

        # Current state
        self.current_row = self.H // 2
        self.current_col = self.W // 2
        self.alpha = 0.5

    def get_similarity_map(self, row: int, col: int) -> np.ndarray:
        """Compute similarity map for a given patch."""
        target_idx = row * self.W + col
        similarities = self.compute_similarity(self.features_norm_jax, target_idx)
        return np.array(similarities).reshape(self.H, self.W)

    def render_frame(self, row: int, col: int) -> tuple[np.ndarray, tuple[float, float, float]]:
        """Render the blended visualization."""
        # Compute similarity
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

        # Draw crosshair at target location
        cx = col * PATCH_SIZE + PATCH_SIZE // 2
        cy = row * PATCH_SIZE + PATCH_SIZE // 2
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
    print("Loading image...")
    image, image_tensor = load_and_preprocess_image(IMAGE_PATH, IMAGE_SIZE, PATCH_SIZE)
    print(f"Image size: {image.size}")
    print(f"Tensor shape: {image_tensor.shape}")

    print("Loading model...")
    model = dinov3_vitl16()

    if WEIGHTS_PATH:
        import torch
        print(f"Loading weights from {WEIGHTS_PATH}")
        state_dict = torch.load(WEIGHTS_PATH, map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']

        from dinov3_jax.utils import convert_pytorch_to_jax
        jax_params = convert_pytorch_to_jax(state_dict)
        model = model.load_state_dict(jax_params)
        del jax_params, state_dict

    print("Extracting features...")
    features = extract_features(model, image_tensor, embed_dim=1024)
    features_np = np.array(features)
    print(f"Feature shape: {features_np.shape}")

    # Create visualizer
    vis = SimilarityVisualizer(image, features_np)

    # Setup DearPyGui
    dpg.create_context()

    img_w, img_h = image.size

    # Create texture
    with dpg.texture_registry():
        # Initial render
        initial_frame, stats = vis.render_frame(vis.current_row, vis.current_col)
        dpg.add_raw_texture(
            width=img_w,
            height=img_h,
            default_value=initial_frame,
            format=dpg.mvFormat_Float_rgba,
            tag="main_texture"
        )

    def update_display():
        frame, stats = vis.render_frame(vis.current_row, vis.current_col)
        dpg.set_value("main_texture", frame)
        dpg.set_value("stats_text",
            f"Patch: ({vis.current_row}, {vis.current_col}) | "
            f"Similarity: [{stats[0]:.3f}, {stats[1]:.3f}] mean={stats[2]:.3f}")

    def handle_mouse_input():
        """Update if left mouse button is down and mouse is over the plot."""
        if not dpg.is_mouse_button_down(0):  # 0 = left button
            return

        mouse_pos = dpg.get_plot_mouse_pos()
        x, y = mouse_pos

        # Check if mouse is within plot bounds
        if x < 0 or x >= img_w or y < 0 or y >= img_h:
            return

        # Convert to patch coordinates (flip y since plot axis is inverted)
        col = int(x / PATCH_SIZE)
        row = int((img_h - y) / PATCH_SIZE)

        # Clamp to valid range
        row = max(0, min(vis.H - 1, row))
        col = max(0, min(vis.W - 1, col))

        # Only update if position changed
        if row != vis.current_row or col != vis.current_col:
            vis.current_row = row
            vis.current_col = col
            update_display()

    def on_alpha_change(sender, app_data):
        vis.alpha = app_data
        update_display()

    # Create window
    with dpg.window(label="DINOv3 Feature Similarity", tag="main_window"):
        dpg.add_text("Click on the image to explore feature similarities", color=(200, 200, 200))
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
            width=img_w,
            height=img_h,
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
                    "main_texture",
                    bounds_min=[0, 0],
                    bounds_max=[img_w, img_h],
                    tag="image_series"
                )

    # Initial stats update
    update_display()

    dpg.create_viewport(title="DINOv3 Feature Similarity Visualizer", width=img_w + 50, height=img_h + 150)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)

    # Manual render loop to check mouse state each frame
    while dpg.is_dearpygui_running():
        handle_mouse_input()
        dpg.render_dearpygui_frame()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
