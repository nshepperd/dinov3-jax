# %%
import ipywidgets as widgets
import jax
import jax.numpy as jnp

# Visualization imports
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import interactive
from PIL import Image

# JAX DINOv3 imports
from dinov3_jax import dinov3_vitl16, dinov3_vits16

# %%
# Configuration
MODEL_NAME = "vit_large"  # Can be "vit_small", "vit_base", or "vit_large"
PATCH_SIZE = 16
IMAGE_SIZE = 768

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Path to your image
image_uri = "/home/em/Dev/neural/minihf/yonaka/data/2024-11-01_15.10.05.png"

# Path to pretrained weights (you'll need to set this)
WEIGHTS_PATH = "/data/models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
# WEIGHTS_PATH = None  # Set this to your weights path


# %%
def load_image_from_path(path: str) -> Image.Image:
    """Load image from file path."""
    with open(path, "rb") as f:
        return Image.open(f).convert("RGB")


def resize_image(
    image: Image.Image, image_size: int = IMAGE_SIZE, patch_size: int = PATCH_SIZE
):
    """Resize image to dimensions divisible by patch size."""
    w, h = image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    new_h = h_patches * patch_size
    new_w = w_patches * patch_size
    return image.resize((new_w, new_h), Image.BILINEAR)


def preprocess_image(image: Image.Image):
    """Preprocess image for DINOv3."""
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(image).astype(np.float32) / 255.0

    # Convert to CHW format
    img_array = img_array.transpose(2, 0, 1)

    # Normalize with ImageNet stats
    for c in range(3):
        img_array[c] = (img_array[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

    # Add batch dimension
    img_array = img_array[np.newaxis, ...]

    return jnp.array(img_array)


# %%
# Load and preprocess image
image = load_image_from_path(image_uri)
image_resized = resize_image(image)
image_tensor = preprocess_image(image_resized)

print(f"Original image size: {image.size}")
print(f"Resized image size: {image_resized.size}")
print(f"Tensor shape: {image_tensor.shape}")

# %%
# Initialize model
if MODEL_NAME == "vit_small":
    model = dinov3_vits16()
    embed_dim = 384
    num_heads = 6
    n_layers = 12
elif MODEL_NAME == "vit_base":
    model = dinov3_vits16()
    embed_dim = 768
    num_heads = 12
    n_layers = 12
elif MODEL_NAME == "vit_large":
    model = dinov3_vitl16()
    embed_dim = 1024
    num_heads = 16
    n_layers = 24
else:
    raise ValueError(f"Unknown model: {MODEL_NAME}")

# Initialize parameters
key = jax.random.PRNGKey(42)

# Load pretrained weights if available
if WEIGHTS_PATH:
    import torch

    print(f"Loading weights from {WEIGHTS_PATH}")
    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]

    from dinov3_jax.eepynox.nn.param import load_state_dict
    from dinov3_jax.utils import convert_pytorch_to_jax

    jax_params = convert_pytorch_to_jax(state_dict)

    model = load_state_dict(model, jax_params)
    # Update params with loaded weights
    # for param_name, param in model.gen_named_parameters():
    #     if param_name in jax_params:
    #         params[param.name] = jax_params[param_name]
    #     else:
    #         print(f"Warning: {param_name} not found in loaded weights")
else:
    print("Using random initialization (set WEIGHTS_PATH to use pretrained weights)")

# Create context
# cx = Context(params, key, mode='eval')

# %%
# Extract features from intermediate layers
print(f"Extracting features from {n_layers} layers...")

# Get intermediate features
# We'll extract the last layer features for visualization
features = model.get_intermediate_layers(
    image_tensor,
    n=n_layers,  # Get last layer
    reshape=True,  # Reshape to spatial format
    norm=True,  # Apply normalization
)

# Get the last layer features
if isinstance(features, tuple):
    x = features[-1]  # Get last layer
else:
    x = features

# Remove batch dimension
x = x.squeeze(0)  # Remove batch dimension

# Rearrange to HWC format
if len(x.shape) == 3 and x.shape[0] == embed_dim:
    # Current shape is (C, H, W), convert to (H, W, C)
    x = x.transpose(1, 2, 0)

print(f"Feature shape: {x.shape}")

# %%
# Convert to numpy for visualization
x_np = np.array(x)

# Optional: PCA visualization (commented out for now)
# from sklearn.decomposition import PCA
# pca = PCA(n_components=3, whiten=True)
# pca.fit(x_np.reshape(-1, x_np.shape[-1]))
# projected = pca.transform(x_np.reshape(-1, x_np.shape[-1])).reshape(x_np.shape[0], x_np.shape[1], 3)
# projected = (projected - projected.min()) / (projected.max() - projected.min())

# %%
# Widget-based Interface for Feature Similarity Visualization


def create_widget_interface(features, title="DINOv3 Feature Similarity Map (JAX)"):
    """
    Create a widget-based interface for exploring similarity maps.

    Args:
        features: numpy array of shape (H, W, C)
        title: str - title for the plot
    """
    H, W, C = features.shape

    # Normalize features for cosine similarity
    features_flat = features.reshape(-1, C)
    # L2 normalize
    features_norm = features_flat / (
        np.linalg.norm(features_flat, axis=1, keepdims=True) + 1e-8
    )

    def plot_similarity(
        target_row, target_col, colormap="viridis", vmin=None, vmax=None
    ):
        # Get target feature
        target_idx = target_row * W + target_col
        target_feature = features_norm[target_idx : target_idx + 1]

        # Compute cosine similarity
        similarities = np.dot(features_norm, target_feature.T).squeeze()
        similarity_map = similarities.reshape(H, W)

        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Left: Original image (if available)
        ax1 = axes[0]
        if image_resized is not None:
            ax1.imshow(image_resized)
            ax1.plot(
                target_col * PATCH_SIZE + PATCH_SIZE // 2,
                target_row * PATCH_SIZE + PATCH_SIZE // 2,
                "r+",
                markersize=20,
                markeredgewidth=3,
            )
            ax1.set_title("Original Image with Target")
        else:
            # Show feature norm magnitude as proxy
            feature_norms = np.linalg.norm(features, axis=2)
            ax1.imshow(feature_norms, cmap="gray")
            ax1.plot(target_col, target_row, "r+", markersize=15, markeredgewidth=3)
            ax1.set_title("Feature Magnitude")
        ax1.set_xlabel("Width")
        ax1.set_ylabel("Height")
        ax1.grid(True, alpha=0.3)

        # Right: Similarity map
        ax2 = axes[1]
        im = ax2.imshow(
            similarity_map, cmap=colormap, interpolation="nearest", vmin=vmin, vmax=vmax
        )

        # Add red cross for target
        ax2.plot(target_col, target_row, "r+", markersize=15, markeredgewidth=3)

        # Styling
        ax2.set_title(f"{title}\nTarget: ({target_row}, {target_col})")
        ax2.set_xlabel("Width (patches)")
        ax2.set_ylabel("Height (patches)")
        ax2.grid(True, alpha=0.3)

        # Colorbar
        fig.colorbar(im, ax=ax2, label="Cosine Similarity")

        # Add similarity statistics
        fig.suptitle(
            f"Similarity Stats - Min: {similarity_map.min():.3f}, "
            f"Max: {similarity_map.max():.3f}, "
            f"Mean: {similarity_map.mean():.3f}",
            fontsize=10,
        )

        plt.tight_layout()
        plt.show()

    # Create sliders
    row_slider = widgets.IntSlider(
        value=H // 2,
        min=0,
        max=H - 1,
        step=1,
        description="Row:",
        style={"description_width": "initial"},
        continuous_update=False,  # Only update on release for performance
    )
    col_slider = widgets.IntSlider(
        value=W // 2,
        min=0,
        max=W - 1,
        step=1,
        description="Col:",
        style={"description_width": "initial"},
        continuous_update=False,
    )

    # Colormap dropdown
    colormap_dropdown = widgets.Dropdown(
        options=[
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "coolwarm",
            "RdBu_r",
            "seismic",
            "jet",
            "turbo",
        ],
        value="viridis",
        description="Colormap:",
        style={"description_width": "initial"},
    )

    # Value range sliders for better visualization
    vmin_slider = widgets.FloatSlider(
        value=-1.0,
        min=-1.0,
        max=1.0,
        step=0.01,
        description="Min value:",
        style={"description_width": "initial"},
        continuous_update=False,
    )
    vmax_slider = widgets.FloatSlider(
        value=1.0,
        min=-1.0,
        max=1.0,
        step=0.01,
        description="Max value:",
        style={"description_width": "initial"},
        continuous_update=False,
    )

    # Checkbox for auto range
    auto_range = widgets.Checkbox(
        value=True,
        description="Auto color range",
        style={"description_width": "initial"},
    )

    def update_plot(target_row, target_col, colormap, vmin, vmax, auto):
        if auto:
            plot_similarity(target_row, target_col, colormap, None, None)
        else:
            plot_similarity(target_row, target_col, colormap, vmin, vmax)

    # Interactive widget
    interactive_plot = interactive(
        update_plot,
        target_row=row_slider,
        target_col=col_slider,
        colormap=colormap_dropdown,
        vmin=vmin_slider,
        vmax=vmax_slider,
        auto=auto_range,
    )

    # Add some instructions
    print("=" * 60)
    print("DINOv3 Feature Similarity Visualizer (JAX Implementation)")
    print("=" * 60)
    print("Instructions:")
    print("- Use Row/Col sliders to select the target patch")
    print("- Red cross marks the selected patch")
    print("- Brighter colors indicate higher similarity")
    print("- Toggle 'Auto color range' to manually adjust visualization")
    print(f"\nFeature map shape: {features.shape}")
    print(f"Each patch corresponds to {PATCH_SIZE}x{PATCH_SIZE} pixels")
    print("=" * 60)

    return interactive_plot


# %%
# Create and display the interactive widget
widget_interface = create_widget_interface(x_np)
display(widget_interface)


# %%
# Optional: Quick similarity plot function for non-interactive use
def quick_similarity_plot(features, target_row=None, target_col=None, figsize=(12, 5)):
    """
    Create a single similarity plot for quick analysis.

    Args:
        features: numpy array of shape (H, W, C)
        target_row, target_col: int - target pixel coordinates (defaults to center)
        figsize: tuple - figure size
    """
    H, W, C = features.shape

    if target_row is None:
        target_row = H // 2
    if target_col is None:
        target_col = W // 2

    # Normalize and compute similarity
    features_flat = features.reshape(-1, C)
    features_norm = features_flat / (
        np.linalg.norm(features_flat, axis=1, keepdims=True) + 1e-8
    )

    target_idx = target_row * W + target_col
    target_feature = features_norm[target_idx : target_idx + 1]
    similarities = np.dot(features_norm, target_feature.T).squeeze()
    similarity_map = similarities.reshape(H, W)

    # Plot
    _fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Feature norms
    ax1 = axes[0]
    feature_norms = np.linalg.norm(features, axis=2)
    im1 = ax1.imshow(feature_norms, cmap="gray")
    ax1.plot(target_col, target_row, "r+", markersize=15, markeredgewidth=3)
    ax1.set_title(f"Feature Norms\nTarget: ({target_row}, {target_col})")
    plt.colorbar(im1, ax=ax1)

    # Right: Similarity map
    ax2 = axes[1]
    im2 = ax2.imshow(similarity_map, cmap="viridis", interpolation="nearest")
    ax2.plot(target_col, target_row, "r+", markersize=15, markeredgewidth=3)
    ax2.set_title("Cosine Similarity Map")
    plt.colorbar(im2, ax=ax2, label="Similarity")

    for ax in axes:
        ax.set_xlabel("Width (patches)")
        ax.set_ylabel("Height (patches)")

    plt.tight_layout()
    plt.show()

    return similarity_map


# Example: Create a quick plot
# quick_similarity_plot(x_np)

print(
    "\nWidget interface created! Use the sliders above to explore feature similarities."
)
print("You can also use quick_similarity_plot(x_np) for a static visualization.")
