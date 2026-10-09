# %%
import torch

REPO_DIR = '../dinov3'

# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
# image = load_image(url)

model = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights='/mnt/netdata/models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth').cuda()
model.requires_grad_(False)
# model = torch.hub.load(REPO_DIR, 'dinov3_convnext_base', source='local', weights='/mnt/netdata/models/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth').cuda()
# model.requires_grad_(False)

# %%
import einops
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.decomposition import PCA

# examples of available DINOv3 models:
MODEL_DINOV3_VITS = "dinov3_vits16"
MODEL_DINOV3_VITSP = "dinov3_vits16plus"
MODEL_DINOV3_VITB = "dinov3_vitb16"
MODEL_DINOV3_VITL = "dinov3_vitl16"
MODEL_DINOV3_VITHP = "dinov3_vith16plus"
MODEL_DINOV3_VIT7B = "dinov3_vit7b16"

MODEL_NAME = MODEL_DINOV3_VITL

PATCH_SIZE = 16
IMAGE_SIZE = 768

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

image_uri = "/home/em/Dev/neural/minihf/yonaka/data/2024-11-01_15.10.05.png"

def load_image_from_url(url: str) -> Image:
    # with urllib.request.urlopen(url) as f:
    with open(url, 'rb') as f:
        return Image.open(f).convert("RGB")
        
# image resize transform to dimensions divisible by patch size
def resize_transform(
    mask_image: Image,
    image_size: int = IMAGE_SIZE,
    patch_size: int = PATCH_SIZE,
) -> torch.Tensor:
    w, h = mask_image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size)))


image = load_image_from_url(image_uri)
image_resized = resize_transform(image)
image_resized_norm = TF.normalize(image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)

MODEL_TO_NUM_LAYERS = {
    MODEL_DINOV3_VITS: 12,
    MODEL_DINOV3_VITSP: 12,
    MODEL_DINOV3_VITB: 12,
    MODEL_DINOV3_VITL: 24,
    MODEL_DINOV3_VITHP: 32,
    MODEL_DINOV3_VIT7B: 40,
}

n_layers = MODEL_TO_NUM_LAYERS[MODEL_NAME]
# n_layers = 4


with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float32):
    feats = model.get_intermediate_layers(image_resized_norm.unsqueeze(0).cuda(), n=range(n_layers), reshape=True, norm=True)
    x = feats[-1].squeeze().detach().cpu()
    # x = x.movedim(0,-1) # for convnext
    dim = x.shape[0]
    x = x.view(dim, -1).permute(1, 0)

x = einops.rearrange(x, '(h w) d -> h w d', h=image_resized_norm.shape[1] // PATCH_SIZE, w=image_resized_norm.shape[2] // PATCH_SIZE)
x.shape  # noqa: B018


# %%
x.shape  # noqa: B018
pca = PCA(n_components=3, whiten=True)
pca.fit(x.reshape(-1, x.shape[-1]))

# %%
def pca_project_2d(im):
    [h, w, c] = im.shape
    return torch.from_numpy(pca.transform(im.numpy().reshape(h*w,c))).reshape(h, w, 3)

# %%
# apply the PCA, and then reshape
# h_patches, w_patches = [int(d / PATCH_SIZE) for d in image_resized.shape[1:]]
# x.view(h_patches, w_patches, -1).shape
# projected_image = torch.from_numpy(pca.transform(x.numpy())).view(h_patches, w_patches, 3)
projected_image = pca_project_2d(x)

# multiply by 2.0 and pass through a sigmoid to get vibrant colors 
projected_image = torch.nn.functional.sigmoid(projected_image.mul(2.0)).permute(2, 0, 1)

# enjoy
from matplotlib import pyplot as plt

plt.figure(dpi=300)
plt.imshow(projected_image.permute(1, 2, 0))
plt.axis('off')
plt.show()

# %%
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from IPython.display import display
from ipywidgets import interactive


class InteractiveSimilarityMap:
    def __init__(self, features, title="DINOv3 Feature Similarity Map"):
        """
        Initialize the interactive similarity map.
        
        Args:
            features: torch.Tensor of shape (H, W, C) - your feature map
            title: str - title for the plot
        """
        self.features = features  # (H, W, C)
        self.title = title
        self.H, self.W, self.C = features.shape
        self.target_pixel = (self.H // 2, self.W // 2)  # Start in center
        
        # Normalize features for cosine similarity
        self.features_flat = features.view(-1, self.C)  # (H*W, C)
        self.features_norm = F.normalize(self.features_flat, p=2, dim=1)
        
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.suptitle(self.title, fontsize=14)
        
        # Initial plot
        self.update_similarity_map()
        
        # Connect click event
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
    def compute_similarity_map(self, target_row, target_col):
        """Compute cosine similarity map for target pixel."""
        target_idx = target_row * self.W + target_col
        target_feature = self.features_norm[target_idx:target_idx+1]  # (1, C)
        
        # Compute cosine similarity with all pixels
        similarities = torch.mm(self.features_norm, target_feature.T).squeeze()  # (H*W,)
        similarity_map = similarities.view(self.H, self.W)  # (H, W)
        
        return similarity_map.numpy()
    
    def update_similarity_map(self):
        """Update the visualization."""
        self.ax.clear()
        
        # Compute similarity map
        sim_map = self.compute_similarity_map(*self.target_pixel)
        
        # Create the heatmap
        im = self.ax.imshow(sim_map, cmap='viridis', interpolation='nearest')
        
        # Add colorbar
        if not hasattr(self, 'cbar'):
            self.cbar = self.fig.colorbar(im, ax=self.ax, label='Cosine Similarity')
        else:
            self.cbar.update_normal(im)
        
        # Mark target pixel with red cross
        self.ax.plot(self.target_pixel[1], self.target_pixel[0], 'r+', 
                    markersize=15, markeredgewidth=3, label='Target Pixel')
        
        # Styling
        self.ax.set_title(f'Target: ({self.target_pixel[0]}, {self.target_pixel[1]})')
        self.ax.set_xlabel('Width')
        self.ax.set_ylabel('Height')
        self.ax.legend()
        
        # Update display
        self.fig.canvas.draw()
    
    def on_click(self, event):
        """Handle mouse click events."""
        if event.inaxes != self.ax:
            return
        
        # Get clicked coordinates
        col = round(event.xdata)
        row = round(event.ydata)
        
        # Bounds checking
        if 0 <= row < self.H and 0 <= col < self.W:
            self.target_pixel = (row, col)
            self.update_similarity_map()
            print(f"Target pixel updated to: ({row}, {col})")

def create_widget_interface(features, title="DINOv3 Feature Similarity Map"):
    """
    Create a widget-based interface for exploring similarity maps.
    
    Args:
        features: torch.Tensor of shape (H, W, C)
        title: str - title for the plot
    """
    H, W, C = features.shape
    
    # Normalize features
    features_flat = features.view(-1, C)
    features_norm = F.normalize(features_flat, p=2, dim=1)
    
    def plot_similarity(target_row, target_col):
        # Compute similarity map
        target_idx = target_row * W + target_col
        target_feature = features_norm[target_idx:target_idx+1]
        similarities = torch.mm(features_norm, target_feature.T).squeeze()
        similarity_map = similarities.view(H, W).numpy()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(similarity_map, cmap='viridis', interpolation='nearest')
        
        # Add red cross for target
        ax.plot(target_col, target_row, 'r+', markersize=15, markeredgewidth=3)
        
        # Styling
        ax.set_title(f'{title}\nTarget: ({target_row}, {target_col})')
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        
        # Colorbar
        fig.colorbar(im, ax=ax, label='Cosine Similarity')
        
        plt.tight_layout()
        plt.show()
    
    # Create sliders
    row_slider = widgets.IntSlider(
        value=H//2, min=0, max=H-1, step=1,
        description='Row:', style={'description_width': 'initial'}
    )
    col_slider = widgets.IntSlider(
        value=W//2, min=0, max=W-1, step=1,
        description='Col:', style={'description_width': 'initial'}
    )
    
    # Interactive widget
    interactive_plot = interactive(plot_similarity, 
                                 target_row=row_slider, 
                                 target_col=col_slider)
    return interactive_plot

# Example usage functions
def demo_with_random_features():
    """Demo with random features for testing."""
    # Create random features for demo
    features = torch.randn(144, 264, 1024)
    
    print("Method 1: Click-based interaction")
    print("Click anywhere on the heatmap to set new target pixel")
    # Keep a reference so the click callbacks aren't garbage collected.
    _sim_map = InteractiveSimilarityMap(features)
    plt.show()
    
    print("\nMethod 2: Widget-based interaction")
    print("Use sliders to select target pixel")
    widget_interface = create_widget_interface(features)
    display(widget_interface)

def analyze_dinov3_features(features):
    """
    Analyze your actual DINOv3 features.
    
    Args:
        features: torch.Tensor of shape (H, W, C) - your DINOv3 features
    """
    print(f"Feature map shape: {features.shape}")
    print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")
    
    # Method 1: Click-based
    print("\n=== Click-based Interface ===")
    print("Click on the heatmap to explore different target pixels")
    sim_map = InteractiveSimilarityMap(features, "DINOv3 Feature Similarity")
    plt.show()
    
    # Method 2: Widget-based  
    print("\n=== Widget-based Interface ===")
    print("Use the sliders below to select target coordinates")
    widget_interface = create_widget_interface(features, "DINOv3 Feature Similarity")
    display(widget_interface)
    
    return sim_map, widget_interface

# Quick analysis function
def quick_similarity_plot(features, target_row=None, target_col=None, figsize=(10, 8)):
    """
    Create a single similarity plot for quick analysis.
    
    Args:
        features: torch.Tensor of shape (H, W, C)
        target_row, target_col: int - target pixel coordinates (defaults to center)
        figsize: tuple - figure size
    """
    H, W, C = features.shape
    
    if target_row is None:
        target_row = H // 2
    if target_col is None:
        target_col = W // 2
    
    # Normalize and compute similarity
    features_flat = features.view(-1, C)
    features_norm = F.normalize(features_flat, p=2, dim=1)
    
    target_idx = target_row * W + target_col
    target_feature = features_norm[target_idx:target_idx+1]
    similarities = torch.mm(features_norm, target_feature.T).squeeze()
    similarity_map = similarities.view(H, W).numpy()
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(similarity_map, cmap='viridis', interpolation='nearest')
    ax.plot(target_col, target_row, 'r+', markersize=15, markeredgewidth=3)
    
    ax.set_title(f'DINOv3 Feature Similarity\nTarget: ({target_row}, {target_col})')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    
    fig.colorbar(im, ax=ax, label='Cosine Similarity')
    plt.tight_layout()
    plt.show()
    
    return fig, ax, similarity_map

# Usage instructions
print("=== DINOv3 Feature Similarity Visualizer ===")
print("\nTo use with your features:")
print("1. analyze_dinov3_features(your_features_tensor)")
print("2. Or for a quick plot: quick_similarity_plot(your_features_tensor)")
print("\nFor demo with random data:")
print("3. demo_with_random_features()")
# analyze_dinov3_features(x.view(h_patches, w_patches, -1))
analyze_dinov3_features(x)

# %%
from PIL import Image

mask = Image.open('/home/em/mask.png')
mask = mask.resize((x.shape[1], x.shape[0]))
mask = TF.to_tensor(mask)
# (mask[0]==1.0)
# image_resized_norm.shape
mask = (mask.mean(0)>0.0)

def plot_mask_similarity(features, mask, title="Mask-based Similarity", figsize=(10, 8)):
    """
    Create similarity plot using average of features selected by boolean mask.
    
    Args:
        features: torch.Tensor of shape (H, W, C) - your feature map
        mask: torch.Tensor or numpy array of shape (H, W) - boolean mask
        title: str - plot title
        figsize: tuple - figure size
    
    Returns:
        fig, ax, similarity_map, target_feature
    """
    H, W, C = features.shape
    
    # Convert mask to torch tensor if needed
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    
    # Ensure mask is boolean
    mask = mask.bool()
    
    # Check mask shape
    if mask.shape != (H, W):
        raise ValueError(f"Mask shape {mask.shape} doesn't match feature map shape {(H, W)}")
    
    # Get features for masked pixels
    masked_features = features[mask]  # (N_masked, C) where N_masked is number of True pixels
    
    if masked_features.shape[0] == 0:
        raise ValueError("Mask contains no True values!")
    
    # Compute average feature vector from masked region
    target_feature = masked_features.mean(dim=0, keepdim=True) - features.mean(dim=(0,1))  # (1, C)
    # target_feature = target_feature / features.std(dim=(0,1))
    
    # Normalize features for cosine similarity
    features_flat = features.view(-1, C)  # (H*W, C)
    features_norm = F.normalize(features_flat, p=2, dim=1)
    target_feature_norm = F.normalize(target_feature, p=2, dim=1)
    
    # Compute cosine similarity with all pixels
    similarities = torch.mm(features_norm, target_feature_norm.T).squeeze()  # (H*W,)
    similarities = torch.sigmoid((similarities-0.3)*10)
    similarity_map = similarities.view(H, W).numpy()  # (H, W)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(similarity_map, cmap='viridis', interpolation='nearest')
    
    # Overlay mask region (show which pixels were averaged)
    mask_np = mask.numpy()
    masked_overlay = np.ma.masked_where(~mask_np, np.ones_like(mask_np))
    ax.contour(masked_overlay, levels=[0.5], colors='red', linewidths=2, alpha=0.8)
    
    # Mark center of masked region with red cross
    # if mask_np.any():
    #     mask_coords = np.where(mask_np)
    #     center_row = int(np.mean(mask_coords[0]))
    #     center_col = int(np.mean(mask_coords[1]))
    #     ax.plot(center_col, center_row, 'r+', markersize=4, markeredgewidth=3)
    
    # Styling
    num_pixels = mask.sum().item()
    ax.set_title(f'{title}\nAveraged over {num_pixels} pixels')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    
    # Colorbar
    fig.colorbar(im, ax=ax, label='Cosine Similarity')
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax, similarity_map, target_feature.squeeze()
plot_mask_similarity(x, mask)
