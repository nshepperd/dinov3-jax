"""Tests for DINOv3 Vision Transformer JAX implementation."""

import unittest
import numpy as np
import jax
import jax.numpy as jnp
from dinov3_jax.utils.weight_converter import load_pytorch_weights
from jaxtorch import Context

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import JAX implementation
from dinov3_jax import vit_small
from dinov3_jax import DinoVisionTransformer as DinoVisionTransformerJAX

# Import PyTorch implementation for comparison
import torch
from dinov3.models.vision_transformer import vit_small as vit_small_pytorch
from dinov3.models.vision_transformer import DinoVisionTransformer as DinoVisionTransformerPT
jax.config.update("jax_default_matmul_precision", "highest")

vits_16_kwargs = dict(
    pos_embed_rope_dtype="fp32",         # default is "bf16"
    pos_embed_rope_rescale_coords=2,     # default is None
    embed_dim=384,                        # default is 768
    num_heads=6,                          # default is 12
    layerscale_init=1.0e-05,            # default is None
    norm_layer="layernormbf16",         # default is "layernorm"
    n_storage_tokens=4,                  # default is 0
    mask_k_bias=True                     # default is False
)
def mk_vit_small_pytorch():
    """Create a PyTorch Vision Transformer small model."""
    model = DinoVisionTransformerPT(**vits_16_kwargs)
    return model

class TestVisionTransformer(unittest.TestCase):
    """Test suite for Vision Transformer compatibility."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.seed = 42
        self.batch_size = 2
        self.img_size = 224
        self.channels = 3
        
        # Set random seeds
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        key = jax.random.PRNGKey(self.seed)
        self.key = key
    
    def test_model_initialization(self):
        """Test that JAX model can be initialized."""
        # Create JAX model
        model_jax = vit_small()
        
        # Initialize parameters
        params = model_jax.initialize(self.key)
        
        # Check that parameters are initialized
        self.assertIsNotNone(params)
        self.assertGreater(len(params), 0)
        
        # Check some key parameters exist
        self.assertIn("cls_token", params)
        self.assertIn("patch_embed.proj.weight", params)
        self.assertIn("blocks.0.attn.qkv.weight", params)
    
    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape."""
        # Create model
        model = vit_small()
        params = model.initialize(self.key)
        
        # Create context
        cx = Context(params, self.key, mode="eval")
        
        # Create dummy input
        x = jnp.ones((self.batch_size, self.channels, self.img_size, self.img_size))
        
        # Forward pass
        output = model(cx, x, is_training=False)
        
        # Check output shape (should be [batch_size, embed_dim])
        self.assertEqual(output.shape, (self.batch_size, 384))  # 384 is embed_dim for vit_small
    
    def test_weight_conversion(self):
        """Test weight conversion from PyTorch to JAX."""
        from dinov3_jax.utils import convert_pytorch_to_jax
        
        # Create PyTorch model
        model_pt = vit_small_pytorch()
        
        # Get PyTorch state dict
        state_dict_pt = model_pt.state_dict()
        
        # Convert to JAX format
        jax_params = convert_pytorch_to_jax(state_dict_pt)
        
        # Check that conversion maintains parameter count
        self.assertEqual(len(state_dict_pt), len(jax_params))
        
        # Check specific parameter conversions
        for key_pt in state_dict_pt.keys():
            # Find corresponding JAX key
            key_jax = key_pt  # Simple mapping for now
            self.assertIn(key_jax, jax_params)
            
            # Check shapes match
            shape_pt = state_dict_pt[key_pt].shape
            shape_jax = jax_params[key_jax].shape
            self.assertEqual(shape_pt, shape_jax, f"Shape mismatch for {key_pt}")
    
    def test_output_consistency(self):
        """Test that JAX and PyTorch models produce similar outputs with same weights."""
        # Create models
        model_jax = DinoVisionTransformerJAX(**vits_16_kwargs)
        model_pt = mk_vit_small_pytorch()
        state_dict = torch.load('/mnt/netdata/models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth')
        model_pt.load_state_dict(state_dict)
        
        # Initialize JAX model
        key = jax.random.PRNGKey(0)
        # params_jax = {k:jnp.array(v.cpu().numpy()) for k, v in state_dict.items()}
        cx = Context({}, key, mode="eval")
        model_jax.setup(cx)
        load_pytorch_weights(model_jax, state_dict, cx)

        # Create same input for both
        x_np = np.random.randn(self.batch_size, self.channels, self.img_size, self.img_size).astype(np.float32)
        x_jax = jnp.array(x_np)
        x_pt = torch.from_numpy(x_np)
        
        # Forward pass JAX
        output_jax = model_jax(cx, x_jax, is_training=False)
        output_jax_np = np.array(output_jax)

        # Forward pass PyTorch
        model_pt.eval()
        with torch.no_grad():
            output_pt = model_pt(x_pt, is_training=False)
        output_pt_np = np.array(output_pt)

        # Check output shapes match
        self.assertEqual(output_jax.shape[0], output_pt.shape[0])
        self.assertEqual(output_jax.shape[1], output_pt.shape[1])
        np.testing.assert_allclose(output_jax_np, output_pt_np, rtol=1e-5, atol=2e-6)

    def test_rope_embedding(self):
        """Test RoPE position embedding generation."""
        from dinov3_jax.layers import RopePositionEmbedding
        
        # Create RoPE module
        rope = RopePositionEmbedding(
            embed_dim=384,
            num_heads=6,
            base=100.0,
            dtype=jnp.float32
        )
        
        # Initialize
        params = {}
        cx = Context(params, self.key)
        rope.setup(cx)
        
        # Generate embeddings
        H, W = 14, 14  # For 224x224 image with patch_size=16
        sin, cos = rope(cx, H=H, W=W)
        
        # Check shapes (RoPE outputs D_head = embed_dim // num_heads)
        D_head = 384 // 6  # 64
        expected_shape = (H * W, D_head)
        self.assertEqual(sin.shape, expected_shape)
        self.assertEqual(cos.shape, expected_shape)
        
        # Check values are in valid range
        self.assertTrue(jnp.all(jnp.abs(sin) <= 1.0))
        self.assertTrue(jnp.all(jnp.abs(cos) <= 1.0))
    
    def test_multi_crop_forward(self):
        """Test forward pass with multiple crops (global and local)."""
        model = vit_small()
        params = model.initialize(self.key)
        cx = Context(params, self.key, mode="train")
        
        # Create multiple crops (1 global, 2 local)
        global_crop = jnp.ones((self.batch_size, self.channels, 224, 224))
        local_crop1 = jnp.ones((self.batch_size, self.channels, 96, 96))
        local_crop2 = jnp.ones((self.batch_size, self.channels, 96, 96))
        
        x_list = [global_crop, local_crop1, local_crop2]
        masks_list = [None, None, None]
        
        # Forward pass
        outputs = model.forward_features(cx, x_list, masks_list)
        
        # Check we get list of outputs
        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), 3)
        
        # Check each output has required keys
        for output in outputs:
            self.assertIn("x_norm_clstoken", output)
            self.assertIn("x_norm_patchtokens", output)
            self.assertIn("x_prenorm", output)


if __name__ == "__main__":
    unittest.main()