#!/usr/bin/env python3
"""
Compute Matches Script for Siamese Capsule Network
Loads the trained model and finds matching pet images from the preprocessed gallery.
"""

import sys
import json
import os
import base64
from io import BytesIO
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Conv2D, MultiHeadAttention, LayerNormalization
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import warnings
warnings.filterwarnings('ignore')

# Configure TensorFlow to use all CPU cores
tf.config.threading.set_intra_op_parallelism_threads(cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(cpu_count())

# ============================================================================
# CUSTOM LAYERS DEFINITIONS (Required for model loading)
# ============================================================================

def squash(vectors, axis=-1):
    """Squashing function for capsule networks with improved numerical stability."""
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    epsilon = 1e-7
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + epsilon)
    return scale * vectors

def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    """Safe norm calculation to avoid numerical issues."""
    squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=keep_dims)
    return tf.sqrt(squared_norm + epsilon)

class CapsuleLayer(Layer):
    """Basic capsule layer implementation."""
    def __init__(self, num_capsules, dim_capsules, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.routings = routings
    
    def build(self, input_shape):
        super(CapsuleLayer, self).build(input_shape)
    
    def call(self, inputs, training=None):
        return inputs
    
    def get_config(self):
        config = super(CapsuleLayer, self).get_config()
        config.update({
            'num_capsules': self.num_capsules,
            'dim_capsules': self.dim_capsules,
            'routings': self.routings
        })
        return config

class PrimaryCapsule(Layer):
    """Primary capsule layer implementation."""
    def __init__(self, dim_capsules, n_channels, kernel_size, strides, padding, **kwargs):
        super(PrimaryCapsule, self).__init__(**kwargs)
        self.dim_capsules = dim_capsules
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        
    def build(self, input_shape):
        self.conv = Conv2D(
            filters=self.dim_capsules * self.n_channels,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            activation='relu',
            name=f'{self.name}_conv'
        )
        self.conv.build(input_shape)
        super(PrimaryCapsule, self).build(input_shape)
        
    def call(self, inputs, training=None):
        outputs = self.conv(inputs)
        batch_size = tf.shape(outputs)[0]
        outputs = tf.reshape(outputs, [batch_size, -1, self.dim_capsules])
        return squash(outputs, axis=-1)
    
    def get_config(self):
        config = super(PrimaryCapsule, self).get_config()
        config.update({
            'dim_capsules': self.dim_capsules,
            'n_channels': self.n_channels,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding
        })
        return config

class EnhancedCapsuleLayer(CapsuleLayer):
    """Enhanced Capsule Layer with self-attention mechanism."""
    
    def __init__(self, num_capsules, dim_capsules, routings=3, attention_heads=4, 
                 use_attention=True, kernel_initializer='glorot_uniform', **kwargs):
        super(EnhancedCapsuleLayer, self).__init__(
            num_capsules, dim_capsules, routings, **kwargs
        )
        self.attention_heads = attention_heads
        self.use_attention = use_attention
        self.kernel_initializer = kernel_initializer
        
    def build(self, input_shape):
        super().build(input_shape)
        self.dim_in = int(input_shape[-1])
        self.W = self.add_weight(
            shape=(self.num_capsules, self.dim_in, self.dim_capsules),
            initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg', distribution='uniform'),
            name=f"{self.name}_W"
        )
        self.b_caps = self.add_weight(
            shape=(self.num_capsules, self.dim_capsules),
            initializer='zeros',
            name=f"{self.name}_b"
        )
        self.gamma = self.add_weight(
            shape=(1,), initializer=tf.keras.initializers.constant(3.0), name=f"{self.name}_gamma"
        )
        self.last_c_entropy = None
        if self.use_attention:
            unique_prefix = f"{self.name}_"
            self.attention_layer = MultiHeadAttention(
                num_heads=self.attention_heads,
                key_dim=self.dim_capsules,
                name=f'{unique_prefix}capsule_attention'
            )
            self.attention_norm = LayerNormalization(name=f'{unique_prefix}attention_norm')
    
    def enhanced_dynamic_routing(self, u_hat, training):
        batch_size = tf.shape(u_hat)[0]
        input_num_capsules = tf.shape(u_hat)[1]
        num_output_capsules = tf.shape(u_hat)[2] 
        capsule_dim = tf.shape(u_hat)[3]
        
        b = tf.zeros([batch_size, input_num_capsules, num_output_capsules, 1])
        
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=2)
            s = tf.reduce_sum(c * u_hat, axis=1, keepdims=True)
            v = squash(s, axis=-1)
            if i < self.routings - 1:
                agreement = tf.reduce_sum(u_hat * v, axis=-1, keepdims=True)
                b = b + agreement
        
        c_final = tf.nn.softmax(b, axis=2)
        entropy = -tf.reduce_sum(c_final * tf.math.log(c_final + 1e-9), axis=2)
        norm_entropy = entropy / tf.math.log(tf.cast(num_output_capsules, tf.float32) + 1e-9)
        self.last_c_entropy = tf.reduce_mean(norm_entropy)
        
        return tf.squeeze(v, axis=1)
    
    def call(self, inputs, training=None):
        u_hat = tf.einsum('b n d, c d h -> b n c h', inputs, self.W) + self.bias_expand()
        if training:
            u_hat += tf.random.normal(tf.shape(u_hat), stddev=0.05)
            global_std = tf.math.reduce_std(u_hat)
            def add_jitter():
                return u_hat + tf.random.normal(tf.shape(u_hat), stddev=0.1)
            u_hat = tf.cond(global_std < 0.02, add_jitter, lambda: u_hat)
        routed = self.enhanced_dynamic_routing(u_hat, training=training if training is not None else False)
        if self.use_attention:
            attended = self.attention_layer(routed, routed, training=training)
            routed = self.attention_norm(attended + routed, training=training)
        routed = squash(routed, axis=-1) * self.gamma
        return routed

    def bias_expand(self):
        return tf.reshape(self.b_caps, (1, 1, self.num_capsules, self.dim_capsules))

    def get_last_routing_entropy(self):
        return self.last_c_entropy
    
    def get_config(self):
        config = super(EnhancedCapsuleLayer, self).get_config()
        config.update({
            'attention_heads': self.attention_heads,
            'use_attention': self.use_attention
        })
        return config


# ============================================================================
# DISTANCE FUNCTIONS (Required for Lambda layers in model)
# ============================================================================

def cosine_distance(vectors):
    """Compute cosine distance between embedding pairs."""
    x, y = vectors
    x_norm = K.l2_normalize(x, axis=1)
    y_norm = K.l2_normalize(y, axis=1)
    cosine_sim = K.sum(x_norm * y_norm, axis=1)
    cosine_dist = 1.0 - cosine_sim
    return K.clip(cosine_dist, 0.0, 1.0)

def euclidean_distance(vectors):
    """Euclidean distance for raw embeddings."""
    x, y = vectors
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    dist = K.sqrt(K.maximum(sum_square, K.epsilon()))
    return dist

def pearson_correlation_distance(vectors):
    """Pearson correlation-based distance."""
    x, y = vectors
    mean_x = K.mean(x, axis=1, keepdims=True)
    mean_y = K.mean(y, axis=1, keepdims=True)
    x_centered = x - mean_x
    y_centered = y - mean_y
    numerator = K.sum(x_centered * y_centered, axis=1)
    denominator_x = K.sqrt(K.sum(K.square(x_centered), axis=1) + K.epsilon())
    denominator_y = K.sqrt(K.sum(K.square(y_centered), axis=1) + K.epsilon())
    correlation = numerator / (denominator_x * denominator_y + K.epsilon())
    correlation = K.clip(correlation, -1.0, 1.0)
    similarity = (correlation + 1.0) * 0.5
    distance = 1.0 - similarity
    distance = K.clip(distance, 0.0, 1.0)
    distance_stabilized = 0.1 + 0.8 * K.sigmoid(6.0 * (distance - 0.5))
    return distance_stabilized

# ============================================================================
# LOSS AND METRIC FUNCTIONS (Required for compiled model)
# ============================================================================

def loss_fn(y_true, y_pred):
    """Enhanced contrastive loss function."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    batch = tf.shape(y_true)[0]
    y_true = tf.reshape(y_true, (batch, 1))
    y_pred = tf.reshape(y_pred, (batch, -1))
    y_pred = tf.reduce_mean(y_pred, axis=1, keepdims=True)
    
    margin = 0.5
    label_smoothing = 0.0
    alpha = 0.1
    
    y_true_smooth = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
    margin_tf = tf.constant(margin, dtype=tf.float32)
    
    pos_weight = 1.0
    neg_weight = 1.35
    
    pos_loss = pos_weight * y_true_smooth * K.square(y_pred)
    neg_loss = neg_weight * (1.0 - y_true_smooth) * K.square(K.maximum(margin_tf - y_pred, 0.0))
    contrastive = K.mean(pos_loss + neg_loss)
    
    k = tf.maximum(1, tf.cast(0.1 * tf.cast(batch, tf.float32), tf.int32))
    neg_residual = K.maximum(margin_tf - y_pred, 0.0) * (1.0 - y_true_smooth)
    topk_vals, _ = tf.math.top_k(tf.reshape(neg_residual, (-1,)), k=k)
    hard_neg_loss = K.mean(K.square(topk_vals))
    
    sep_reg = K.mean(K.exp(-5.0 * K.square(y_pred - margin_tf * 0.5)))
    focal_weight = K.square(K.abs(y_true_smooth - K.sigmoid(1.0 - y_pred)))
    focal_loss = K.mean(focal_weight * (pos_loss + neg_loss))
    
    return contrastive + 0.3 * hard_neg_loss + alpha * sep_reg + 0.1 * focal_loss

def accuracy_metric(y_true, y_pred):
    """Accuracy metric for distance predictions."""
    threshold = 0.5
    y_pred_normalized = y_pred
    predictions = K.cast(y_pred_normalized < threshold, K.floatx())
    y_true_cast = K.cast(y_true, K.floatx())
    correct = K.cast(K.equal(y_true_cast, predictions), K.floatx())
    accuracy = K.mean(correct)
    accuracy = tf.where(tf.math.is_finite(accuracy), accuracy, 0.5)
    return K.clip(accuracy, 0.0, 1.0)

# ============================================================================
# MATCHING ENGINE
# ============================================================================

class PetMatchingEngine:
    """Engine to compute similarity matches using the trained Siamese model."""
    
    def __init__(self, model_path, debug=False, batch_size=32, num_workers=None):
        self.model_path = model_path
        self.debug = debug
        self.model = None
        self.embedding_model = None
        self.debug_info = []
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers else cpu_count()
        self.log(f"Initialized with {self.num_workers} workers and batch size {self.batch_size}")
        
    def log(self, msg):
        """Log debug messages."""
        if self.debug:
            self.debug_info.append(msg)
            print(f"[DEBUG] {msg}", file=sys.stderr)
    
    def load_model(self):
        """Load the Siamese model with custom objects."""
        self.log(f"Loading model from: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Define custom objects for model loading
        custom_objects = {
            'squash': squash,
            'safe_norm': safe_norm,
            'CapsuleLayer': CapsuleLayer,
            'PrimaryCapsule': PrimaryCapsule,
            'EnhancedCapsuleLayer': EnhancedCapsuleLayer,
            'cosine_distance': cosine_distance,
            'euclidean_distance': euclidean_distance,
            'pearson_correlation_distance': pearson_correlation_distance,
            'loss_fn': loss_fn,
            'accuracy_metric': accuracy_metric
        }
        
        try:
            # Load with safe_mode=False to allow Lambda layers
            # compile=False to skip loading custom metrics (not needed for inference)
            self.model = load_model(self.model_path, custom_objects=custom_objects, safe_mode=False, compile=False)
            self.log(f"Model loaded successfully (without compilation)")
            self.log(f"Model inputs: {len(self.model.inputs)}, outputs: {len(self.model.outputs)}")
            
            # Extract the base network for embedding extraction
            # The siamese model has 2 inputs (anchor, positive) and 1 output (distance)
            # We need to access the base network that produces embeddings
            if len(self.model.layers) > 2:
                # Find the base network layer (MobileNetV2_CapsNet_Hybrid)
                for layer in self.model.layers:
                    if 'MobileNetV2' in layer.name or 'Hybrid' in layer.name:
                        self.embedding_model = layer
                        self.log(f"Found embedding model: {layer.name}")
                        break
            
            if self.embedding_model is None:
                self.log("Using full model for embeddings")
                # If we can't find the base network, create a wrapper
                # that uses just the first input
                from tensorflow.keras.models import Model
                self.embedding_model = Model(
                    inputs=self.model.inputs[0],
                    outputs=self.model.layers[2].output[1]  # Get embeddings from base network
                )
            
        except Exception as e:
            self.log(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_image(self, img_path, target_size=(224, 224)):
        """Load and preprocess an image for the model."""
        self.log(f"Preprocessing image: {img_path}")
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # Load image
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        
        # Normalize to [0, 1] range (MobileNetV2 preprocessing)
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def get_embedding(self, img_path):
        """Extract embedding vector from an image."""
        img_array = self.preprocess_image(img_path)
        
        # Get embeddings
        output = self.embedding_model.predict(img_array, verbose=0)
        
        # Handle different output formats
        if isinstance(output, list):
            # If output is [capsules, embeddings], take embeddings (second element)
            embedding = output[1]
        else:
            embedding = output
        
        # Flatten to 1D vector
        embedding = embedding.flatten()
        
        return embedding
    
    def get_embeddings_batch(self, img_paths):
        """Extract embeddings for multiple images in batch (faster)."""
        if len(img_paths) == 0:
            return np.array([]), []
        
        # Preprocess all images
        img_arrays = []
        valid_paths = []
        for img_path in img_paths:
            try:
                img_array = self.preprocess_image(img_path)
                img_arrays.append(img_array)
                valid_paths.append(img_path)
            except Exception as e:
                self.log(f"Error preprocessing {img_path}: {str(e)}")
                continue
        
        if len(img_arrays) == 0:
            return np.array([]), []
        
        # Stack into batch
        batch = np.vstack(img_arrays)
        
        # Get embeddings in batch
        output = self.embedding_model.predict(batch, verbose=0, batch_size=self.batch_size)
        
        # Handle different output formats
        if isinstance(output, list):
            embeddings = output[1]
        else:
            embeddings = output
        
        # Flatten each embedding
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
        
        return embeddings, valid_paths
    
    def cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors."""
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)
        
        # Compute cosine similarity
        similarity = np.dot(vec1_norm, vec2_norm)
        similarity = (similarity + 1.0) / 2.0  # Map from [-1, 1] to [0, 1]
        return float(similarity)
    
    def generate_thumbnail(self, img_path, size=(150, 150)):
        """
        Generate a base64-encoded thumbnail of an image.
        
        Args:
            img_path: Path to the image
            size: Thumbnail size (width, height)
        
        Returns:
            Base64-encoded JPEG thumbnail string
        """
        try:
            img = Image.open(img_path)
            img.thumbnail(size, Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save to bytes buffer
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            buffer.seek(0)
            
            # Encode to base64
            img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            return img_base64
        except Exception as e:
            self.log(f"Error generating thumbnail for {img_path}: {str(e)}")
            return None
    
    def generate_thumbnails_parallel(self, img_paths, size=(150, 150)):
        """Generate thumbnails for multiple images in parallel."""
        thumbnails = {}
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all thumbnail generation tasks
            future_to_path = {executor.submit(self.generate_thumbnail, path, size): path 
                             for path in img_paths}
            
            # Collect results as they complete
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    thumb = future.result()
                    thumbnails[path] = thumb
                except Exception as e:
                    self.log(f"Thumbnail generation failed for {path}: {str(e)}")
                    thumbnails[path] = None
        
        return thumbnails
    
    def find_matches(self, query_image_path, gallery_dir, pet_types, top_k=10):
        """
        Find top-k matching images from the gallery.
        
        Args:
            query_image_path: Path to the query image
            gallery_dir: Directory containing preprocessed images
            pet_types: Comma-separated pet types (e.g., "cat,dog")
            top_k: Number of top matches to return
        
        Returns:
            List of matches with similarity scores
        """
        self.log(f"Finding matches for: {query_image_path}")
        self.log(f"Gallery directory: {gallery_dir}")
        self.log(f"Pet types: {pet_types}")
        self.log(f"Top K: {top_k}")
        
        # Parse pet types
        pet_types_list = [pt.strip().title() for pt in pet_types.split(',')]
        self.log(f"Parsed pet types: {pet_types_list}")
        
        # Get query embedding
        try:
            query_embedding = self.get_embedding(query_image_path)
            self.log(f"Query embedding shape: {query_embedding.shape}")
            self.log(f"Query embedding stats: mean={query_embedding.mean():.4f}, std={query_embedding.std():.4f}")
        except Exception as e:
            raise Exception(f"Failed to extract query embedding: {str(e)}")
        
        # Collect gallery images
        gallery_images = []
        for pet_type in pet_types_list:
            type_dir = os.path.join(gallery_dir, pet_type + 's')  # Cats, Dogs
            if not os.path.exists(type_dir):
                self.log(f"Warning: Directory not found: {type_dir}")
                continue
            
            self.log(f"Scanning directory: {type_dir}")
            
            for filename in os.listdir(type_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    img_path = os.path.join(type_dir, filename)
                    gallery_images.append({
                        'path': img_path,
                        'filename': filename,
                        'type': pet_type
                    })
        
        self.log(f"Found {len(gallery_images)} gallery images")
        
        if len(gallery_images) == 0:
            raise Exception(f"No gallery images found in {gallery_dir}")
        
        # Process gallery images in batches for efficiency
        gallery_paths = [item['path'] for item in gallery_images]
        gallery_embeddings_list = []
        
        self.log(f"Processing gallery images in batches of {self.batch_size}...")
        
        # Process in batches
        for i in range(0, len(gallery_paths), self.batch_size):
            batch_paths = gallery_paths[i:i + self.batch_size]
            try:
                batch_embeddings, valid_paths = self.get_embeddings_batch(batch_paths)
                gallery_embeddings_list.extend(batch_embeddings)
                
                self.log(f"Processed batch {i // self.batch_size + 1}/{(len(gallery_paths) + self.batch_size - 1) // self.batch_size}")
            except Exception as e:
                self.log(f"Error processing batch starting at index {i}: {str(e)}")
                # Fallback to individual processing for this batch
                for path in batch_paths:
                    try:
                        emb = self.get_embedding(path)
                        gallery_embeddings_list.append(emb)
                    except Exception as e2:
                        self.log(f"Error processing {path}: {str(e2)}")
                        continue
        
        if len(gallery_embeddings_list) == 0:
            raise Exception("Failed to extract any gallery embeddings")
        
        self.log(f"Successfully extracted {len(gallery_embeddings_list)} embeddings")
        
        # Compute similarities using vectorized operations
        gallery_embeddings = np.array(gallery_embeddings_list)
        
        # Normalize embeddings for cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        gallery_norms = gallery_embeddings / (np.linalg.norm(gallery_embeddings, axis=1, keepdims=True) + 1e-10)
        
        # Compute all similarities at once (vectorized)
        similarities = np.dot(gallery_norms, query_norm)
        similarities = (similarities + 1.0) / 2.0  # Map from [-1, 1] to [0, 1]
        
        # Create matches list
        matches = []
        for i, (gallery_item, similarity) in enumerate(zip(gallery_images[:len(similarities)], similarities)):
            matches.append({
                'path': gallery_item['path'],
                'filename': gallery_item['filename'],
                'type': gallery_item['type'],
                'similarity': float(similarity),
                'distance': float(1.0 - similarity)
            })
        
        self.log(f"Successfully computed {len(matches)} similarities")
        
        # Sort by similarity (descending)
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Get top-k matches
        top_k_matches = matches[:top_k]
        
        # Generate thumbnails in parallel for top-k matches
        self.log(f"Generating thumbnails for top {top_k} matches in parallel...")
        top_k_paths = [match['path'] for match in top_k_matches]
        thumbnails = self.generate_thumbnails_parallel(top_k_paths)
        
        # Build final results with ranks and thumbnails
        top_matches = []
        for rank, match in enumerate(top_k_matches, start=1):
            top_matches.append({
                'rank': rank,
                'path': match['path'],
                'filename': match['filename'],
                'type': match['type'],
                'similarity': match['similarity'],
                'distance': match['distance'],
                'thumb_base64': thumbnails.get(match['path'])
            })
        
        self.log(f"Top match similarity: {top_matches[0]['similarity']:.4f}" if top_matches else "No matches")
        
        return top_matches


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    # Parse command line arguments
    if len(sys.argv) < 5:
        print(json.dumps({
            'ok': False,
            'error': 'Usage: compute_matches.py <query_image> <pet_types> <gallery_dir> <top_k> [--debug]'
        }))
        sys.exit(1)
    
    query_image = sys.argv[1]
    pet_types = sys.argv[2]
    gallery_dir = sys.argv[3]
    top_k = int(sys.argv[4])
    debug = '--debug' in sys.argv
    
    # Get model path (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(os.path.dirname(script_dir), 'model', 'final_best_model.keras')
    
    result = {
        'ok': False,
        'matches': [],
        'debug': [],
        'attempts': []
    }
    
    try:
        # Initialize engine with optimized settings
        num_workers = cpu_count()
        batch_size = min(32, max(8, num_workers * 2))  # Adaptive batch size
        
        engine = PetMatchingEngine(
            model_path, 
            debug=debug, 
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        if debug:
            result['debug'].append(f"Using {num_workers} CPU cores with batch size {batch_size}")
        
        # Load model
        engine.load_model()
        
        # Find matches
        matches = engine.find_matches(query_image, gallery_dir, pet_types, top_k)
        
        # Format results for PHP
        formatted_matches = []
        for match in matches:
            formatted_matches.append({
                'rank': match['rank'],
                'path': match['path'].replace('\\', '/'),  # Convert to forward slashes for PHP
                'filename': match['filename'],
                'type': match['type'],
                'similarity': round(match['similarity'], 4),
                'distance': round(match['distance'], 4),
                'confidence': round(match['similarity'] * 100, 2),  # Convert to percentage
                'score': round(match['similarity'] * 100, 2),  # Same as confidence, for JS compatibility
                'thumb_base64': match['thumb_base64']  # Base64 encoded thumbnail
            })
        
        result['ok'] = True
        result['matches'] = formatted_matches
        result['debug'] = engine.debug_info
        
    except Exception as e:
        result['error'] = str(e)
        result['debug'] = [f"Exception: {str(e)}"]
        if debug:
            import traceback
            result['debug'].append(traceback.format_exc())
    
    # Output JSON result
    print(json.dumps(result, indent=2 if debug else None))


if __name__ == '__main__':
    main()
