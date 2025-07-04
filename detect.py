import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw
import argparse
import yaml
from types import SimpleNamespace
import os
import numpy as np
from sklearn.cluster import KMeans

from models.detector import Detector
from models.steganography import SteganoNetwork
from models.template import LearnableTemplate

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return SimpleNamespace(**config)

def load_image(image_path, model_size=512):
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Store original size
    original_size = image.size
    
    # Calculate aspect ratio preserving size
    aspect_ratio = original_size[0] / original_size[1]
    if aspect_ratio > 1:
        new_size = (int(model_size * aspect_ratio), model_size)
    else:
        new_size = (model_size, int(model_size / aspect_ratio))
    
    # Create transform for model input
    transform = transforms.Compose([
        transforms.Resize(new_size, Image.Resampling.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Transform image for model
    tensor = transform(image).unsqueeze(0)
    
    return tensor, original_size, new_size, image

def detect_in_patches(image_tensor, detector, stego_net, template, patch_size=128, stride=64):
    """Process large images in patches for detection."""
    B, C, H, W = image_tensor.size()
    
    # Initialize confidence scores and locations
    patch_results = []
    
    # Process each patch
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            # Extract patch
            patch = image_tensor[:, :, y:y+patch_size, x:x+patch_size]
            
            # Extract template from patch
            with torch.no_grad():
                # Extract potential template from patch using steganography network
                extracted_template = stego_net.extract_template(patch)
                
                # Create detection input by concatenating original template with extracted one
                detection_input = torch.cat([template, extracted_template], dim=1)
                
                # Run detection
                confidence = detector(detection_input).item()
                
                # Calculate template similarity score
                template_similarity = F.mse_loss(template, extracted_template).item()
                
                # Calculate structural similarity
                ssim_score = 1 - template_similarity  # Higher is better
                
                # Calculate normalized confidence score with exponential penalty for high similarity
                norm_confidence = confidence * np.exp(-template_similarity * 2)
                
                patch_results.append({
                    'confidence': confidence,
                    'template_similarity': template_similarity,
                    'ssim_score': ssim_score,
                    'norm_confidence': norm_confidence,
                    'x': x,
                    'y': y,
                    'size': patch_size
                })
    
    # Sort patches by normalized confidence
    patch_results.sort(key=lambda x: x['norm_confidence'], reverse=True)
    
    # Calculate statistics
    confidences = np.array([p['confidence'] for p in patch_results])
    similarities = np.array([p['template_similarity'] for p in patch_results])
    ssim_scores = np.array([p['ssim_score'] for p in patch_results])
    norm_confidences = np.array([p['norm_confidence'] for p in patch_results])
    
    # Cluster analysis on normalized confidences
    # Reshape for clustering
    X = norm_confidences.reshape(-1, 1)
    
    # Perform k-means clustering with 2 clusters
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # Get cluster centers and sort them
    centers = kmeans.cluster_centers_.flatten()
    cluster_order = np.argsort(centers)
    
    # Calculate cluster statistics
    cluster_sizes = np.bincount(clusters)
    cluster_means = [np.mean(norm_confidences[clusters == i]) for i in range(2)]
    cluster_stds = [np.std(norm_confidences[clusters == i]) for i in range(2)]
    
    # Calculate separation between clusters
    cluster_separation = abs(cluster_means[1] - cluster_means[0]) / (cluster_stds[0] + cluster_stds[1] + 1e-6)
    
    # Identify suspicious patches (those in the high-confidence cluster)
    high_conf_cluster = cluster_order[1]  # Index of the cluster with higher confidence
    suspicious_patches = [
        p for i, p in enumerate(patch_results)
        if clusters[i] == high_conf_cluster and
        p['template_similarity'] < 0.2 and  # Relaxed similarity threshold
        p['ssim_score'] > 0.6  # Relaxed structural similarity threshold
    ]
    
    stats = {
        'mean_confidence': float(np.mean(confidences)),
        'max_confidence': float(np.max(confidences)),
        'min_confidence': float(np.min(confidences)),
        'std_confidence': float(np.std(confidences)),
        'mean_similarity': float(np.mean(similarities)),
        'min_similarity': float(np.min(similarities)),
        'mean_ssim': float(np.mean(ssim_scores)),
        'max_ssim': float(np.max(ssim_scores)),
        'max_norm_confidence': float(np.max(norm_confidences)),
        'mean_norm_confidence': float(np.mean(norm_confidences)),
        'cluster_separation': float(cluster_separation),
        'cluster_means': cluster_means,
        'cluster_stds': cluster_stds,
        'cluster_sizes': cluster_sizes.tolist(),
        'num_patches': len(confidences),
        'high_confidence_patches': suspicious_patches,
        'all_patches': patch_results
    }
    
    return stats

def visualize_detections(image, results, new_size, output_path):
    # Create a copy of the image for visualization
    draw_image = image.resize(new_size, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(draw_image)
    
    # Draw high confidence patches
    for patch in results['high_confidence_patches']:
        x, y = patch['x'], patch['y']
        size = patch['size']
        confidence = patch['confidence']
        
        # Draw rectangle with confidence score
        draw.rectangle([x, y, x + size, y + size], outline='red', width=2)
        draw.text((x + 5, y + 5), f"{confidence:.2f}", fill='red')
    
    # Save visualization
    draw_image.save(output_path)
    return output_path

def analyze_detection_results(results, min_high_conf_patches=1):
    """Analyze detection results using cluster analysis."""
    
    # Count high confidence patches with good template similarity
    high_conf_patches = results['high_confidence_patches']
    num_high_conf = len(high_conf_patches)
    
    # Get cluster statistics
    cluster_separation = results['cluster_separation']
    cluster_means = results['cluster_means']
    cluster_stds = results['cluster_stds']
    cluster_sizes = results['cluster_sizes']
    
    # Calculate confidence and similarity metrics
    max_norm_conf = results['max_norm_confidence']
    mean_norm_conf = results['mean_norm_confidence']
    min_sim = results['min_similarity']
    max_ssim = results['max_ssim']
    
    # Calculate cluster-based confidence score
    cluster_score = min(1.0, cluster_separation / 2.0)  # Normalize to 0-1 range
    
    # Calculate patch distribution score
    size_ratio = min(cluster_sizes) / max(cluster_sizes)
    distribution_score = 1.0 - size_ratio  # Higher score if clusters are imbalanced
    
    # Multi-criteria decision with weights
    confidence_weight = 0.4
    separation_weight = 0.3
    distribution_weight = 0.3
    
    # Calculate weighted score
    weighted_score = (
        confidence_weight * max_norm_conf +
        separation_weight * cluster_score +
        distribution_weight * distribution_score
    )
    
    # Decision criteria using multiple factors
    template_detected = (
        (num_high_conf >= min_high_conf_patches) and     # Must have suspicious patches
        (cluster_separation > 1.5) and                   # Clusters must be well-separated
        (max_norm_conf > 0.4) and                        # Must have good normalized confidence
        (distribution_score > 0.3) and                   # Must have imbalanced clusters
        (weighted_score > 0.5)                           # Must have good overall score
    )
    
    # Convert to 0-100 range for display
    confidence_score = min(100, max(0, weighted_score * 100))
    
    return {
        'is_modified': template_detected,
        'num_high_conf_patches': num_high_conf,
        'max_confidence': results['max_confidence'],
        'mean_confidence': results['mean_confidence'],
        'std_confidence': results['std_confidence'],
        'min_similarity': min_sim,
        'mean_similarity': results['mean_similarity'],
        'max_ssim': max_ssim,
        'mean_ssim': results['mean_ssim'],
        'max_norm_confidence': max_norm_conf,
        'mean_norm_confidence': mean_norm_conf,
        'cluster_separation': cluster_separation,
        'distribution_score': distribution_score,
        'confidence_score': confidence_score
    }

def main():
    parser = argparse.ArgumentParser(description='Detect if image has hidden face')
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_epoch_100.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to config file')
    parser.add_argument('--threshold', type=float, default=0.8,
                       help='Detection threshold (0-1)')
    parser.add_argument('--model_size', type=int, default=512,
                       help='Size to process image at (default: 512)')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize models
    detector = Detector(
        in_channels=6  # 3 channels each for image and potential template
    ).to(device)
    
    stego_net = SteganoNetwork(
        in_channels=3,
        num_blocks=4
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    detector.load_state_dict(checkpoint['detector'])
    stego_net.load_state_dict(checkpoint['stego_net'])
    
    # Load learnable template - it's stored as state dict, need to create model first
    template_model = LearnableTemplate(size=128).to(device)
    template_model.load_state_dict(checkpoint['learnable_template'])
    template_model.eval()
    with torch.no_grad():
        template = template_model(1)  # Generate one template

    # Set models to eval mode
    detector.eval()
    stego_net.eval()

    # Load input image
    input_image, original_size, new_size, pil_image = load_image(args.input, args.model_size)
    input_image = input_image.to(device)
    
    print(f"Processing at size: {new_size[0]}x{new_size[1]}")
    
    # Run patch-based detection
    results = detect_in_patches(input_image, detector, stego_net, template)
    
    # Analyze results with strict criteria
    analysis = analyze_detection_results(results, args.threshold)
    
    # Visualize detections
    vis_path = os.path.splitext(args.input)[0] + '_detection.jpg'
    visualize_detections(pil_image, results, new_size, vis_path)
    
    # Print detailed results
    print("\nDetection Results:")
    print("-----------------")
    print(f"Mean confidence: {analysis['mean_confidence']:.4f}")
    print(f"Max confidence: {analysis['max_confidence']:.4f}")
    print(f"Normalized confidence (max): {analysis['max_norm_confidence']:.4f}")
    print(f"Normalized confidence (mean): {analysis['mean_norm_confidence']:.4f}")
    print(f"Cluster separation: {analysis['cluster_separation']:.4f}")
    print(f"Distribution score: {analysis['distribution_score']:.4f}")
    print(f"High confidence patches: {analysis['num_high_conf_patches']}")
    print(f"Template similarity (min): {analysis['min_similarity']:.4f}")
    print(f"Template similarity (mean): {analysis['mean_similarity']:.4f}")
    print(f"Structural similarity (max): {analysis['max_ssim']:.4f}")
    print(f"Overall detection score: {analysis['confidence_score']:.1f}/100")
    
    # Print conclusion
    if analysis['is_modified']:
        print("\n⚠️  This image appears to be MODIFIED with a hidden face template")
        print(f"Found {analysis['num_high_conf_patches']} suspicious patches")
        print(f"Highest detection confidence: {analysis['max_confidence']*100:.2f}%")
        print(f"Highest normalized confidence: {analysis['max_norm_confidence']:.4f}")
        print(f"Cluster separation: {analysis['cluster_separation']:.4f}")
        print(f"Distribution score: {analysis['distribution_score']:.4f}")
        print(f"Overall detection score: {analysis['confidence_score']:.1f}/100")
    else:
        print("\n✅ This image appears to be ORIGINAL (no hidden face detected)")
        print(f"Highest detection confidence: {analysis['max_confidence']*100:.2f}%")
        print(f"Highest normalized confidence: {analysis['max_norm_confidence']:.4f}")
        print(f"Cluster separation: {analysis['cluster_separation']:.4f}")
        print(f"Distribution score: {analysis['distribution_score']:.4f}")
        print(f"Overall detection score: {analysis['confidence_score']:.1f}/100")
    
    print(f"\nVisualization saved to: {vis_path}")

if __name__ == '__main__':
    main() 