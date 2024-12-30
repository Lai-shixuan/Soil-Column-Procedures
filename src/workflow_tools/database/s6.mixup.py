import numpy as np

def extract_random_patch(image, label, patch_size):
    """Extract a random patch from image and label"""
    h, w = image.shape
    x = np.random.randint(0, w - patch_size)
    y = np.random.randint(0, h - patch_size)
    
    img_patch = image[y:y+patch_size, x:x+patch_size].copy()
    label_patch = label[y:y+patch_size, x:x+patch_size].copy()
    
    return img_patch, label_patch, (x, y)

def paste_patch(image, label, patch_img, patch_label, position):
    """Paste a patch onto the image and label"""
    x, y = position
    patch_size = patch_img.shape[0]
    
    image[y:y+patch_size, x:x+patch_size] = patch_img
    label[y:y+patch_size, x:x+patch_size] = patch_label
    
    return image, label

def random_patch_mixup(images, labels, patch_size_range=(32, 96), num_patches=3):
    """
    Perform random patch mixing augmentation
    
    Args:
        images: List of np.float32 grayscale images
        labels: List of corresponding labels
        patch_size_range: Tuple of (min_size, max_size) for random patch sizes
        num_patches: Number of patches to mix per image
    
    Returns:
        augmented_images: List of augmented images
        augmented_labels: List of augmented labels
    """
    num_images = len(images)
    augmented_images = []
    augmented_labels = []
    
    for i in range(num_images):
        # Copy original image and label
        aug_img = images[i].copy()
        aug_label = labels[i].copy()
        
        # Add random patches from other images
        for _ in range(num_patches):
            # Select random source image (different from current)
            source_idx = np.random.choice([j for j in range(num_images) if j != i])
            
            # Random patch size
            patch_size = np.random.randint(patch_size_range[0], patch_size_range[1])
            
            # Extract patch from source
            patch_img, patch_label, _ = extract_random_patch(
                images[source_idx], 
                labels[source_idx], 
                patch_size
            )
            
            # Find position in target image
            target_x = np.random.randint(0, aug_img.shape[1] - patch_size)
            target_y = np.random.randint(0, aug_img.shape[0] - patch_size)
            
            # Paste patch
            aug_img, aug_label = paste_patch(
                aug_img, aug_label, 
                patch_img, patch_label, 
                (target_x, target_y)
            )
            
        augmented_images.append(aug_img)
        augmented_labels.append(aug_label)
    
    return augmented_images, augmented_labels

# Example usage:
"""
# Assuming you have your images and labels in lists:
images = [...]  # List of np.float32 grayscale images
labels = [...]  # List of corresponding labels

# Perform augmentation
augmented_images, augmented_labels = random_patch_mixup(
    images, 
    labels,
    patch_size_range=(32, 96),
    num_patches=3
)
"""
