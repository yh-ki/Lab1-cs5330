import cv2
import numpy as np
import glob
import gradio as gr
from skimage.metrics import structural_similarity as ssim
from scipy.spatial import KDTree


# Load and preprocess the main image
def preprocess_image(image_path, k=8, tile_size=32):
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    new_h = (h // tile_size) * tile_size
    new_w = (w // tile_size) * tile_size
    image = cv2.resize(image, (new_w, new_h))  # Resize to ensure proportionality

    # Apply color quantization using K-means clustering
    img = image.reshape((-1, 3))
    img = np.float32(img)

    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(img, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    quantized_image = centers[labels.flatten()].reshape(image.shape)

    return quantized_image


# Define tile paths
tile_paths = glob.glob('face/*.png')  # Folder containing tile images
tiles = [cv2.imread(tile) for tile in tile_paths]

# Precompute average colors for tiles
tile_avgs = np.array([np.mean(tile, axis=(0, 1)) for tile in tiles])
kd_tree = KDTree(tile_avgs)


def get_best_tile(avg_color):
    _, index = kd_tree.query(avg_color)
    return tiles[index]


def generate_mosaic(image, tile_size):
    h, w, _ = image.shape
    mosaic = np.zeros_like(image)

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = image[y:y + tile_size, x:x + tile_size]
            avg_color = np.mean(tile, axis=(0, 1), dtype=int)
            best_tile = get_best_tile(avg_color)
            best_tile_resized = cv2.resize(best_tile, (tile_size, tile_size))
            mosaic[y:y + tile_size, x:x + tile_size] = best_tile_resized

    return mosaic


def compute_similarity(original, mosaic):
    """Compute similarity using Mean Squared Error (MSE) and SSIM."""
    mse = np.mean((original - mosaic) ** 2)
    gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_mosaic = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
    ssim_score = ssim(gray_original, gray_mosaic)
    return f"MSE: {mse:.2f}, SSIM: {ssim_score:.4f}"

def mosaic_interface(image, tile_size):
    processed_image = preprocess_image(image)
    mosaic = generate_mosaic(processed_image, tile_size)
    similarity = compute_similarity(processed_image, mosaic)
    return mosaic, similarity


# Gradio interface
iface = gr.Interface(
    fn=mosaic_interface,
    inputs=[gr.Image(type='filepath'), gr.Radio([8, 16, 32], label="Tile Size")],
    outputs=[gr.Image(), gr.Text()],
    title="Interactive Image Mosaic Generator",
    description="Upload an image to generate a mosaic-style reconstruction. Choose the tile size from the given options."
)

if __name__ == "__main__":
    iface.launch()