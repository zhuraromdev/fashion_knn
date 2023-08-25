import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Load pre-trained ResNet50 model with pre-processing
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

# Load multiple base models
num_models = 3  # You can adjust this number
base_models = [tf.keras.applications.ResNet50(weights='imagenet', include_top=False,
                                              pooling='avg', input_shape=(224, 224, 3))
               for _ in range(num_models)]
def create_resnet_model():
    return ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))


tf.get_logger().setLevel('ERROR')

def encode_image(img_array, model):
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    encoded = model.predict(img_array)
    return encoded

def load_index_images(index_dir, models):
    index_images = []
    index_files = [file for file in os.listdir(index_dir) if file != '.DS_Store']
    with ThreadPoolExecutor() as executor, \
            tqdm(total=len(index_files), desc="Processing Index Images", leave=False) as pbar:
        futures = []
        for img_file in index_files:
            img_path = os.path.join(index_dir, img_file)
            img_array = process_image(img_path)
            futures.append(executor.submit(encode_image_ensemble, img_array, models))
            pbar.update(1)
        index_images = [(img_file, future.result()) for img_file, future in zip(index_files, futures)]
    return index_images


def encode_image_ensemble(features, models):
    encoded_features = [model.predict(features) for model in models]
    mean_encoded_features = np.mean(encoded_features, axis=0)
    return mean_encoded_features


def visualize_neighbors(input_image_path, neighbor_file_names, index_dir):
    input_img = Image.open(input_image_path)
    plt.figure(figsize=(10, 5))

    # Display the input image
    plt.subplot(1, len(neighbor_file_names) + 1, 1)
    plt.imshow(input_img)
    plt.title("Input Image")
    plt.axis('off')

    # Display the neighbors
    for i, neighbor_name in enumerate(neighbor_file_names):
        neighbor_img_path = os.path.join(index_dir, neighbor_name)
        neighbor_img = Image.open(neighbor_img_path)
        plt.subplot(1, len(neighbor_file_names) + 1, i + 2)
        plt.imshow(neighbor_img)
        plt.title(f"Neighbor {i + 1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def process_image(img_path, target_size=(224, 224)):
    img = Image.open(img_path)
    img = img.resize(target_size)  # Resize to match the base model's input shape
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)  # Apply ResNet50 preprocessing
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def main():
    parser = argparse.ArgumentParser(description="Retrieve similar images using k-NN.")
    parser.add_argument("query_image", type=str, help="Path to the query image.")
    parser.add_argument("--top_n", type=int, default=5, help="Number of top candidates to retrieve.")
    args = parser.parse_args()

    # Directory paths for index and query images
    index_dir = 'data_test/index_small'
    query_dir = 'data_test/query'

    query_image_path = os.path.join(query_dir, args.query_image)
    top_n = args.top_n

    # Load index images in parallel with progress bar
    index_images = load_index_images(index_dir, base_models)

    # Encode index images using the base model and store in a dictionary
    index_encodings = {img_file: encode_image(img_array, base_model) for img_file, img_array in index_images}

    # Extract embeddings and filenames for k-NN
    embeddings = list(index_encodings.values())
    file_names = list(index_encodings.keys())

    # Convert embeddings to a numpy array
    embeddings_array = np.array(embeddings)

    # Reshape embeddings_array to a 2D array
    num_images = embeddings_array.shape[0]
    embedding_dim = np.prod(embeddings_array.shape[1:])
    embeddings_array_2d = embeddings_array.reshape(num_images, embedding_dim)

    # Initialize k-NN model
    knn_model = NearestNeighbors(n_neighbors=top_n, metric='cosine')
    knn_model.fit(embeddings_array_2d)

    # Process the query image and encode it
    query_img_array = process_image(query_image_path)
    query_encoding = encode_image(query_img_array, base_model)

    # Encode query image using ensemble of models
    query_encoding_ensemble = encode_image_ensemble(query_encoding, base_models)

    # Reshape query_encoding_ensemble to a 2D array
    query_encoding_ensemble_2d = query_encoding_ensemble.reshape(1, -1)

    # Find k-NN in the embedding space
    distances, indices = knn_model.kneighbors(query_encoding_ensemble_2d, return_distance=True)

    print(f"Query Image: {query_image_path}")
    print("Similar Images:")
    neighbor_file_names = [file_names[idx] for idx in indices[0]]
    for i, neighbor_name in enumerate(neighbor_file_names):
        print(f"{i + 1}. {neighbor_name} - Cosine Distance: {distances[0][i]}")

    # Visualize the input image and its top neighbors
    visualize_neighbors(query_image_path, neighbor_file_names[:top_n], index_dir)


if __name__ == "__main__":
    main()
