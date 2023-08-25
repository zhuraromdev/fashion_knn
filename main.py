import argparse
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


class ImageRetrievalSystem:
    def __init__(self, index_dir, query_dir, embeddings_file='embeddings_original.pkl'):
        self.index_dir = index_dir
        self.query_dir = query_dir
        self.embeddings_file = embeddings_file

        self.base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
        tf.get_logger().setLevel('ERROR')

        self.index_encodings = self.load_embeddings() if os.path.exists(embeddings_file) else None

    def encode_image(self, img_array):
        # print("Image array shape before preprocessing:", img_array.shape)  # Debugging
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        encoded = self.base_model.predict(img_array)
        return encoded

    def load_embeddings(self):
        with open(self.embeddings_file, 'rb') as f:
            return pickle.load(f)

    def save_embeddings(self, index_encodings):
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(index_encodings, f)

    def process_image(self, img_path):
        img = Image.open(img_path)
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = img_array[..., ::-1]  # Ensure correct channel order
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def load_index_images(self):
        index_images = []
        index_files = [file for file in os.listdir(self.index_dir) if file != '.DS_Store']
        with ThreadPoolExecutor() as executor, \
                tqdm(total=len(index_files), desc="Processing Index Images", leave=False) as pbar:
            futures = []
            for img_file in index_files:
                img_path = os.path.join(self.index_dir, img_file)
                img_array = self.process_image(img_path)
                futures.append(executor.submit(self.encode_image, img_array))
                pbar.update(1)
            index_images = [(img_file, future.result()) for img_file, future in zip(index_files, futures)]
        return index_images

    def get_index_encodings(self):
        if self.index_encodings is None:
            if os.path.exists(self.embeddings_file):
                self.index_encodings = self.load_embeddings()
            else:
                index_images = self.load_index_images()
                self.index_encodings = {img_file: img_array for img_file, img_array in index_images}
                self.save_embeddings(self.index_encodings)
        return self.index_encodings

    def visualize_neighbors(self, input_image_path, neighbor_file_names, index_dir):
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

    def main(self, query_image, top_n):
        query_image_path = os.path.join(self.query_dir, query_image)

        # Load index images in parallel with progress bar
        index_images = self.load_index_images()

        # Encode index images and store in a dictionary
        # index_encodings = {img_file: img_array for img_file, img_array in index_images}
        index_encodings = self.get_index_encodings()

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
        query_img_array = self.process_image(query_image_path)
        query_encoding = self.encode_image(query_img_array)

        # Reshape query_encoding to a 2D array
        query_encoding_2d = query_encoding.reshape(1, -1)

        # Find k-NN in the embedding space
        distances, indices = knn_model.kneighbors(query_encoding_2d, return_distance=True)

        print(f"Query Image: {query_image_path}")
        print("Similar Images:")
        neighbor_file_names = [file_names[idx] for idx in indices[0]]
        for i, neighbor_name in enumerate(neighbor_file_names):
            print(f"{i + 1}. {neighbor_name} - Cosine Distance: {distances[0][i]}")

        # Visualize the input image and its top neighbors
        self.visualize_neighbors(query_image_path, neighbor_file_names[:top_n], self.index_dir)


class AugmentedImageRetrievalSystem(ImageRetrievalSystem):
    def augment_and_expand_dataset(self, index_images, augmentation_factor):
        augmented_images = []
        for _, img_array in index_images:
            augmented_images.append(img_array)
            augmented_images.extend(self.augment_image(img_array) for _ in range(augmentation_factor))

        return list(zip([img_file for img_file, _ in index_images], augmented_images))

    def augment_image(self, image):
        # Apply data augmentation transformations here
        # You can use libraries like imgaug for various augmentations
        augmented_image = image  # Placeholder, replace with actual augmentation
        return augmented_image

    def main(self, query_image, top_n, augmentation_factor):
        query_image_path = os.path.join(self.query_dir, query_image)

        # Load index images in parallel with progress bar
        index_images = self.load_index_images()

        # Augment and expand dataset
        augmented_index_images = self.augment_and_expand_dataset(index_images, augmentation_factor)

        # Encode index images and store in a dictionary
        index_encodings = {img_file: img_array for img_file, img_array in augmented_index_images}

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
        query_img_array = self.process_image(query_image_path)
        query_encoding = self.encode_image(query_img_array)

        # Reshape query_encoding to a 2D array
        query_encoding_2d = query_encoding.reshape(1, -1)

        # Find k-NN in the embedding space
        distances, indices = knn_model.kneighbors(query_encoding_2d, return_distance=True)

        print(f"Query Image: {query_image_path}")
        print("Similar Images:")
        neighbor_file_names = [file_names[idx] for idx in indices[0]]
        for i, neighbor_name in enumerate(neighbor_file_names):
            print(f"{i + 1}. {neighbor_name} - Cosine Distance: {distances[0][i]}")

        # Visualize the input image and its top neighbors
        self.visualize_neighbors(query_image_path, neighbor_file_names[:top_n], self.index_dir)


class PerformanceTester:
    def __init__(self, retrieval_system):
        self.retrieval_system = retrieval_system

    def calculate_metrics(self, query_image, top_n):
        query_image_path = os.path.join(self.retrieval_system.query_dir, query_image)

        index_images = self.retrieval_system.load_index_images()

        index_encodings = {img_file: img_array for img_file, img_array in index_images}
        embeddings = list(index_encodings.values())

        query_img_array = self.retrieval_system.process_image(query_image_path)
        query_encoding = self.retrieval_system.encode_image(query_img_array)

        similarities = []

        for encoding in embeddings:
            similarity = np.dot(query_encoding, encoding.T) / (
                        np.linalg.norm(query_encoding) * np.linalg.norm(encoding))
            similarities.append(similarity)

        # Calculate metrics
        top_similarities = sorted(similarities, reverse=True)[:top_n]
        mean_similarity = sum(top_similarities) / top_n

        true_positives = sum(similarity >= mean_similarity for similarity in similarities)
        false_positives = top_n - true_positives
        false_negatives = len(similarities) - true_positives

        accuracy = true_positives / len(similarities)
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)

        return mean_similarity, accuracy, precision, recall


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve similar images using k-NN.")
    parser.add_argument("query_image", type=str, help="Path to the query image.")
    parser.add_argument("--top_n", type=int, default=5, help="Number of top candidates to retrieve.")
    parser.add_argument("--augmentation_factor", type=int, default=5, help="Factor for data augmentation.")
    args = parser.parse_args()

    index_dir = 'data_test/index_small'
    query_dir = 'data_test/query'

    # Orig
    retrieval_system_orig = ImageRetrievalSystem(index_dir, query_dir)
    retrieval_system_orig.main(args.query_image, args.top_n)

    # Augmented
    retrieval_system_aug = AugmentedImageRetrievalSystem(index_dir, query_dir)
    retrieval_system_aug.main(args.query_image, args.top_n, args.augmentation_factor)
