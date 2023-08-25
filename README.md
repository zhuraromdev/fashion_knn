# Image Retrieval System

This code implements an image retrieval system using a k-Nearest Neighbors (k-NN) approach based on image embeddings. The system utilizes the ResNet50 model for image encoding and calculates cosine similarity in the embedding space to retrieve similar images.

## Prerequisites

- Python 3.x
- Required packages can be installed using:
  ```shell
  pip install -r requirements.txt


## Usage
- Place index images in the data_test/index_small directory and query images in the data_test/query directory.
- Run the following command to retrieve similar images for a given query image:
  ```shell
    python main.py <query_image_path> [--top_n <num>] [--augmentation_factor <factor>]
  <query_image_path>: Path to the query image.

  - -top_n <num> (optional, default=5): Number of top similar candidates to retrieve.
  - -augmentation_factor <factor> (optional, default=5): Augmentation factor for data augmentation (only for AugmentedImageRetrievalSystem).

- The script will display the input query image and its top similar neighbors.

## Classes
### ImageRetrievalSystem
This class manages the image retrieval process using a non-augmented dataset.

- encode_image(img_array): Encodes an image array using the ResNet50 model.
- load_index_images(): Loads index images, processes them, and encodes them.
- get_index_encodings(): Returns index image encodings, loading or generating them as needed.
- visualize_neighbors(input_image_path, neighbor_file_names, index_dir): Displays the input image and its neighbors.
- main(query_image, top_n): Main retrieval process.

### AugmentedImageRetrievalSystem

This class extends ImageRetrievalSystem to support data augmentation for improving retrieval.
- augment_and_expand_dataset(index_images, augmentation_factor): Augments index images and expands the dataset.
- augment_image(image): Applies data augmentation transformations (placeholder).
- main(query_image, top_n, augmentation_factor): Main retrieval process with augmentation.

### PerformanceTester
This class calculates retrieval system performance metrics.

- calculate_metrics(query_image, top_n): Calculates mean similarity, accuracy, precision, and recall.

## Example
To retrieve similar images for a query image named example.jpg and display the top 3 similar images:
  ```shell
      python main.py data_test/query/example.jpg --top_n 3
  ```

# Additional Enhancements

## Ensemble Encoding with Multiple Models

This enhancement introduces the concept of ensemble encoding using multiple instances of the ResNet50 model. By utilizing an ensemble of models, the image retrieval system can generate more robust and diverse image embeddings, potentially leading to improved retrieval performance.

### Creating and Loading Multiple Base Models

Multiple instances of the ResNet50 model are loaded to form an ensemble. The code snippet below demonstrates how to load multiple base models:

  ```python
  num_models = 3  # You can adjust this number
  base_models = [tf.keras.applications.ResNet50(weights='imagenet', include_top=False,
                                                pooling='avg', input_shape=(224, 224, 3))
                 for _ in range(num_models)]
  ```


## Ensemble Encoding
The encode_image_ensemble function computes the ensemble encoding for an input image using the loaded base models. This function calculates the individual encodings for each model and then computes the mean of these encodings to form the ensemble encoding.

  ```python
  def encode_image_ensemble(features, models):
      encoded_features = [model.predict(features) for model in models]
      mean_encoded_features = np.mean(encoded_features, axis=0)
      return mean_encoded_features
  ```

## Running the Enhanced System
To utilize the ensemble encoding with multiple models, you can follow these steps:
- Load multiple base models as shown above.
- Update the encode_image and encode_image_ensemble functions to use the appropriate model.
- Update the main function to include the ensemble encoding process.

```python
python multiple_models.py <query_image_path> [--top_n <num>]
```

## Benefits of Ensemble Encoding
Ensemble encoding offers the following benefits:
- Enhanced Robustness: Ensemble encoding leverages multiple models to capture different aspects of the image, resulting in more diverse and robust image representations.
- Improved Retrieval: The diversified embeddings obtained from ensemble encoding can lead to improved similarity measurements and better retrieval results.

# Results

Original model
![Original model](result/Figure_1.png)

Augmentation model
![Augmentation model](result/Figure_2.png)


# Future Improvements

The current version of the image retrieval system provides a solid foundation for similarity-based image retrieval. However, there are several avenues for further enhancement and development. Here are some ideas for future improvements:

### 1. Fine-Tuning and Transfer Learning

Consider fine-tuning the base ResNet50 model using a dataset that is more specific to your domain. Fine-tuning involves training the model on your dataset while retaining the pre-trained weights. This process can help the model learn features that are more relevant to your images, potentially leading to improved embeddings and retrieval performance.

### 2. Advanced Data Augmentation

Explore advanced data augmentation techniques to further improve the robustness of the system. Libraries like `imgaug` offer a wide range of augmentation options, allowing you to introduce variations in lighting, rotation, scale, and more. Enhanced data augmentation can help the model better capture different image variations and viewpoints.

### 3. Neural Architecture Search (NAS)

Consider utilizing Neural Architecture Search (NAS) techniques to find optimal neural network architectures for image embedding. NAS automates the process of architecture design and can help discover architectures that are better suited for your specific retrieval task.

### 4. Learning to Rank

Implement "Learning to Rank" techniques to refine the ordering of retrieved images. Learning to Rank algorithms use supervised learning to optimize the ranking of retrieved images based on user preferences. This can result in a more intuitive and relevant image ordering for users.

### 5. User Feedback Integration

Incorporate user feedback mechanisms to improve the system's performance over time. Allow users to provide feedback on retrieved results, and use this feedback to fine-tune the model or adjust retrieval parameters dynamically.

### 6. Scalability and Deployment

Consider deploying the image retrieval system in a production environment. This may involve optimizing the code, parallelizing computations, and utilizing cloud resources for scalability. Deploying the system as a web service or integrating it with an existing platform can make it accessible to a wider audience.

### 7. Deep Metric Learning

Explore deep metric learning techniques to learn a more suitable similarity metric for your retrieval task. Deep metric learning aims to learn an embedding space where the distances between embeddings reflect the desired similarity measure.

### 8. Multimodal Retrieval

Extend the system to support multimodal retrieval, where users can query using both text and images. This can involve incorporating natural language processing models and fusion techniques to combine information from different modalities.

### 9. Explainability

Integrate explainability techniques to provide insights into why certain images are retrieved as similar to the query. Explainable AI methods can help users understand the model's decision-making process.

### 10. Benchmarking and Evaluation

Create a comprehensive evaluation framework to benchmark the system's performance against various datasets and retrieval scenarios. This can provide insights into the system's strengths and areas for improvement.

These suggestions offer a starting point for enhancing and extending the capabilities of the image retrieval system. Depending on your goals and resources, you can choose to explore one or more of these directions to create a more powerful and versatile retrieval system.
 