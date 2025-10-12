# 👨‍💻 Deep Face Verification System using Siamese Networks (SNN)

This repository contains the complete implementation of a **Face Verification** system developed as a capstone project. The system is designed to authenticate registered individuals and reject new, unauthorized users based on their facial features.

The core technology is a **Siamese Neural Network (SNN)** trained using **Metric Learning** techniques to produce highly discriminative facial embeddings. The project demonstrates a strong command of modern MLOps practices, including:

  * **Advanced Deep Learning Architecture:** Siamese Network design in **PyTorch**.
  * **Metric Learning:** Optimization using **Contrastive Loss**.
  * **Production Deployment:** Exposing the model via a scalable **FastAPI** RESTful service.

-----

## 💡 Core Methodology: Siamese Networks and Metric Learning

The system operates on the principle of minimizing the distance between the representations of similar faces and maximizing the distance for dissimilar faces in the embedding space.

### 1\. Siamese Network Architecture

  * **Structure:** The SNN consists of two identical **Convolutional Neural Network (CNN) branches** with **shared weights**. This ensures both input images (anchor and positive/negative) are mapped to the same latent space.
  * **Embeddings:** The output of the network is a low-dimensional vector (the **face embedding**) that uniquely represents the identity of the person.

### 2\. Contrastive Loss Function

  * **Training Objective:** The model is trained using **Contrastive Loss**, which enforces the following separation:
      * **Same Person (Positive Pair):** The distance between embeddings is pushed towards zero.
      * **Different People (Negative Pair):** The distance is penalized if it falls below a specific margin, ensuring separation.

### 3\. Real-Time Verification Logic

During inference, the system determines the match using **Euclidean Distance** (pairwise distance) compared against a system-defined **Threshold**:

  * **Matching Process:** The input image's embedding is compared against the embeddings of all registered images for each person. The system selects the registered person with the **minimum average dissimilarity**.
  * **Decision:** If this minimum dissimilarity is below the pre-calibrated **Threshold**, the identity is verified. Otherwise, the person is rejected as a "New/Unrecognized User."

-----

## 📊 Performance and Key Results

| Metric / Component | Description | Value / Details |
| :--- | :--- | :--- |
| **Model Architecture** | Custom Siamese Network with CNN backbone | Implemented entirely in **PyTorch** |
| **Training Dataset** | **VGGFace2** (or a relevant subset) | Used for robust feature learning. |
| **Best Validation Accuracy** | Highest pairwise accuracy achieved during training. | $\mathbf{83\%}$ (achieved on samples of 2000 train / 500 test) |
| **Loss Function** | **Contrastive Loss** | Key for effective metric learning. |
| **Deployment Framework** | **FastAPI + Uvicorn** | Used to create a fast, asynchronous API for deployment. |

-----

## 📦 Repository Structure

The project is structured into three main components: source code, deployment, and documentation.

| Directory | Content and Purpose |
| :--- | :--- |
| `src/` | **Core Training Code:** Contains `ML_Vision_Project.ipynb` (detailed training notebook) and `ml_vision_project.py` (clean script). |
| `deployment/` | **API Deployment:** Contains `face_recognition_api.py` for FastAPI implementation and the `register/` folder for reference images. |
| `docs/` | **Project Documentation:** Includes the final comprehensive technical report (`report.pdf`, `report.docx`). |
| `output/` | **Demonstration Output:** Contains `project_demo_videos.zip` showing the project in operation (API testing and verification scenarios). |
| `model/` | **Trained Weights:** Placeholder for the final model weights (`Siamese_model.pth`). |
| `requirements.txt` | **Dependencies:** Lists all required Python packages for full reproducibility. |

-----

## ⚙️ Setup and Deployment

### 1\. Installation

All dependencies are listed in `requirements.txt`. Install them using `pip`:

```bash
pip install -r requirements.txt
```

### 2\. API Deployment

To run the verification service locally:

1.  Ensure your trained model (`Siamese_model.pth`) is placed in the `model/` directory.
2.  Populate the `deployment/register/` folder with reference images for verification.
3.  Start the FastAPI server using Uvicorn:

<!-- end list -->

```bash
uvicorn deployment.face_recognition_api:app --host 0.0.0.0 --port 8000
```

The system will now expose a POST endpoint for real-time face verification.

### 3\. API Endpoint

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/identify/` | Uploads a face image, calculates its embedding, compares it against registered users, and returns the best match and its dissimilarity score. |

## 🎥 Project Demonstration

The `output/project_demo_videos.zip` file provides an essential **Proof of Concept** by showcasing the live API functionality:

  * **Scenario 1:** Successful Verification (Registered User, Dissimilarity below Threshold).
  * **Scenario 2:** Rejection (New User, Dissimilarity above Threshold).
  * **Scenario 3:** Demonstration of the **FastAPI** server interface (e.g., Swagger UI) for submitting requests.