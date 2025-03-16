# Brain Decoding Models for Visual Brain - Assignment 3
# Student Name: [Your Name]
# Student Roll Number: [Your Roll Number]
# Date: March 16, 2025

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import requests
from io import BytesIO
from tqdm.notebook import tqdm
import pickle
import h5py
import nibabel as nib
from pycocotools.coco import COCO
from transformers import CLIPProcessor, CLIPModel

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Load the CLIP model for multimodal embeddings
print("Loading CLIP model for multimodal embeddings...")
model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name)
clip_processor = CLIPProcessor.from_pretrained(model_name)

# Define paths (update these based on your setup)
DATA_DIR = "path/to/nsd_data/"
BRAIN_DATA_PATH = os.path.join(DATA_DIR, "nsddata_betas/ppdata/")
STIM_INFO_PATH = os.path.join(DATA_DIR, "nsd_stim_info_merged.csv")
COCO_ANNO_DIR = os.path.join(DATA_DIR, "annotations_trainval2017/annotations/")

IMAGE_DIR = os.path.join(DATA_DIR, "stimuli/nsd/")

# ROI mapping dictionaries for better visualization and analysis
ROI_GROUPS = {
    'prf-visualrois': ['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4'],
    'floc-bodies': ['EBA', 'FBA-1', 'FBA-2', 'mTL-bodies'],
    'floc-faces': ['OFA', 'FFA-1', 'FFA-2', 'mTL-faces', 'aTL-faces'],
    'floc-places': ['OPA', 'PPA', 'RSC'],
    'floc-words': ['OWFA', 'VWFA-1', 'VWFA-2', 'mfs-words', 'mTL-words']
}

# Flatten ROI dictionary for easy access
ALL_ROIS = []
for rois in ROI_GROUPS.values():
    ALL_ROIS.extend(rois)

# Function to load stimulus information
def load_stimulus_info():
    """
    Load and preprocess stimulus information from CSV
    """
    print("Loading stimulus information...")
    stim_info = pd.read_csv(STIM_INFO_PATH, index_col=0)
    print(f"Loaded information for {len(stim_info)} stimuli")
    return stim_info

# Function to get image captions from COCO
def get_coco_captions(nsdId, stim_info):
    """
    Get COCO captions for a given NSD image ID
    """
    try:
        # Get COCO ID and split
        coco_id = stim_info[stim_info['nsdId'] == nsdId]['cocoId'].values[0]
        coco_split = stim_info[stim_info['nsdId'] == nsdId]['cocoSplit'].values[0]
        
        # Load COCO annotation data
        coco_annotation_file = os.path.join(COCO_ANNO_DIR, f"captions_{coco_split}.json")
        coco_data = COCO(coco_annotation_file)
        coco_ann_ids = coco_data.getAnnIds(coco_id)
        coco_annotations = coco_data.loadAnns(coco_ann_ids)
        
        # Extract captions
        captions = [anno['caption'] for anno in coco_annotations]
        return captions
    except Exception as e:
        print(f"Error getting captions for nsdId {nsdId}: {e}")
        return []

# Function to load image and get CLIP embeddings
def get_clip_embedding(image_path):
    """
    Load image and extract CLIP embeddings
    """
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Process image with CLIP processor
        inputs = clip_processor(images=image, return_tensors="pt")
        
        # Get embeddings
        with torch.no_grad():
            outputs = clip_model.get_image_features(**inputs)
        
        # Return numpy array
        return outputs.cpu().numpy()[0]
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Function to load brain data for a subject
def load_brain_data(subject_id, hemisphere=None):
    """
    Load brain data for a specific subject and optionally a specific hemisphere
    
    Parameters:
    - subject_id: int (1-8)
    - hemisphere: str ('left', 'right', or None for both)
    
    Returns:
    - brain_data: dict with keys for hemispheres, each containing voxel data
    - roi_data: dict with ROI masks
    """
    print(f"Loading brain data for subject {subject_id}...")
    
    # Format subject ID with leading zeros
    subject_str = f"subj{subject_id:02d}"
    
    # Define paths for hemisphere data
    brain_data = {}
    roi_data = {}
    hemispheres = ['left', 'right'] if hemisphere is None else [hemisphere]
    
    for hemi in hemispheres:
        # Load voxel data
        voxel_file = os.path.join(BRAIN_DATA_PATH, subject_str, f"{hemi}.hdf5")
        with h5py.File(voxel_file, 'r') as f:
            voxel_data = f['betas'][:]  # Shape: (num_trials, num_voxels)
        brain_data[hemi] = voxel_data
        
        # Load ROI masks
        roi_data[hemi] = {}
        for roi_group, roi_names in ROI_GROUPS.items():
            roi_file = os.path.join(BRAIN_DATA_PATH, subject_str, f"{roi_group}_{hemi}.npy")
            try:
                roi_masks = np.load(roi_file, allow_pickle=True).item()
                for roi_name in roi_names:
                    if roi_name in roi_masks:
                        roi_data[hemi][roi_name] = roi_masks[roi_name]
            except FileNotFoundError:
                print(f"Could not find ROI file: {roi_file}")
    
    print(f"Loaded brain data with shapes: {', '.join([f'{k}: {v.shape}' for k, v in brain_data.items()])}")
    return brain_data, roi_data

# Function to extract data for a specific trial
def get_trial_data(subject_id, trial_ids, brain_data, roi_data, stim_info):
    """
    Extract brain data and stimulus information for specific trials
    
    Parameters:
    - subject_id: int
    - trial_ids: list of trial IDs
    - brain_data: dict with brain response data
    - roi_data: dict with ROI masks
    - stim_info: dataframe with stimulus information
    
    Returns:
    - X: dict with brain responses for each hemisphere
    - y: numpy array with CLIP embeddings for each trial
    - nsd_ids: list of NSD IDs for each trial
    """
    # Get NSD IDs for trials
    trials_df = pd.read_csv(os.path.join(BRAIN_DATA_PATH, f"subj{subject_id:02d}/trials.tsv"), 
                           sep='\t')
    nsd_ids = [trials_df.loc[trials_df['trial_id'] == tid, 'nsd_id'].values[0] for tid in trial_ids]
    
    # Extract brain responses for each hemisphere
    X = {}
    for hemi, voxel_data in brain_data.items():
        X[hemi] = voxel_data[trial_ids, :]
    
    # Extract CLIP embeddings for stimuli
    y = []
    for nsd_id in nsd_ids:
        image_path = os.path.join(IMAGE_DIR, f"nsd_{nsd_id:05d}.png")
        embedding = get_clip_embedding(image_path)
        y.append(embedding)
    
    y = np.array(y)
    
    return X, y, nsd_ids

# Function to extract ROI-specific brain responses
def extract_roi_responses(brain_responses, roi_masks):
    """
    Extract responses for specific ROIs
    
    Parameters:
    - brain_responses: numpy array of shape (n_trials, n_voxels)
    - roi_masks: dict with ROI masks
    
    Returns:
    - roi_responses: dict with ROI-specific responses
    """
    roi_responses = {}
    for roi_name, mask in roi_masks.items():
        if mask.sum() > 0:  # Check if the ROI has any voxels
            roi_responses[roi_name] = brain_responses[:, mask]
    
    return roi_responses

# Function to train regression model
def train_regression_model(X_train, y_train, alpha=1.0):
    """
    Train Ridge regression model
    
    Parameters:
    - X_train: numpy array of shape (n_samples, n_features)
    - y_train: numpy array of shape (n_samples, n_target_dims)
    - alpha: regularization strength
    
    Returns:
    - model: trained Ridge regression model
    """
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train Ridge regression model
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

# Function to predict embeddings using trained model
def predict_embeddings(model, X_test, scaler):
    """
    Predict embeddings using trained model
    
    Parameters:
    - model: trained Ridge regression model
    - X_test: numpy array of shape (n_samples, n_features)
    - scaler: fitted StandardScaler
    
    Returns:
    - y_pred: numpy array of shape (n_samples, n_target_dims)
    """
    # Standardize features
    X_test_scaled = scaler.transform(X_test)
    
    # Predict embeddings
    y_pred = model.predict(X_test_scaled)
    
    return y_pred

# Function to calculate similarity between predicted and ground truth embeddings
def calculate_similarity(y_pred, y_true):
    """
    Calculate similarity between predicted and ground truth embeddings
    
    Parameters:
    - y_pred: numpy array of shape (n_samples, n_dims)
    - y_true: numpy array of shape (n_samples, n_dims)
    
    Returns:
    - similarity: numpy array of shape (n_samples,)
    """
    # Normalize embeddings
    y_pred_norm = y_pred / np.linalg.norm(y_pred, axis=1, keepdims=True)
    y_true_norm = y_true / np.linalg.norm(y_true, axis=1, keepdims=True)
    
    # Calculate cosine similarity
    similarity = np.sum(y_pred_norm * y_true_norm, axis=1)
    
    return similarity

# Function to find top K similar images from the training set
def find_top_k_similar(y_pred, y_train, nsd_ids_train, k=5):
    """
    Find top K similar images from the training set
    
    Parameters:
    - y_pred: numpy array of shape (n_test, n_dims)
    - y_train: numpy array of shape (n_train, n_dims)
    - nsd_ids_train: list of NSD IDs for training samples
    - k: number of top matches to return
    
    Returns:
    - top_matches: list of dicts with similarity scores and NSD IDs
    """
    # Normalize embeddings
    y_pred_norm = y_pred / np.linalg.norm(y_pred, axis=1, keepdims=True)
    y_train_norm = y_train / np.linalg.norm(y_train, axis=1, keepdims=True)
    
    # Calculate cosine similarity matrix
    sim_matrix = np.dot(y_pred_norm, y_train_norm.T)  # Shape: (n_test, n_train)
    
    # Find top K matches for each test sample
    top_matches = []
    for i in range(sim_matrix.shape[0]):
        # Get indices of top K matches
        top_indices = np.argsort(sim_matrix[i])[::-1][:k]
        
        # Get similarity scores and NSD IDs
        top_scores = sim_matrix[i][top_indices]
        top_nsd_ids = [nsd_ids_train[idx] for idx in top_indices]
        
        # Store results
        top_matches.append({
            'scores': top_scores,
            'nsd_ids': top_nsd_ids
        })
    
    return top_matches

# Function to visualize regression coefficients
def visualize_regression_coefficients(model, roi_masks, roi_name):
    """
    Visualize regression coefficients for a specific ROI
    
    Parameters:
    - model: trained Ridge regression model
    - roi_masks: dict with ROI masks
    - roi_name: name of the ROI to visualize
    """
    # Get coefficients
    coeffs = model.coef_  # Shape: (n_target_dims, n_features)
    
    # Get mask for the ROI
    mask = roi_masks[roi_name]
    
    # Calculate mean absolute coefficients across target dimensions
    mean_abs_coeffs = np.mean(np.abs(coeffs[:, mask]), axis=0)
    
    # Plot histogram of coefficients
    plt.figure(figsize=(10, 6))
    plt.hist(mean_abs_coeffs, bins=50, alpha=0.7)
    plt.title(f'Mean Absolute Coefficients for {roi_name}')
    plt.xlabel('Mean Absolute Coefficient')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Plot top and bottom coefficients
    n_top = 20
    top_indices = np.argsort(mean_abs_coeffs)[::-1][:n_top]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(n_top), mean_abs_coeffs[top_indices], color='teal')
    plt.title(f'Top {n_top} Voxels for {roi_name}')
    plt.xlabel('Voxel Index (within ROI)')
    plt.ylabel('Mean Absolute Coefficient')
    plt.grid(True, alpha=0.3)
    plt.show()

# Function to visualize image reconstruction
def visualize_images(nsd_ids, stim_info, k=5):
    """
    Visualize images for given NSD IDs