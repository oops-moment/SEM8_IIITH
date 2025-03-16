# Continue from previous code...

def visualize_images(nsd_ids, stim_info, k=5):
    """
    Visualize images for given NSD IDs
    
    Parameters:
    - nsd_ids: list of NSD IDs
    - stim_info: dataframe with stimulus information
    - k: number of images to visualize
    """
    # Limit to k images
    nsd_ids = nsd_ids[:k]
    
    # Create figure
    fig, axes = plt.subplots(1, k, figsize=(20, 4))
    
    # Plot each image
    for i, nsd_id in enumerate(nsd_ids):
        # Load image
        image_path = os.path.join(IMAGE_DIR, f"nsd_{nsd_id:05d}.png")
        image = Image.open(image_path)
        
        # Display image
        axes[i].imshow(image)
        axes[i].set_title(f"NSD ID: {nsd_id}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Function to visualize similarity matrix
def visualize_similarity_matrix(y_pred, y_true, nsd_ids_test):
    """
    Visualize similarity matrix between predicted and ground truth embeddings
    
    Parameters:
    - y_pred: numpy array of shape (n_samples, n_dims)
    - y_true: numpy array of shape (n_samples, n_dims)
    - nsd_ids_test: list of NSD IDs for test samples
    """
    # Normalize embeddings
    y_pred_norm = y_pred / np.linalg.norm(y_pred, axis=1, keepdims=True)
    y_true_norm = y_true / np.linalg.norm(y_true, axis=1, keepdims=True)
    
    # Calculate similarity matrix
    sim_matrix = np.dot(y_pred_norm, y_true_norm.T)
    
    # Plot similarity matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, cmap='viridis', vmin=0, vmax=1)
    plt.title('Similarity Matrix: Predicted vs. Ground Truth')
    plt.xlabel('Ground Truth Index')
    plt.ylabel('Predicted Index')
    plt.show()
    
    # Calculate diagonal values (similarity between predicted and true embedding for same sample)
    diagonal_sim = np.diag(sim_matrix)
    
    # Plot diagonal values
    plt.figure(figsize=(12, 6))
    plt.plot(diagonal_sim, 'o-', color='teal')
    plt.title('Similarity Between Predicted and True Embeddings')
    plt.xlabel('Sample Index')
    plt.ylabel('Cosine Similarity')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Print mean similarity
    print(f"Mean cosine similarity: {np.mean(diagonal_sim):.4f}")
    
    # Print top 5 and bottom 5 similarities
    top_indices = np.argsort(diagonal_sim)[::-1][:5]
    bottom_indices = np.argsort(diagonal_sim)[:5]
    
    print("\nTop 5 similarities:")
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. NSD ID {nsd_ids_test[idx]}: {diagonal_sim[idx]:.4f}")
    
    print("\nBottom 5 similarities:")
    for i, idx in enumerate(bottom_indices):
        print(f"  {i+1}. NSD ID {nsd_ids_test[idx]}: {diagonal_sim[idx]:.4f}")

# Function to identify most predictive ROIs
def identify_predictive_rois(brain_data, roi_data, X_train, y_train, X_test, y_test, nsd_ids_test):
    """
    Identify most predictive ROIs by training and evaluating models on each ROI
    
    Parameters:
    - brain_data: dict with brain response data
    - roi_data: dict with ROI masks
    - X_train: dict with training data for each hemisphere
    - y_train: numpy array with training labels
    - X_test: dict with test data for each hemisphere
    - y_test: numpy array with test labels
    - nsd_ids_test: list of NSD IDs for test samples
    
    Returns:
    - roi_performance: dict with performance metrics for each ROI
    """
    roi_performance = {}
    
    # Iterate through all ROIs
    for roi_name in ALL_ROIS:
        print(f"\nEvaluating ROI: {roi_name}")
        
        # Extract ROI-specific responses for left and right hemispheres
        roi_responses_left = {}
        roi_responses_right = {}
        
        if roi_name in roi_data['left']:
            mask_left = roi_data['left'][roi_name]
            roi_responses_left['train'] = X_train['left'][:, mask_left]
            roi_responses_left['test'] = X_test['left'][:, mask_left]
        
        if roi_name in roi_data['right']:
            mask_right = roi_data['right'][roi_name]
            roi_responses_right['train'] = X_train['right'][:, mask_right]
            roi_responses_right['test'] = X_test['right'][:, mask_right]
        
        # Skip if ROI doesn't exist in either hemisphere
        if not roi_responses_left and not roi_responses_right:
            print(f"  ROI {roi_name} not found in either hemisphere")
            continue
        
        # Train models for each hemisphere
        models = {}
        scalers = {}
        
        if roi_responses_left:
            models['left'], scalers['left'] = train_regression_model(
                roi_responses_left['train'], y_train)
        
        if roi_responses_right:
            models['right'], scalers['right'] = train_regression_model(
                roi_responses_right['train'], y_train)
        
        # Predict embeddings for each hemisphere
        y_pred = {}
        
        if roi_responses_left:
            y_pred['left'] = predict_embeddings(
                models['left'], roi_responses_left['test'], scalers['left'])
        
        if roi_responses_right:
            y_pred['right'] = predict_embeddings(
                models['right'], roi_responses_right['test'], scalers['right'])
        
        # Average predictions from both hemispheres if available
        if roi_responses_left and roi_responses_right:
            y_pred_avg = (y_pred['left'] + y_pred['right']) / 2
        elif roi_responses_left:
            y_pred_avg = y_pred['left']
        else:
            y_pred_avg = y_pred['right']
        
        # Calculate similarity
        similarity = calculate_similarity(y_pred_avg, y_test)
        
        # Store results
        roi_performance[roi_name] = {
            'mean_similarity': np.mean(similarity),
            'max_similarity': np.max(similarity),
            'min_similarity': np.min(similarity)
        }
        
        print(f"  Mean similarity: {roi_performance[roi_name]['mean_similarity']:.4f}")
    
    return roi_performance

# Function to compare ROI performance
def compare_roi_performance(roi_performance):
    """
    Compare performance of different ROIs
    
    Parameters:
    - roi_performance: dict with performance metrics for each ROI
    """
    # Sort ROIs by mean similarity
    sorted_rois = sorted(roi_performance.items(), key=lambda x: x[1]['mean_similarity'], reverse=True)
    
    # Plot performance comparison
    plt.figure(figsize=(12, 8))
    
    # Extract ROI names and mean similarities
    roi_names = [roi for roi, _ in sorted_rois]
    mean_similarities = [perf['mean_similarity'] for _, perf in sorted_rois]
    
    # Group ROIs by category
    roi_categories = []
    for roi in roi_names:
        for category, rois in ROI_GROUPS.items():
            if roi in rois:
                roi_categories.append(category)
                break
    
    # Assign colors based on categories
    category_colors = {
        'prf-visualrois': 'teal',
        'floc-bodies': 'orangered',
        'floc-faces': 'mediumpurple',
        'floc-places': 'steelblue',
        'floc-words': 'darkgreen'
    }
    
    # Create bar plot
    bars = plt.bar(roi_names, mean_similarities, 
                   color=[category_colors.get(cat, 'gray') for cat in roi_categories])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', rotation=90, fontsize=8)
    
    # Customize plot
    plt.title('ROI Performance Comparison')
    plt.xlabel('ROI')
    plt.ylabel('Mean Cosine Similarity')
    plt.xticks(rotation=90)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    handles = [plt.Rectangle((0,0),1,1, color=color) for color in category_colors.values()]
    plt.legend(handles, category_colors.keys(), loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    # Print top 5 ROIs
    print("\nTop 5 Most Predictive ROIs:")
    for i, (roi, perf) in enumerate(sorted_rois[:5]):
        print(f"  {i+1}. {roi}: Mean Similarity = {perf['mean_similarity']:.4f}")

# Main function for the brain decoding pipeline
def brain_decoding_pipeline(subject_id=1, test_size=0.2):
    """
    Main function for the brain decoding pipeline
    
    Parameters:
    - subject_id: int (1-8)
    - test_size: float (0.0-1.0)
    """
    print("Starting brain decoding pipeline...")
    
    # Load stimulus information
    stim_info = load_stimulus_info()
    
    # Load brain data
    brain_data, roi_data = load_brain_data(subject_id)
    
    # Get all available trial IDs
    trials_df = pd.read_csv(os.path.join(BRAIN_DATA_PATH, f"subj{subject_id:02d}/trials.tsv"), 
                           sep='\t')
    all_trial_ids = trials_df['trial_id'].values
    
    # Split trials into training and testing sets
    train_ids, test_ids = train_test_split(all_trial_ids, test_size=test_size, random_state=42)
    
    print(f"Split {len(all_trial_ids)} trials into {len(train_ids)} training and {len(test_ids)} testing")
    
    # Extract data for training and testing
    X_train, y_train, nsd_ids_train = get_trial_data(
        subject_id, train_ids, brain_data, roi_data, stim_info)
    
    X_test, y_test, nsd_ids_test = get_trial_data(
        subject_id, test_ids, brain_data, roi_data, stim_info)
    
    # Train models for each hemisphere
    print("\nTraining models for each hemisphere...")
    models = {}
    scalers = {}
    
    for hemi in ['left', 'right']:
        print(f"Training model for {hemi} hemisphere...")
        models[hemi], scalers[hemi] = train_regression_model(X_train[hemi], y_train)
    
    # Predict embeddings for each hemisphere
    print("\nPredicting embeddings for each hemisphere...")
    y_pred = {}
    
    for hemi in ['left', 'right']:
        print(f"Predicting embeddings for {hemi} hemisphere...")
        y_pred[hemi] = predict_embeddings(models[hemi], X_test[hemi], scalers[hemi])
    
    # Average predictions from both hemispheres
    print("\nAveraging predictions from both hemispheres...")
    y_pred_avg = (y_pred['left'] + y_pred['right']) / 2
    
    # Calculate similarity
    print("\nCalculating similarity...")
    similarity = calculate_similarity(y_pred_avg, y_test)
    
    print(f"Mean cosine similarity: {np.mean(similarity):.4f}")
    print(f"Max cosine similarity: {np.max(similarity):.4f}")
    print(f"Min cosine similarity: {np.min(similarity):.4f}")
    
    # Find top 5 similar images
    print("\nFinding top 5 similar images...")
    top_matches = find_top_k_similar(y_pred_avg, y_train, nsd_ids_train, k=5)
    
    # Visualize similarity matrix
    print("\nVisualizing similarity matrix...")
    visualize_similarity_matrix(y_pred_avg, y_test, nsd_ids_test)
    
    # Visualize images for top 5 matches
    print("\nVisualizing top 5 matches for first test sample...")
    top_match = top_matches[0]
    visualize_images(top_match['nsd_ids'], stim_info)
    
    # Print captions for top 5 matches
    print("\nCaptions for top 5 matches:")
    for i, nsd_id in enumerate(top_match['nsd_ids']):
        captions = get_coco_captions(nsd_id, stim_info)
        print(f"  {i+1}. NSD ID {nsd_id} (Similarity: {top_match['scores'][i]:.4f}):")
        for j, caption in enumerate(captions):
            print(f"     {j+1}. {caption}")
    
    # Identify most predictive ROIs
    print("\nIdentifying most predictive ROIs...")
    roi_performance = identify_predictive_rois(
        brain_data, roi_data, X_train, y_train, X_test, y_test, nsd_ids_test)
    
    # Compare ROI performance
    print("\nComparing ROI performance...")
    compare_roi_performance(roi_performance)
    
    # Visualize regression coefficients for top ROI
    print("\nVisualizing regression coefficients for top ROI...")
    top_roi = max(roi_performance, key=lambda k: roi_performance[k]['mean_similarity'])
    
    for hemi in ['left', 'right']:
        if top_roi in roi_data[hemi]:
            print(f"Visualizing coefficients for {top_roi} in {hemi} hemisphere...")
            visualize_regression_coefficients(models[hemi], roi_data[hemi], top_roi)
    
    # Optional: Attempt image reconstruction (for bonus points)
    print("\nAttempting image reconstruction...")
    try:
        # This is a very basic approach - more sophisticated methods exist
        from sklearn.decomposition import PCA
        
        # Create a PCA model for dimensionality reduction of embeddings
        pca = PCA(n_components=512)
        pca.fit(y_train)
        
        # Create a dataset of images and their embeddings
        image_dataset = []
        for nsd_id in nsd_ids_train:
            image_path = os.path.join(IMAGE_DIR, f"nsd_{nsd_id:05d}.png")
            image = np