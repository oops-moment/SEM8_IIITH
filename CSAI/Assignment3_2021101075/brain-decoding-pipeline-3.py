# Continue from previous code...

        # Create a dataset of images and their embeddings
        image_dataset = []
        for nsd_id in nsd_ids_train:
            image_path = os.path.join(IMAGE_DIR, f"nsd_{nsd_id:05d}.png")
            image = np.array(Image.open(image_path).resize((224, 224)))
            image_dataset.append(image)
        
        image_dataset = np.array(image_dataset)
        
        # For simplicity, we'll implement a basic nearest-neighbor approach for reconstruction
        # For each test prediction, find the closest training embedding and use its image
        for test_idx in range(5):  # Visualize first 5 test samples
            # Get predicted embedding
            pred_emb = y_pred_avg[test_idx]
            true_emb = y_test[test_idx]
            
            # Normalize embeddings
            pred_emb_norm = pred_emb / np.linalg.norm(pred_emb)
            y_train_norm = y_train / np.linalg.norm(y_train, axis=1, keepdims=True)
            
            # Calculate similarities with training embeddings
            similarities = np.dot(pred_emb_norm, y_train_norm.T)
            
            # Find the most similar training sample
            most_similar_idx = np.argmax(similarities)
            
            # Get the corresponding image
            reconstructed_img = image_dataset[most_similar_idx]
            original_nsd_id = nsd_ids_test[test_idx]
            original_img_path = os.path.join(IMAGE_DIR, f"nsd_{original_nsd_id:05d}.png")
            original_img = np.array(Image.open(original_img_path).resize((224, 224)))
            
            # Display original and reconstructed images
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(original_img)
            plt.title(f"Original (NSD ID: {original_nsd_id})")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(reconstructed_img)
            plt.title(f"Reconstructed (NSD ID: {nsd_ids_train[most_similar_idx]})")
            plt.axis('off')
            
            plt.suptitle(f"Similarity Score: {similarities[most_similar_idx]:.4f}")
            plt.show()
            
            # Get captions
            original_captions = get_coco_captions(original_nsd_id, stim_info)
            reconstructed_captions = get_coco_captions(nsd_ids_train[most_similar_idx], stim_info)
            
            print("Original image captions:")
            for i, caption in enumerate(original_captions[:3]):
                print(f"  {i+1}. {caption}")
            
            print("\nReconstructed image captions:")
            for i, caption in enumerate(reconstructed_captions[:3]):
                print(f"  {i+1}. {caption}")
            
            print("\n" + "-"*80 + "\n")
            
    except Exception as e:
        print(f"Error during image reconstruction: {e}")

# Function to study the effect of ROI combination
def study_roi_combinations(brain_data, roi_data, X_train, y_train, X_test, y_test):
    """
    Study the effect of combining different ROIs on decoding performance
    
    Parameters:
    - brain_data: dict with brain response data
    - roi_data: dict with ROI masks
    - X_train: dict with training data for each hemisphere
    - y_train: numpy array with training labels
    - X_test: dict with test data for each hemisphere
    - y_test: numpy array with test labels
    """
    # Define ROI combinations to test
    roi_combinations = {
        'Visual (V1-V4)': ['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4'],
        'Face regions': ['OFA', 'FFA-1', 'FFA-2', 'mTL-faces', 'aTL-faces'],
        'Place regions': ['OPA', 'PPA', 'RSC'],
        'Body regions': ['EBA', 'FBA-1', 'FBA-2', 'mTL-bodies'],
        'Word regions': ['OWFA', 'VWFA-1', 'VWFA-2', 'mfs-words', 'mTL-words'],
        'Ventral stream': ['V1v', 'V2v', 'V3v', 'hV4', 'VWFA-1', 'FFA-1', 'PPA'],
        'Dorsal stream': ['V1d', 'V2d', 'V3d', 'OPA', 'RSC'],
        'All ROIs': ALL_ROIS
    }
    
    combination_performance = {}
    
    for combo_name, rois in roi_combinations.items():
        print(f"\nEvaluating ROI combination: {combo_name}")
        
        # Create masks for the combination of ROIs
        combo_masks = {'left': None, 'right': None}
        
        for hemi in ['left', 'right']:
            # Combine masks for all ROIs in the combination
            for roi in rois:
                if roi in roi_data[hemi]:
                    if combo_masks[hemi] is None:
                        combo_masks[hemi] = roi_data[hemi][roi].copy()
                    else:
                        combo_masks[hemi] = np.logical_or(combo_masks[hemi], roi_data[hemi][roi])
            
            # Check if any ROIs were found
            if combo_masks[hemi] is None or np.sum(combo_masks[hemi]) == 0:
                print(f"  No ROIs found for {combo_name} in {hemi} hemisphere")
                combo_masks[hemi] = None
        
        # Skip if no ROIs were found in either hemisphere
        if combo_masks['left'] is None and combo_masks['right'] is None:
            print(f"  No ROIs found for {combo_name} in either hemisphere")
            continue
        
        # Extract voxel responses for the ROI combination
        combo_responses = {'left': {}, 'right': {}}
        
        for hemi in ['left', 'right']:
            if combo_masks[hemi] is not None:
                combo_responses[hemi]['train'] = X_train[hemi][:, combo_masks[hemi]]
                combo_responses[hemi]['test'] = X_test[hemi][:, combo_masks[hemi]]
                print(f"  {hemi.capitalize()} hemisphere: {np.sum(combo_masks[hemi])} voxels")
        
        # Train models for each hemisphere
        models = {}
        scalers = {}
        
        for hemi in ['left', 'right']:
            if combo_masks[hemi] is not None:
                models[hemi], scalers[hemi] = train_regression_model(
                    combo_responses[hemi]['train'], y_train)
        
        # Predict embeddings for each hemisphere
        y_pred = {}
        
        for hemi in ['left', 'right']:
            if combo_masks[hemi] is not None:
                y_pred[hemi] = predict_embeddings(
                    models[hemi], combo_responses[hemi]['test'], scalers[hemi])
        
        # Average predictions from both hemispheres if available
        if 'left' in y_pred and 'right' in y_pred:
            y_pred_avg = (y_pred['left'] + y_pred['right']) / 2
        elif 'left' in y_pred:
            y_pred_avg = y_pred['left']
        elif 'right' in y_pred:
            y_pred_avg = y_pred['right']
        else:
            print(f"  No predictions available for {combo_name}")
            continue
        
        # Calculate similarity
        similarity = calculate_similarity(y_pred_avg, y_test)
        
        # Store results
        combination_performance[combo_name] = {
            'mean_similarity': np.mean(similarity),
            'max_similarity': np.max(similarity),
            'min_similarity': np.min(similarity),
            'num_rois': len(rois),
            'num_voxels': sum(np.sum(combo_masks[hemi]) if combo_masks[hemi] is not None else 0 
                            for hemi in ['left', 'right'])
        }
        
        print(f"  Mean similarity: {combination_performance[combo_name]['mean_similarity']:.4f}")
    
    # Visualize combination performance
    sorted_combos = sorted(combination_performance.items(), 
                         key=lambda x: x[1]['mean_similarity'], reverse=True)
    
    plt.figure(figsize=(12, 8))
    
    # Extract combo names and mean similarities
    combo_names = [combo for combo, _ in sorted_combos]
    mean_similarities = [perf['mean_similarity'] for _, perf in sorted_combos]
    
    # Create bar plot
    bars = plt.bar(combo_names, mean_similarities, color='teal')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', rotation=0, fontsize=9)
    
    # Customize plot
    plt.title('ROI Combination Performance Comparison')
    plt.xlabel('ROI Combination')
    plt.ylabel('Mean Cosine Similarity')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Print top 3 combinations
    print("\nTop 3 Most Effective ROI Combinations:")
    for i, (combo, perf) in enumerate(sorted_combos[:3]):
        print(f"  {i+1}. {combo}: Mean Similarity = {perf['mean_similarity']:.4f}")
        print(f"     Number of ROIs: {perf['num_rois']}")
        print(f"     Total voxels: {perf['num_voxels']}")

# Execute the main pipeline
if __name__ == "__main__":
    print("Brain Decoding Models for Visual Brain - Assignment 3")
    print("Student Name: [Your Name]")
    print("Student Roll Number: [Your Roll Number]")
    print("Date: March 16, 2025")
    
    # Set subject ID (1-8)
    subject_id = 1
    
    # Run the main pipeline
    brain_decoding_pipeline(subject_id=subject_id, test_size=0.2)
    
    # Optional: Load data once more to study ROI combinations
    brain_data, roi_data = load_brain_data(subject_id)
    
    # Get all available trial IDs
    trials_df = pd.read_csv(os.path.join(BRAIN_DATA_PATH, f"subj{subject_id:02d}/trials.tsv"), 
                           sep='\t')
    all_trial_ids = trials_df['trial_id'].values
    
    # Split trials into training and testing sets
    train_ids, test_ids = train_test_split(all_trial_ids, test_size=0.2, random_state=42)
    
    # Extract data for training and testing
    stim_info = load_stimulus_info()
    X_train, y_train, nsd_ids_train = get_trial_data(
        subject_id, train_ids, brain_data, roi_data, stim_info)
    X_test, y_test, nsd_ids_test = get_trial_data(
        subject_id, test_ids, brain_data, roi_data, stim_info)
    
    # Study ROI combinations
    study_roi_combinations(brain_data, roi_data, X_train, y_train, X_test, y_test)
    
    print("\nBrain decoding pipeline completed successfully!")
