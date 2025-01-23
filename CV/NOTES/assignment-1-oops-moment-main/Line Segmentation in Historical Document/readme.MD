# Image Preprocessing and Text Line Detection

## 3.1.2 Image Preprocessing for Text Line Detection (15 Marks)

### Task 1: Implement a function `preprocess_image(image)` that applies a sequence of image processing techniques to enhance the image for text line detection.

**Evaluation:**
- The function converts the image to grayscale, applies binary thresholding, and then uses morphological dilation to connect text lines. After each major preprocessing step, the intermediate result is displayed.

**Justification of Techniques:**
- **Grayscale Conversion:** Simplifies the image by reducing color complexity, making it easier to process.
- **Binary Inverse Thresholding:** Used to distinguish text from the background.
- **Dilation:** Connects fragmented parts of text lines, making it easier to detect them as continuous regions.

---

## 3.2 Text Line Detection and Segmentation (50 Marks)

### 3.2.1 Rectangular Bounding Box Detection (25 Marks)

### Task 2: Implement a function `detect_lines_with_bboxes(image)` that detects text lines and encloses each with a rectangular bounding box.

**Evaluation:**
- The function applies the preprocessing steps from Task 1 and detects the bounding boxes around text lines using contours.
- The image with bounding boxes is displayed, and each detected text line is saved as a separate image file.

**Discussion on Performance:**
- The bounding box detection works well for most cases but may fail with overlapping or very small text lines.
- Artifacts or very short lines might be missed if the threshold values (width and height) are not tuned appropriately.

### Task 3: Save each detected text line as a separate image.

**Evaluation:**
- The function saves each cropped text line as a separate image in the specified output directory.

---

## Explanation:

1. **Detecting the Circle:**
   - We use the Hough Circle Transform (`cv2.HoughCircles`) to detect the circular seal in the image. This function returns the center and radius of the detected circles.
   - For simplicity, the first detected circle is selected for processing, but this can be extended to handle multiple circles if needed.

2. **Masking the Circular Region:**
   - We create a circular mask using the center and radius of the detected circle. This mask isolates the circular region of interest.

3. **Preprocessing:**
   - Similar to the previous task, we apply binary inverse thresholding and dilation to connect text lines and reduce noise.
   - The dilated image helps to ensure that the lines within the circle are connected even if they are slightly fragmented.

4. **Contour Detection and Cropping:**
   - Contours are found within the dilated image, and bounding boxes are drawn around potential text lines.
   - Each detected line inside the circle is cropped and saved as a separate image file (e.g., `circle_line_1.png`).

5. **Displaying and Saving Results:**
   - The circle is overlaid on the original image for visualization.
   - The segmented text lines within the circular region are saved as individual image files.

---

## Discussion of Challenges:

- **Congested Lines:** Lines inside the circular seal are likely congested and close together, which can cause detection algorithms to merge them or miss some lines. To mitigate this, the dilation kernel is carefully chosen to connect lines without over-merging them.
  
- **Circular Region Isolation:** Since the lines are inside the circle, it is essential to mask the circular region to focus the line detection algorithm only within that area, avoiding irrelevant portions of the image.

- **Thresholding:** The threshold values for segmentation (used in binary thresholding and dilation) may need to be adjusted depending on the image quality, the text density inside the seal, and the seal size.