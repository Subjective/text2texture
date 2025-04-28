// Define backend URLs
export const GENERATE_IMAGE_URL = 'http://127.0.0.1:5000/api/generate_image'; // Endpoint to generate image from text
export const PREDICT_MASK_URL = 'http://127.0.0.1:5000/api/predict_mask'; // Endpoint for SAM2-like mask prediction
export const GENERATE_MASKS_URL = 'http://127.0.0.1:5000/api/generate_masks'; // NEW: Endpoint for automatic mask generation
export const GENERATE_MODEL_URL = 'http://127.0.0.1:5000/api/generate_model'; // Endpoint for final 3D model generation