// Define backend URLs
export const GENERATE_IMAGE_URL = '/api/generate_image'; // Endpoint to generate image from text
export const PREDICT_MASK_URL = '/api/predict_mask'; // Endpoint for SAM2-like mask prediction
export const GENERATE_MASKS_URL = '/api/generate_masks'; // NEW: Endpoint for automatic mask generation
export const GENERATE_MODEL_URL = '/api/generate_model'; // Endpoint for final 3D model generation