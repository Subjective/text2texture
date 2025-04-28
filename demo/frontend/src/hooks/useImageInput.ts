import { useState, useCallback } from 'react';
import { generateImageFromText } from '../services/api';
import { base64ToBlob } from '../utils/blobUtils';

export type InputMethod = "upload" | "text";

export interface UseImageInputReturn {
  inputMethod: InputMethod;
  imageFile: File | null;
  textPrompt: string;
  imageSrc: string | null; // Base64 Data URL for display/masking
  uploadedImageFilename: string | null;
  isLoadingImageGen: boolean; // Loading state specific to text-to-image
  imageError: string | null; // Error state specific to this hook
  setInputMethod: React.Dispatch<React.SetStateAction<InputMethod>>;
  setTextPrompt: React.Dispatch<React.SetStateAction<string>>;
  handleFileChange: (event: React.ChangeEvent<HTMLInputElement>) => Promise<boolean>; // Return promise indicating success
  handleGenerateImageFromText: () => Promise<{ success: boolean; imageSrc?: string; imageFile?: File; filename?: string }>; // Return details needed by workflow
  resetImageInputState: () => void; // Specific reset for this hook's state
  clearImageError: () => void;
}

export function useImageInput(): UseImageInputReturn {
  const [inputMethod, setInputMethod] = useState<InputMethod>("upload");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [textPrompt, setTextPrompt] = useState<string>("");
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [uploadedImageFilename, setUploadedImageFilename] = useState<string | null>(null);
  const [isLoadingImageGen, setIsLoadingImageGen] = useState<boolean>(false);
  const [imageError, setImageError] = useState<string | null>(null);

  const clearImageError = useCallback(() => {
    setImageError(null);
  }, []);

  // Resets state managed by this hook
  const resetImageInputState = useCallback(() => {
    // setInputMethod("upload"); // Keep current input method? Or reset? Let's keep it.
    setImageFile(null);
    // setTextPrompt(""); // Keep text prompt? Let's clear it.
    setTextPrompt("");
    setImageSrc(null);
    setUploadedImageFilename(null);
    setIsLoadingImageGen(false);
    setImageError(null);
    // Reset file input visually if needed (might need DOM access, handle in component)
    // const fileInput = document.getElementById('imageUpload') as HTMLInputElement;
    // if (fileInput) fileInput.value = "";
  }, []);

  // Handle file selection
  const handleFileChange = useCallback(async (event: React.ChangeEvent<HTMLInputElement>): Promise<boolean> => {
    clearImageError(); // Clear previous errors
    const file = event.target.files?.[0];

    if (file) {
      // Reset relevant state *before* processing new file
      setImageFile(null);
      setImageSrc(null);
      setUploadedImageFilename(null);

      setImageFile(file); // Store the file object
      setUploadedImageFilename(file.name); // Store the original filename
      setInputMethod("upload"); // Ensure input method is correct

      return new Promise<boolean>((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => {
          const result = e.target?.result as string;
          if (typeof result === 'string') {
            setImageSrc(result); // Set Base64 source for display
            resolve(true); // Resolve promise on success
          } else {
            console.error("FileReader result is not a string:", result);
            setImageError("Failed to read the selected image file content.");
            resetImageInputState(); // Reset on error
            resolve(false); // Resolve promise on failure
          }
        };
        reader.onerror = () => {
          console.error("Failed to read file.");
          setImageError("Failed to read the selected file.");
          resetImageInputState(); // Reset on error
          resolve(false); // Resolve promise on failure
        };
        reader.readAsDataURL(file);
      });
    } else {
      // Handle case where user cancels file selection
      if (!imageFile && !imageSrc) {
        resetImageInputState();
      }
      return Promise.resolve(false); // Resolve false if no file selected/cancelled
    }
  }, [clearImageError, resetImageInputState, imageFile, imageSrc]); // Dependencies

  // Handle generating image from text prompt
  const handleGenerateImageFromText = useCallback(async (): Promise<{ success: boolean; imageSrc?: string; imageFile?: File; filename?: string }> => {
    if (!textPrompt.trim()) {
      setImageError("Please enter a text prompt.");
      return { success: false };
    }

    // Reset state before starting generation
    resetImageInputState(); // Clears errors, previous images etc.
    setInputMethod("text"); // Ensure method is text
    setIsLoadingImageGen(true);

    try {
      const base64ImageData = await generateImageFromText(textPrompt); // Call API service
      const fullImageSrc = `data:image/png;base64,${base64ImageData}`; // Assuming PNG
      const generatedFilename = "generated_image.png";

      // Convert Base64 back to a File object for consistency
      const blob = base64ToBlob(base64ImageData, 'image/png');
      const generatedFile = new File([blob], generatedFilename, { type: 'image/png' });

      // Set state *after* successful generation and conversion
      setImageSrc(fullImageSrc);
      setImageFile(generatedFile);
      setUploadedImageFilename(generatedFilename);
      setIsLoadingImageGen(false);
      return { success: true, imageSrc: fullImageSrc, imageFile: generatedFile, filename: generatedFilename };

    } catch (err: any) {
      console.error("Error generating image from text:", err);
      setImageError(err.message || "Failed to generate image from text.");
      resetImageInputState(); // Reset state on error
      setIsLoadingImageGen(false); // Ensure loading is false
      return { success: false };
    }
  }, [textPrompt, resetImageInputState]); // Dependencies

  return {
    inputMethod,
    imageFile,
    textPrompt,
    imageSrc,
    uploadedImageFilename,
    isLoadingImageGen,
    imageError,
    setInputMethod,
    setTextPrompt,
    handleFileChange,
    handleGenerateImageFromText,
    resetImageInputState,
    clearImageError,
  };
}