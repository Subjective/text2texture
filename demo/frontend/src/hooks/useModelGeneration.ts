import { useState, useCallback } from 'react';
import { generate3DModel } from '../services/api';
import { SavedMask } from '../types/app.types';
import { ModelParamsState } from './useModelParams'; // Import params state type

export interface ModelGenerationResult {
  fileUrl: string;
  fileType: string;
  heightmapUrl: string | null;
}

export interface UseModelGenerationReturn {
  result: ModelGenerationResult | null;
  isLoadingModel: boolean;
  generationError: string | null;
  handleGenerate3DModel: (
    imageFile: File | null, // Make nullable to handle error case
    activeMasks: SavedMask[],
    params: ModelParamsState
  ) => Promise<ModelGenerationResult | null>; // Return result or null on failure
  resetGenerationState: () => void;
  clearGenerationError: () => void;
}

export function useModelGeneration(): UseModelGenerationReturn {
  const [result, setResult] = useState<ModelGenerationResult | null>(null);
  const [isLoadingModel, setIsLoadingModel] = useState<boolean>(false);
  const [generationError, setGenerationError] = useState<string | null>(null);

  const clearGenerationError = useCallback(() => {
    setGenerationError(null);
  }, []);

  const resetGenerationState = useCallback(() => {
    setResult(null);
    setIsLoadingModel(false);
    setGenerationError(null);
  }, []);

  const handleGenerate3DModel = useCallback(
    async (
      imageFile: File | null,
      activeMasks: SavedMask[],
      params: ModelParamsState
    ): Promise<ModelGenerationResult | null> => {
      clearGenerationError();

      if (!imageFile) {
        setGenerationError("Source image file is missing. Please re-upload or regenerate.");
        return null; // Return null to indicate failure
      }

      setIsLoadingModel(true);
      setResult(null); // Clear previous result

      try {
        const generationResult = await generate3DModel(imageFile, activeMasks, params);
        setResult(generationResult);
        setIsLoadingModel(false);
        return generationResult; // Return the result on success
      } catch (err: any) {
        console.error("Error generating 3D model:", err);
        setGenerationError(err.message || "An unexpected error occurred during 3D model generation.");
        setIsLoadingModel(false);
        setResult(null); // Ensure result is null on error
        return null; // Return null on failure
      }
    },
    [clearGenerationError] // Dependencies
  );

  return {
    result,
    isLoadingModel,
    generationError,
    handleGenerate3DModel,
    resetGenerationState,
    clearGenerationError,
  };
}