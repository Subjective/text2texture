import { useState, useCallback } from 'react';
import { Step } from '../types/app.types'; // Import Step type

export interface UseAppWorkflowReturn {
  currentStep: Step;
  error: string | null; // Combined error state
  isLoading: boolean; // Combined loading state (derived from other hooks)
  setCurrentStep: React.Dispatch<React.SetStateAction<Step>>;
  setError: (errorMessage: string | null) => void; // Centralized error setter
  clearError: () => void;
  // Placeholder for combined loading state logic
  setIsLoading: (loadingState: {
      mask?: boolean;
      autoMask?: boolean;
      model?: boolean;
      imageGen?: boolean;
  }) => void;
  // Placeholder for a function that calls resets on other hooks
  resetApp: () => void;
}

// Note: The actual implementation of combined isLoading and resetApp
// will depend on how the hooks are integrated in the main App component.
// This hook primarily manages the step and the central error message.

export function useAppWorkflow(initialStep: Step = "input"): UseAppWorkflowReturn {
  const [currentStep, setCurrentStep] = useState<Step>(initialStep);
  const [error, setGlobalError] = useState<string | null>(null);

  // Individual loading states managed internally for the combined flag
  const [loadingStates, setLoadingStates] = useState({
      mask: false,
      autoMask: false,
      model: false,
      imageGen: false,
  });

  // Derived combined loading state
  const isLoading = loadingStates.mask || loadingStates.autoMask || loadingStates.model || loadingStates.imageGen;

  const clearError = useCallback(() => {
    setGlobalError(null);
  }, []);

  const setError = useCallback((errorMessage: string | null) => {
    setGlobalError(errorMessage);
  }, []);

  // Function to update parts of the loading state
  const setIsLoading = useCallback((loadingUpdate: Partial<typeof loadingStates>) => {
      setLoadingStates(prev => ({ ...prev, ...loadingUpdate }));
  }, []);

  // Placeholder reset function - will need integration with other hooks' resets
  const resetApp = useCallback(() => {
    console.log("Resetting app workflow...");
    setCurrentStep("input");
    clearError();
    // In the real App component, call reset functions from other hooks here
    // e.g., imageInput.resetImageInputState(); masking.handleClearSavedMasks(); etc.
    setIsLoading({ mask: false, autoMask: false, model: false, imageGen: false }); // Reset all loading flags
  }, [clearError]);


  return {
    currentStep,
    error,
    isLoading,
    setCurrentStep,
    setError,
    clearError,
    setIsLoading, // Expose this to let other hooks report their status
    resetApp,
  };
}