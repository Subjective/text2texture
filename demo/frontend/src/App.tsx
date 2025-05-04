import React, { useState, useEffect, useCallback } from 'react';

// Import Hooks
import { useAppWorkflow } from './hooks/useAppWorkflow';
import { useImageInput } from './hooks/useImageInput';
import { useMasking } from './hooks/useMasking';
import { useModelParams } from './hooks/useModelParams';
import { useModelGeneration } from './hooks/useModelGeneration';

// Import Components
import { InputSection } from './components/InputSection/InputSection';
import { MaskingSection } from './components/MaskingSection/MaskingSection';
import { ParamsSection } from './components/ParamsSection/ParamsSection';
import { ResultSection } from './components/ResultSection/ResultSection';
import { ErrorMessage } from './components/ErrorMessage/ErrorMessage';
import { LoadingIndicator } from './components/LoadingIndicator/LoadingIndicator';

// Import Types (if needed, though hooks should manage most)
// import { Step } from './types/app.types';

function App() {
  // --- Instantiate Hooks ---
  const workflow = useAppWorkflow();
  const imageInput = useImageInput();
  const masking = useMasking();
  const modelParams = useModelParams();
  const modelGeneration = useModelGeneration();
  const [inputSectionKey, setInputSectionKey] = useState<number>(1); // Key for resetting InputSection
  const [isFullScreen, setIsFullScreen] = useState<boolean>(false); // State for fullscreen toggle

  // --- Hook Coordination & Effects ---

  // Update combined loading state in workflow hook
  useEffect(() => {
    workflow.setIsLoading({
      imageGen: imageInput.isLoadingImageGen,
      mask: masking.isLoadingMask,
      autoMask: masking.isLoadingAutoMask,
      model: modelGeneration.isLoadingModel,
    });
  }, [
    imageInput.isLoadingImageGen,
    masking.isLoadingMask,
    masking.isLoadingAutoMask,
    modelGeneration.isLoadingModel,
    workflow.setIsLoading // Include setIsLoading in deps
  ]);

  // Update combined error state in workflow hook
  useEffect(() => {
    const error = imageInput.imageError || masking.maskingError || modelGeneration.generationError;
    workflow.setError(error);
  }, [imageInput.imageError, masking.maskingError, modelGeneration.generationError, workflow.setError]);

  // Handle image loading for masking hook's imageRef
  useEffect(() => {
    if (imageInput.imageSrc) {
      const img = new Image();
      img.onload = () => {
        console.log(`Image loaded for masking: ${img.naturalWidth}x${img.naturalHeight}`);
        masking.imageRef.current = img;
        // Explicitly trigger redraw now that the image ref is set
        masking.drawCanvas();
      };
      img.onerror = () => {
        console.error("Failed to load image into Image object for masking.");
        workflow.setError("Failed to process the loaded image data.");
        // Reset relevant states if image object fails
        imageInput.resetImageInputState();
        masking.handleClearSavedMasks(); // Clear masks as well
        workflow.setCurrentStep("input");
      };
      img.src = imageInput.imageSrc;
    } else {
      // Clear the imageRef if imageSrc is nullified
      masking.imageRef.current = null;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageInput.imageSrc, workflow.setError, workflow.currentStep]);

  // Trigger canvas redraw when fullscreen is toggled and we have an image
  useEffect(() => {
    if (imageInput.imageSrc && masking.imageRef.current && workflow.currentStep === 'masking') {
      // Small timeout to ensure DOM is fully updated
      const timer = setTimeout(() => {
        masking.drawCanvas();
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [isFullScreen, workflow.currentStep, imageInput.imageSrc, masking]);

  // Handle step transition after image generation
  const handleGenerateImage = useCallback(async () => {
    workflow.setCurrentStep("generating"); // Set step before async call
    masking.handleClearSavedMasks(); // Clear old masks before generating new image
    const result = await imageInput.handleGenerateImageFromText();
    if (result.success) {
      workflow.setCurrentStep("masking");
    } else {
      // If generation fails, move back from 'generating' state
      // (Error handling itself is done via useEffect)
      workflow.setCurrentStep("input"); // Or wherever appropriate on failure
    }
    // Error handling is done via the useEffect watching imageInput.imageError
  }, [imageInput.handleGenerateImageFromText, masking.handleClearSavedMasks, workflow.setCurrentStep]);

  // Handle step transition after model generation
  const handleGenerateModel = useCallback(async () => {
    workflow.setCurrentStep("generating"); // Set step before async call
    const params = modelParams.getParams();
    const activeMasks = masking.savedMasks.filter(m => m.isActive);
    const result = await modelGeneration.handleGenerate3DModel(
      imageInput.imageFile,
      activeMasks,
      params
    );
    if (result) {
      workflow.setCurrentStep("result");
    } else {
      // If generation fails, move back from 'generating' state
      // (Error handling itself is done via useEffect)
      workflow.setCurrentStep("params"); // Or wherever appropriate on failure
    }
    // Error handling via useEffect
  }, [
    modelParams,
    masking.savedMasks,
    imageInput.imageFile,
    modelGeneration.handleGenerate3DModel,
    workflow.setCurrentStep
  ]);

  // Handle file selection and step transition
  const handleFileSelected = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const success = await imageInput.handleFileChange(event);
    if (success) {
      masking.handleClearSavedMasks(); // Clear old masks when new image is loaded
      workflow.setCurrentStep("masking");
    }
    // Error handling is done via the useEffect watching imageInput.imageError
  }, [imageInput.handleFileChange, masking.handleClearSavedMasks, workflow.setCurrentStep]);


  // Handle navigation actions
  const goToParams = useCallback(() => {
    if (!imageInput.imageSrc) {
      workflow.setError("Cannot proceed without an image.");
      workflow.setCurrentStep("input");
    } else {
      workflow.setCurrentStep("params");
    }
  }, [imageInput.imageSrc, workflow.setCurrentStep, workflow.setError]);

  // Implement the combined reset logic
  const handleStartOver = useCallback(() => {
    imageInput.resetImageInputState();
    masking.handleClearSavedMasks(); // Resets masks and points
    modelParams.setBlockWidth(100); // Reset params to defaults (example)
    modelParams.setBlockLength(100);
    modelParams.setBlockThickness(10);
    modelParams.setDepth(25);
    modelParams.setBaseHeight(0);
    modelParams.setMode("protrude");
    modelParams.setInvert(false);
    modelParams.setIncludeColor(false);
    modelGeneration.resetGenerationState();
    workflow.resetApp(); // Resets step, error, loading flags
    setInputSectionKey(prevKey => prevKey + 1); // Increment key to force remount InputSection
  }, [imageInput, masking, modelParams, modelGeneration, workflow]);

  // Handle input method change (includes reset logic)
  const handleInputMethodChange = useCallback((method: typeof imageInput.inputMethod) => {
    handleStartOver(); // Reset everything when switching method
    imageInput.setInputMethod(method);
  }, [handleStartOver, imageInput.setInputMethod]);

  // Handle fullscreen toggle
  const handleFullscreenToggle = useCallback(() => {
    // Toggle fullscreen state
    setIsFullScreen(prev => !prev);
    
    // Ensure input key is incremented to force re-render of file input
    setInputSectionKey(prev => prev + 1);
    
    // Schedule a redraw of the masking canvas after DOM update
    if (imageInput.imageSrc && workflow.currentStep === 'masking') {
      // Allow DOM to update first
      setTimeout(() => {
        masking.drawCanvas();
      }, 100);
    }
  }, [imageInput.imageSrc, masking, workflow.currentStep]);

  // --- Render Logic ---
  return (
    <div className="max-w-6xl mx-auto p-4 md:p-6 text-gray-900 dark:text-gray-100 font-sans overflow-x-hidden">
      {/* Page Header */}
      <header className="text-center mb-6 md:mb-8">
        <h1 className="text-3xl md:text-4xl font-bold">Text2Texture</h1>
        <p className="text-md md:text-lg text-gray-600 dark:text-gray-400 mt-2">
          Generate 3D models from images or text prompts.
        </p>
      </header>

      {/* Error Display Area */}
      <ErrorMessage error={workflow.error} onDismiss={workflow.clearError} />

      {/* Main Content Grid */}
      <main className={`grid ${isFullScreen ? 'grid-cols-1' : 'grid-cols-1 lg:grid-cols-2'} gap-6 md:gap-8 mt-4`}>

        {/* Left Column: Input, Masking, Parameters */}
        {!isFullScreen && (
          <div className="space-y-6 md:space-y-8">

            {/* Step 1: Input */}
            <InputSection
              key={inputSectionKey}
              inputMethod={imageInput.inputMethod}
              textPrompt={imageInput.textPrompt}
              isLoading={workflow.isLoading}
              isLoadingImageGen={imageInput.isLoadingImageGen}
              uploadedImageFilename={imageInput.uploadedImageFilename}
              onInputMethodChange={handleInputMethodChange}
              onTextPromptChange={(e) => imageInput.setTextPrompt(e.target.value)}
              onFileChange={handleFileSelected} // Use the new async handler
              onGenerateImageFromText={handleGenerateImage} // Use coordinated handler
            />

            {/* Step 2: Masking (Conditional) */}
            {/* Only render masking section when in the 'masking' step and an image exists */}
            {workflow.currentStep === 'masking' && imageInput.imageSrc && (
              <MaskingSection
                canvasRef={masking.canvasRef}
                points={masking.points}
                savedMasks={masking.savedMasks}
                selectedBox={masking.selectedBox} // Pass selectedBox
                maskingMode={masking.maskingMode} // Pass maskingMode
                imageSrc={imageInput.imageSrc}
                isLoading={workflow.isLoading}
                isLoadingMask={masking.isLoadingMask}
                isLoadingAutoMask={masking.isLoadingAutoMask}
                setMaskingMode={masking.setMaskingMode} // Pass setMaskingMode
                handleMouseDown={masking.handleMouseDown} // Pass handleMouseDown
                handleMouseMove={masking.handleMouseMove} // Pass handleMouseMove
                handleMouseUp={masking.handleMouseUp} // Pass handleMouseUp
                triggerPrediction={masking.triggerPrediction} // Pass triggerPrediction
                // Pass imageSrc to the handler trigger
                handleGenerateMasksAutomatically={() => masking.handleGenerateMasksAutomatically(imageInput.imageSrc)}
                handleResetCurrentPoints={masking.handleResetCurrentPoints}
                handleToggleMaskActive={masking.handleToggleMaskActive}
                handleRenameMask={masking.handleRenameMask}
                handleDeleteMask={masking.handleDeleteMask}
                handleClearSavedMasks={masking.handleClearSavedMasks}
                handleDownloadActiveMasks={masking.handleDownloadActiveMasks}
                goToParams={goToParams}
              />
            )}

            {/* Step 3: Parameters (Conditional) */}
            {workflow.currentStep === 'params' && imageInput.imageSrc && (
              <ParamsSection
                params={modelParams} // Pass the whole hook result
                isLoading={workflow.isLoading}
                isLoadingModel={modelGeneration.isLoadingModel}
                onGenerateModel={handleGenerateModel} // Use coordinated handler
                goBack={() => workflow.setCurrentStep('masking')}
              />
            )}

            {/* Loading Indicator during Generation Steps */}
            {workflow.isLoading && workflow.currentStep === 'generating' && (
              <LoadingIndicator
                message={
                  masking.isLoadingMask ? "Predicting mask..." :
                    masking.isLoadingAutoMask ? "Generating masks..." :
                      imageInput.isLoadingImageGen ? "Generating image..." :
                        modelGeneration.isLoadingModel ? "Generating 3D model..." : "Processing..."
                }
              />
            )}

          </div>
        )} {/* End Left Column Conditional Render */}


        {/* Right Column: 3D Viewer / Result */}
        <div className={`${!isFullScreen ? 'lg:sticky lg:top-6 h-[60vh] lg:h-[calc(100vh-4rem)]' : 'h-[calc(100vh-8rem)]'} min-h-[400px]`}>
          <section id="viewer-section" className="bg-gray-100 dark:bg-gray-800 p-3 md:p-4 rounded-lg shadow-md h-full flex flex-col">
            <div className="flex justify-between items-center mb-3 flex-shrink-0"> {/* Flex container for title and button */}
              <h2 className="text-lg md:text-xl font-semibold text-gray-800 dark:text-gray-100">
                {modelGeneration.result ? 'Generated Model Preview' : '3D Preview Area'}
              </h2>
              {/* Fullscreen Toggle Button */}
              <button
                onClick={handleFullscreenToggle}
                className="p-1 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200"
                title={isFullScreen ? "Exit Fullscreen" : "Enter Fullscreen"}
                aria-label={isFullScreen ? "Exit Fullscreen" : "Enter Fullscreen"}
              >
                {isFullScreen ? (
                  // Exit fullscreen icon
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M4 14h6m0 0v6m0-6-7 7m17-11h-6m0 0V4m0 6 7-7"/>
                  </svg>
                ) : (
                  // Enter fullscreen icon
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M15 3h6m0 0v6m0-6-7 7M9 21H3m0 0v-6m0 6 7-7"/>
                  </svg>
                )}
              </button>
            </div>

            {/* Loading/Generating State for Viewer */}
            {modelGeneration.isLoadingModel && workflow.currentStep === 'generating' && (
              <div className="flex-grow flex items-center justify-center bg-gray-200 dark:bg-gray-700 rounded-md">
                <div className="text-center p-4">
                  <svg className="animate-spin h-8 w-8 text-blue-600 dark:text-blue-400 mx-auto mb-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
                  </svg>
                  <p className="text-lg font-medium text-gray-700 dark:text-gray-300">Generating 3D Model...</p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">This might take a moment.</p>
                </div>
              </div>
            )}

            {/* Result Display */}
            {workflow.currentStep === 'result' && modelGeneration.result && (
              <ResultSection
                result={modelGeneration.result}
                isLoading={workflow.isLoading}
                onStartOver={handleStartOver}
              />
            )}

            {/* Post-Generation Navigation (Not Fullscreen) */}
            {workflow.currentStep === 'result' && !isFullScreen && (
              <div className="mt-4 flex justify-center gap-4 flex-shrink-0">
                <button
                  onClick={() => workflow.setCurrentStep('masking')}
                  className="px-4 py-2 bg-gray-200 dark:bg-gray-600 text-gray-800 dark:text-gray-100 rounded hover:bg-gray-300 dark:hover:bg-gray-500 transition-colors text-sm"
                >
                  Modify Masks
                </button>
                <button
                  onClick={() => workflow.setCurrentStep('params')}
                  className="px-4 py-2 bg-gray-200 dark:bg-gray-600 text-gray-800 dark:text-gray-100 rounded hover:bg-gray-300 dark:hover:bg-gray-500 transition-colors text-sm"
                >
                  Modify Parameters
                </button>
              </div>
            )}

            {/* Placeholder when no result and not loading */}
            {!workflow.isLoading && workflow.currentStep !== 'result' && workflow.currentStep !== 'generating' && (
              <div className="flex-grow flex items-center justify-center bg-gray-200 dark:bg-gray-700 rounded-md text-center p-4">
                <p className="text-gray-500 dark:text-gray-400">
                  {workflow.currentStep === 'input' ? 'Upload or generate an image to begin.' :
                    workflow.currentStep === 'masking' ? 'Select points and predict masks, or proceed to parameters.' :
                      workflow.currentStep === 'params' ? 'Configure parameters and click "Generate 3D Model".' :
                        '3D model preview will appear here.'}
                </p>
              </div>
            )}

          </section>
        </div> {/* End Right Column */}

      </main>

      {/* Footer */}
      <footer className="text-center mt-8 md:mt-12 text-gray-500 dark:text-gray-400 text-sm">
        © 2025 Joshua Yin
      </footer>
    </div>
  );
}

export default App;
