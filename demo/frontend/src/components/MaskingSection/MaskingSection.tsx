import React, { MouseEvent } from 'react'; // Added MouseEvent
import { Point, SavedMask, BoundingBox, MaskingMode } from '../../types/app.types'; // Added BoundingBox, MaskingMode
import { MaskList } from '../MaskList/MaskList';

interface MaskingSectionProps {
  // Refs
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  // State from useMasking
  points: Point[];
  savedMasks: SavedMask[];
  selectedBox: BoundingBox | null;
  maskingMode: MaskingMode;
  imageSrc: string | null;
  // Loading states
  isLoading: boolean;
  isLoadingMask: boolean;
  isLoadingAutoMask: boolean;
  // Handlers from useMasking hook
  setMaskingMode: (mode: MaskingMode) => void;
  handleMouseDown: (event: MouseEvent<HTMLCanvasElement>) => void;
  handleMouseMove: (event: MouseEvent<HTMLCanvasElement>) => void;
  handleMouseUp: (event: MouseEvent<HTMLCanvasElement>) => void;
  triggerPrediction: () => Promise<void>; // Renamed
  handleGenerateMasksAutomatically: () => Promise<void>;
  handleResetCurrentPoints: () => void; // Renamed in hook, but keeps same name here for clarity
  handleToggleMaskActive: (id: string) => void;
  handleRenameMask: (id: string, newName: string) => void;
  handleDeleteMask: (id: string) => void;
  handleClearSavedMasks: () => void;
  handleDownloadActiveMasks: () => void;
  // Navigation
  goToParams: () => void;
}

export function MaskingSection({
  canvasRef,
  points,
  savedMasks,
  selectedBox, // New prop
  maskingMode, // New prop
  imageSrc,
  isLoading,
  isLoadingMask,
  isLoadingAutoMask,
  setMaskingMode, // New prop
  handleMouseDown, // New prop
  handleMouseMove, // New prop
  handleMouseUp, // New prop
  triggerPrediction, // Renamed prop
  handleGenerateMasksAutomatically,
  handleResetCurrentPoints,
  handleToggleMaskActive,
  handleRenameMask,
  handleDeleteMask,
  handleClearSavedMasks,
  handleDownloadActiveMasks,
  goToParams
}: MaskingSectionProps) {

  // Don't render if no image is loaded yet
  if (!imageSrc) {
    // Optionally return a placeholder or null
    return (
      <section id="masking-section" className="bg-white dark:bg-gray-800 p-4 md:p-6 rounded-lg shadow-md opacity-50">
        <h2 className="text-xl md:text-2xl font-semibold mb-4 border-b pb-2 dark:border-gray-600">2. Select Masks (Optional)</h2>
        <p className="text-sm text-gray-600 dark:text-gray-400">Please upload or generate an image first.</p>
      </section>
    );
  }

  return (
    <section id="masking-section" className="bg-white dark:bg-gray-800 p-4 md:p-6 rounded-lg shadow-md">
      <h2 className="text-xl md:text-2xl font-semibold mb-4 border-b pb-2 dark:border-gray-600">2. Select Masks (Optional)</h2>

      {/* Mode Selection */}
      <div className="flex gap-2 mb-3">
        <button
          onClick={() => setMaskingMode('point')}
          disabled={isLoading}
          className={`px-3 py-1 rounded-md text-sm font-medium focus:outline-none focus:ring-2 focus:ring-offset-2 dark:focus:ring-offset-gray-800 disabled:opacity-50 ${
            maskingMode === 'point'
              ? 'bg-blue-600 text-white focus:ring-blue-500'
              : 'bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-500 focus:ring-gray-400'
          }`}
        >
          Select Points
        </button>
        <button
          onClick={() => setMaskingMode('box')}
          disabled={isLoading}
          className={`px-3 py-1 rounded-md text-sm font-medium focus:outline-none focus:ring-2 focus:ring-offset-2 dark:focus:ring-offset-gray-800 disabled:opacity-50 ${
            maskingMode === 'box'
              ? 'bg-blue-600 text-white focus:ring-blue-500'
              : 'bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-500 focus:ring-gray-400'
          }`}
        >
          Draw Box
        </button>
      </div>

      {/* Instructions based on mode */}
      {maskingMode === 'point' && (
        <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">Click on the image to place points for manual mask prediction.</p>
      )}
      {maskingMode === 'box' && (
         <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">Click and drag on the image to draw a bounding box for mask prediction.</p>
      )}
       <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">Alternatively, use 'Generate Masks Automatically' to detect objects.</p>


      {/* Canvas Area for Masking */}
      <div className="w-full aspect-[4/3] bg-gray-200 dark:bg-gray-700 rounded-lg shadow overflow-hidden relative mb-4 border dark:border-gray-600 min-h-[200px]">
        <canvas
          ref={canvasRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          // Update cursor based on mode and loading state
          className={`w-full h-full block ${isLoadingMask ? 'cursor-wait' : (maskingMode === 'box' ? 'cursor-crosshair' : 'cursor-copy')} ${isLoading ? 'opacity-50 pointer-events-none' : ''}`}
          style={{ imageRendering: 'pixelated' }}
        />
        {(isLoadingMask || isLoadingAutoMask) && ( // Show overlay for either mask loading state
          <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center z-10 rounded-lg">
            <svg className="animate-spin h-8 w-8 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
            </svg>
            <span className="ml-3 text-white font-medium">
              {isLoadingAutoMask ? 'Generating masks...' : 'Predicting mask...'}
            </span>
          </div>
        )}
      </div>

      {/* Display Current Selection Info */}
      {maskingMode === 'point' && points.length > 0 && (
        <div className="mb-3 p-2 bg-blue-50 dark:bg-gray-700 border border-blue-200 dark:border-blue-800 rounded text-sm">
          <p className="text-blue-800 dark:text-blue-200">
            {points.length} {points.length === 1 ? 'point' : 'points'} selected.
          </p>
        </div>
      )}
      {maskingMode === 'box' && selectedBox && (
         <div className="mb-3 p-2 bg-green-50 dark:bg-gray-700 border border-green-200 dark:border-green-800 rounded text-sm">
           <p className="text-green-800 dark:text-green-200">
             Box selected: [{selectedBox.map(c => Math.round(c)).join(', ')}]. Ready to predict.
           </p>
         </div>
       )}

      {/* Masking Buttons */}
      <div className="flex flex-wrap justify-start gap-2 mb-4">
        {/* Automatic Generation Button */}
        <button
          onClick={handleGenerateMasksAutomatically}
          // Disable if overall loading, or if no image source exists
          disabled={isLoading || !imageSrc}
          className="px-4 py-2 bg-purple-600 text-white text-sm font-semibold rounded-lg shadow hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-opacity-50 disabled:opacity-50 disabled:cursor-not-allowed"
          title="Automatically detect objects and generate masks (replaces existing masks)"
        >
          {isLoadingAutoMask ? 'Generating...' : 'Generate Masks'}
        </button>
        {/* Manual Point-based Prediction Button */}
        {/* Prediction Button (Points or Box) */}
        <button
          onClick={triggerPrediction}
          disabled={isLoading || (maskingMode === 'point' && points.length === 0) || (maskingMode === 'box' && !selectedBox)}
          className="px-4 py-2 bg-blue-600 text-white text-sm font-semibold rounded-lg shadow hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 disabled:opacity-50 disabled:cursor-not-allowed"
          title={maskingMode === 'point' ? "Predict mask from selected points" : "Predict mask from selected box"}
        >
          {isLoadingMask ? 'Predicting...' : `Predict with ${maskingMode === 'point' ? 'Points' : 'Box'}`}
        </button>
        {/* Reset Button (Points or Box) */}
        <button
          onClick={handleResetCurrentPoints} // This handler now clears points OR box
          disabled={isLoading || (maskingMode === 'point' && points.length === 0) || (maskingMode === 'box' && !selectedBox)}
          className="px-4 py-2 bg-yellow-500 text-white text-sm font-semibold rounded-lg shadow hover:bg-yellow-600 focus:outline-none focus:ring-2 focus:ring-yellow-400 focus:ring-opacity-50 disabled:opacity-50 disabled:cursor-not-allowed"
          title={maskingMode === 'point' ? "Clear selected points" : "Clear selected box"}
        >
          Reset Selection
        </button>
      </div>

      {/* Saved Masks List */}
      <MaskList
        savedMasks={savedMasks}
        isLoading={isLoading}
        onToggleActive={handleToggleMaskActive}
        onDelete={handleDeleteMask}
        onRename={handleRenameMask}
        onClearAll={handleClearSavedMasks}
        onDownloadActive={handleDownloadActiveMasks}
      />

      {/* Navigation Button */}
      <div className="mt-6 text-center">
        <button
          onClick={goToParams}
          disabled={isLoading}
          className="px-6 py-2 bg-green-600 text-white font-semibold rounded-lg shadow hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-50 disabled:opacity-50"
        >
          Proceed to 3D Parameters &raquo;
        </button>
      </div>

    </section>
  );
}
