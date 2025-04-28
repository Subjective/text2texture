import React from 'react';
import { UseModelParamsReturn } from '../../hooks/useModelParams'; // Import hook return type

interface ParamsSectionProps {
  params: UseModelParamsReturn; // Pass the entire hook return object
  isLoading: boolean; // Overall loading state
  isLoadingModel: boolean; // Specific loading state for model generation
  onGenerateModel: () => void; // Trigger function for generation
  goBack: () => void; // Function to navigate back
}

export function ParamsSection({
  params,
  isLoading,
  isLoadingModel,
  onGenerateModel,
  goBack
}: ParamsSectionProps) {

  return (
    <section id="params-section" className="bg-white dark:bg-gray-800 p-4 md:p-6 rounded-lg shadow-md">
      <h2 className="text-xl md:text-2xl font-semibold mb-4 border-b pb-2 dark:border-gray-600">3. Configure 3D Model</h2>

      {/* Back Button */}
      <button onClick={goBack} disabled={isLoading} className="text-sm text-blue-600 hover:underline mb-4 dark:text-blue-400 disabled:opacity-50">&laquo; Back to Mask Selection</button>

      {/* Parameter Form Fields */}
      <div className="space-y-4">
        {/* Two-column grid for numeric inputs */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label htmlFor="blockWidth" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Block Width (mm)</label>
            <input
              id="blockWidth" type="number" value={params.blockWidth} min="1" step="1"
              onChange={(e) => params.setBlockWidth(Number(e.target.value))}
              className="mt-1 block w-full p-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white disabled:opacity-50"
              disabled={isLoading}
            />
          </div>
          <div>
            <label htmlFor="blockLength" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Block Length (mm)</label>
            <input
              id="blockLength" type="number" value={params.blockLength} min="1" step="1"
              onChange={(e) => params.setBlockLength(Number(e.target.value))}
              className="mt-1 block w-full p-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white disabled:opacity-50"
              disabled={isLoading}
            />
          </div>
          <div>
            <label htmlFor="blockThickness" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Base Thickness (mm)</label>
            <input
              id="blockThickness" type="number" value={params.blockThickness} min="0.1" step="0.1"
              onChange={(e) => params.setBlockThickness(Number(e.target.value))}
              className="mt-1 block w-full p-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white disabled:opacity-50"
              disabled={isLoading}
            />
          </div>
          <div>
            <label htmlFor="depth" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Max Feature Depth (mm)</label>
            <input
              id="depth" type="number" value={params.depth} min="0.1" step="0.1"
              onChange={(e) => params.setDepth(Number(e.target.value))}
              className="mt-1 block w-full p-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white disabled:opacity-50"
              disabled={isLoading}
              title="Maximum height (protrude) or depth (carve) of features relative to the base"
            />
          </div>
          {/* Base Height - might be less commonly used, keep it simple */}
          {/* <div>
            <label htmlFor="baseHeight" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Base Height Offset (mm)</label>
            <input
              id="baseHeight" type="number" value={params.baseHeight} step="0.1"
              onChange={(e) => params.setBaseHeight(Number(e.target.value))}
              className="mt-1 block w-full p-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white disabled:opacity-50"
              disabled={isLoading}
              title="Vertical offset for the entire base"
            />
          </div> */}
        </div>

        {/* Mode, Invert, Color Options */}
        <div className="flex flex-col sm:flex-row justify-between space-y-2 sm:space-y-0 sm:space-x-4 pt-2">
          {/* Mode Selection */}
          <fieldset>
            <legend className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Mode:</legend>
            <div className="flex items-center space-x-3">
              <label className="flex items-center space-x-1 cursor-pointer">
                <input type="radio" name="mode" value="protrude" checked={params.mode === "protrude"} onChange={() => params.setMode("protrude")} className="form-radio h-4 w-4 text-blue-600 border-gray-300 focus:ring-blue-500" disabled={isLoading} />
                <span className="text-sm text-gray-800 dark:text-gray-200">Protrude</span>
              </label>
              <label className="flex items-center space-x-1 cursor-pointer">
                <input type="radio" name="mode" value="carve" checked={params.mode === "carve"} onChange={() => params.setMode("carve")} className="form-radio h-4 w-4 text-blue-600 border-gray-300 focus:ring-blue-500" disabled={isLoading} />
                <span className="text-sm text-gray-800 dark:text-gray-200">Carve</span>
              </label>
            </div>
          </fieldset>

          {/* Invert Checkbox */}
          <div className="flex items-center pt-5"> {/* Added pt-5 for alignment */}
            <input
              id="invert" type="checkbox" checked={params.invert} onChange={(e) => params.setInvert(e.target.checked)}
              className="form-checkbox h-4 w-4 text-blue-600 rounded border-gray-300 focus:ring-blue-500 disabled:opacity-50"
              disabled={isLoading}
            />
            <label htmlFor="invert" className="ml-2 block text-sm text-gray-700 dark:text-gray-300">Invert Heightmap</label>
          </div>

          {/* Include Color Checkbox */}
          <div className="flex items-center pt-5"> {/* Added pt-5 for alignment */}
            <input
              id="includeColor" type="checkbox" checked={params.includeColor} onChange={(e) => params.setIncludeColor(e.target.checked)}
              className="form-checkbox h-4 w-4 text-blue-600 rounded border-gray-300 focus:ring-blue-500 disabled:opacity-50"
              disabled={isLoading}
            />
            <label htmlFor="includeColor" className="ml-2 block text-sm text-gray-700 dark:text-gray-300">Include Color</label>
          </div>
        </div>
      </div>

      {/* Final Generate Button */}
      <div className="mt-6 pt-4 border-t dark:border-gray-600">
        <button
          onClick={onGenerateModel}
          disabled={isLoading || isLoadingModel} // Disable if overall or model-specific loading
          className="w-full py-3 px-4 bg-green-600 hover:bg-green-700 text-white font-semibold rounded-md shadow-sm transition duration-150 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoadingModel ? 'Generating Model...' : `Generate ${params.includeColor ? 'PLY' : 'STL'} Model`}
        </button>
      </div>
    </section>
  );
}
