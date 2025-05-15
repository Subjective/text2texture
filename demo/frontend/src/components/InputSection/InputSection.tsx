import React from 'react';
import { InputMethod } from '../../hooks/useImageInput'; // Import type

interface InputSectionProps {
  inputMethod: InputMethod;
  textPrompt: string;
  isLoading: boolean; // Combined loading state from workflow
  isLoadingImageGen: boolean; // Specific loading state for text-to-image
  uploadedImageFilename: string | null; // Filename from state
  lastPromptForPlaceholder?: string | null; // Last prompt for placeholder
  onInputMethodChange: (method: InputMethod) => void;
  onTextPromptChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
  onFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onGenerateImageFromText: () => void; // Simple trigger function
}

export function InputSection({
  inputMethod,
  textPrompt,
  isLoading,
  isLoadingImageGen,
  uploadedImageFilename,
  lastPromptForPlaceholder,
  onInputMethodChange,
  onTextPromptChange,
  onFileChange,
  onGenerateImageFromText
}: InputSectionProps) {

  return (
    <section id="input-section" className="bg-white dark:bg-gray-800 p-4 md:p-6 rounded-lg shadow-md">
      <h2 className="text-xl md:text-2xl font-semibold mb-4 border-b pb-2 dark:border-gray-600">1. Input Image</h2>

      {/* Input Method Selection */}
      <fieldset className="mb-4">
        <legend className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Choose Input Method:</legend>
        <div className="flex items-center space-x-4">
          <label className="flex items-center space-x-2 cursor-pointer">
            <input
              type="radio"
              name="inputMethod"
              value="upload"
              checked={inputMethod === "upload"}
              onChange={() => onInputMethodChange("upload")}
              className="form-radio h-4 w-4 text-blue-600 border-gray-300 focus:ring-blue-500"
              disabled={isLoading}
            />
            <span className="text-sm text-gray-800 dark:text-gray-200">Upload Image</span>
          </label>
          <label className="flex items-center space-x-2 cursor-pointer">
            <input
              type="radio"
              name="inputMethod"
              value="text"
              checked={inputMethod === "text"}
              onChange={() => onInputMethodChange("text")}
              className="form-radio h-4 w-4 text-blue-600 border-gray-300 focus:ring-blue-500"
              disabled={isLoading}
            />
            <span className="text-sm text-gray-800 dark:text-gray-200">Generate from Text</span>
          </label>
        </div>
      </fieldset>

      {/* Conditional Input Fields */}
      {inputMethod === "upload" && (
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Select Image File:
          </label>
          <div className="flex flex-col">
            {/* Custom file input that hides the native browser UI */}
            <div className="relative">
              <input
                type="file"
                id="imageUpload"
                accept="image/png, image/jpeg, image/webp"
                onChange={onFileChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                disabled={isLoading}
                aria-label="Upload file"
              />
              <div className="flex items-center">
                <button
                  type="button"
                  className="py-2 px-4 bg-blue-50 text-blue-700 text-sm font-semibold rounded-l-lg border border-blue-100 hover:bg-blue-100 dark:bg-gray-700 dark:text-gray-300 dark:border-gray-600 dark:hover:bg-gray-600 disabled:opacity-50"
                  disabled={isLoading}
                >
                  Choose File
                </button>
                <div className="flex-grow px-3 py-2 text-sm text-gray-500 dark:text-gray-400 border border-l-0 border-gray-300 dark:border-gray-600 rounded-r-lg">
                  {uploadedImageFilename || "No file chosen"}
                </div>
              </div>
            </div>
          </div>
          <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">PNG, JPG, or WEBP files accepted.</p>
        </div>
      )}

      {inputMethod === "text" && (
        <div>
          <label htmlFor="textPrompt" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Enter Text Prompt:
          </label>
          <textarea
            id="textPrompt"
            rows={3}
            value={textPrompt}
            onChange={onTextPromptChange}
            placeholder={lastPromptForPlaceholder || "e.g., Cows on mars, with asteroids in the background"}
            className="block w-full p-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white disabled:opacity-50"
            disabled={isLoading}
          />
          <button
            onClick={onGenerateImageFromText}
            disabled={isLoading || isLoadingImageGen || !textPrompt.trim()} // Disable if overall loading, image gen loading, or no prompt
            className="mt-2 px-4 py-2 bg-blue-600 text-white text-sm font-semibold rounded-lg shadow hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoadingImageGen ? 'Generating...' : 'Generate Image'}
          </button>
        </div>
      )}
    </section>
  );
}
