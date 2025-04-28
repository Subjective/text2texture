import React from 'react';
import { InputMethod } from '../../hooks/useImageInput'; // Import type

interface InputSectionProps {
  inputMethod: InputMethod;
  textPrompt: string;
  isLoading: boolean; // Combined loading state from workflow
  isLoadingImageGen: boolean; // Specific loading state for text-to-image
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
          <label htmlFor="imageUpload" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Select Image File:
          </label>
          <input
            type="file"
            id="imageUpload" // Keep ID for potential label association or resets
            accept="image/png, image/jpeg, image/webp"
            onChange={onFileChange}
            className="block w-full text-sm text-gray-500 dark:text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 dark:file:bg-gray-700 dark:file:text-gray-300 dark:hover:file:bg-gray-600 disabled:opacity-50"
            disabled={isLoading}
          />
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
            placeholder="e.g., A detailed cobblestone path, top-down view"
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