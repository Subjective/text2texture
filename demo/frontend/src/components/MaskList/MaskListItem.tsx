import React, { useState, useEffect } from 'react';
import { SavedMask, TextureType } from '../../types/app.types'; // Update import to include TextureType

// Props for individual mask list items
export interface MaskListItemProps {
  mask: SavedMask;
  index: number;
  isLoading: boolean; // Combined loading state
  onToggleActive: (id: string) => void;
  onDelete: (id: string) => void;
  onRename: (id: string, newName: string) => void; // Pass rename handler
  onTextureTypeChange: (id: string, textureType: TextureType) => void; // Add texture type handler
}

export function MaskListItem({ 
  mask, 
  index, 
  isLoading, 
  onToggleActive, 
  onDelete, 
  onRename,
  onTextureTypeChange 
}: MaskListItemProps) {
  // Local state to manage the input field value during editing
  const [editingName, setEditingName] = useState<string>(mask.name);

  // Update local state if the mask name prop changes from outside (e.g., initial load)
  useEffect(() => {
    setEditingName(mask.name);
  }, [mask.name]);

  // Handle changes in the input field
  const handleNameChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setEditingName(event.target.value); // Update local state only
  };

  // Handle saving the name when the input loses focus
  const handleNameBlur = () => {
    // Only call the parent rename function if the name actually changed
    if (editingName !== mask.name) {
      // Use default name if editingName is empty, otherwise use editingName
      onRename(mask.id, editingName.trim() || `Mask ${index + 1}`);
    }
    // If the user clears the input and blurs, reset local state to the (potentially new default) prop name
    if (!editingName.trim()) {
      // Ensure the parent state's name is reflected back if input was cleared
      // This relies on the parent updating the mask.name prop correctly after onRename
      setEditingName(mask.name);
    }
  };

  // Handle saving name on Enter key press
  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter') {
      handleNameBlur(); // Trigger the same logic as blur
      event.currentTarget.blur(); // Optionally remove focus
    } else if (event.key === 'Escape') {
      setEditingName(mask.name); // Revert changes on Escape
      event.currentTarget.blur();
    }
  };

  // Handle texture type change
  const handleTextureTypeChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    onTextureTypeChange(mask.id, event.target.value as TextureType);
  };

  return (
    <li className="flex items-center justify-between p-2 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700">
      <input
        type="text"
        value={editingName} // Bind to local state
        onChange={handleNameChange} // Update local state
        onBlur={handleNameBlur} // Update parent state on blur
        onKeyDown={handleKeyDown} // Handle Enter/Escape keys
        className="text-sm text-gray-800 dark:text-gray-200 bg-transparent border-b border-gray-300 dark:border-gray-500 focus:outline-none focus:border-blue-500 mr-2 flex-grow"
        placeholder={`Mask ${index + 1}`}
        disabled={isLoading}
        aria-label={`Mask name for mask ${index + 1}`}
      />
      
      {/* Texture type dropdown */}
      <select
        value={mask.textureType}
        onChange={handleTextureTypeChange}
        className="text-xs bg-transparent border border-gray-300 dark:border-gray-500 rounded px-1 py-0.5 text-gray-700 dark:text-gray-300 focus:outline-none focus:border-blue-500 mr-2"
        disabled={isLoading}
        aria-label={`Texture type for mask ${index + 1}`}
      >
        <option value="checkerboard">Checkerboard</option>
        <option value="vertical_stripes">Vertical Stripes</option>
        <option value="horizontal_stripes">Horizontal Stripes</option>
        <option value="auto">Auto</option>
      </select>
      
      <span className="text-xs text-gray-500 dark:text-gray-400 mr-2 flex-shrink-0">
        (Score: {mask.score?.toFixed(2) ?? 'N/A'})
      </span>
      <div className="flex items-center space-x-3 flex-shrink-0">
        <label className="flex items-center space-x-1 cursor-pointer" title="Toggle visibility/usage">
          <input
            type="checkbox"
            checked={mask.isActive}
            onChange={() => onToggleActive(mask.id)}
            className="form-checkbox h-4 w-4 text-blue-600 rounded border-gray-300 focus:ring-blue-500 disabled:opacity-50"
            disabled={isLoading}
          />
          <span className="text-xs text-gray-700 dark:text-gray-300">Use</span>
        </label>
        <button
          onClick={() => onDelete(mask.id)}
          title="Delete Mask"
          className="text-red-500 hover:text-red-700 disabled:opacity-50"
          disabled={isLoading}
          aria-label={`Delete mask ${index + 1}`}
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
          </svg>
        </button>
      </div>
    </li>
  );
}