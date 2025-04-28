import React from 'react';
import { SavedMask } from '../../types/app.types';
import { MaskListItem } from './MaskListItem'; // Import the item component

interface MaskListProps {
  savedMasks: SavedMask[];
  isLoading: boolean;
  onToggleActive: (id: string) => void;
  onDelete: (id: string) => void;
  onRename: (id: string, newName: string) => void;
  onClearAll: () => void; // Add handler for clearing all masks
  onDownloadActive: () => void; // Add handler for downloading active masks
}

export function MaskList({
  savedMasks,
  isLoading,
  onToggleActive,
  onDelete,
  onRename,
  onClearAll,
  onDownloadActive
}: MaskListProps) {

  const activeMaskCount = savedMasks.filter(m => m.isActive).length;

  if (savedMasks.length === 0) {
    return null; // Don't render anything if there are no masks
  }

  return (
    <div className="mt-4 border-t pt-4 dark:border-gray-600">
      <h3 className="text-lg font-semibold mb-2 text-gray-700 dark:text-gray-200">
        Saved Masks ({activeMaskCount} active)
      </h3>
      <ul className="space-y-1 max-h-48 overflow-y-auto pr-2 border rounded-md p-1 dark:border-gray-600">
        {savedMasks.map((mask, index) => (
          <MaskListItem
            key={mask.id}
            mask={mask}
            index={index}
            isLoading={isLoading} // Pass combined loading state
            onToggleActive={onToggleActive}
            onDelete={onDelete}
            onRename={onRename}
          />
        ))}
      </ul>
      <div className="flex flex-wrap justify-start gap-2 mt-3">
        <button
          onClick={onClearAll}
          disabled={isLoading}
          className="px-3 py-1 text-xs bg-red-100 text-red-700 rounded-md hover:bg-red-200 dark:bg-red-900 dark:text-red-200 dark:hover:bg-red-800 disabled:opacity-50"
          title="Remove all saved masks"
        >
          Clear All Masks
        </button>
        <button
          onClick={onDownloadActive}
          disabled={isLoading || activeMaskCount === 0}
          className="px-3 py-1 text-xs bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 dark:bg-gray-600 dark:text-gray-200 dark:hover:bg-gray-500 disabled:opacity-50"
          title="Download active masks as PNG files"
        >
          Download Active Masks
        </button>
      </div>
    </div>
  );
}