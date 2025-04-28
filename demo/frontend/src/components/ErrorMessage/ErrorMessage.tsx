import React from 'react';

interface ErrorMessageProps {
  error: string | null;
  onDismiss: () => void; // Callback to clear the error
}

export function ErrorMessage({ error, onDismiss }: ErrorMessageProps) {
  if (!error) {
    return null; // Don't render anything if there's no error
  }

  return (
    <div className="mt-4 p-3 bg-red-100 border border-red-400 text-red-700 dark:bg-red-900 dark:border-red-700 dark:text-red-200 rounded-lg shadow">
      <div className="flex justify-between items-start">
        <div>
          <strong className="font-semibold">Error:</strong>
          {/* Use pre-wrap to preserve potential newlines in error messages */}
          <p className="text-sm whitespace-pre-wrap">{error}</p>
        </div>
        <button
          onClick={onDismiss}
          className="ml-2 text-red-800 dark:text-red-300 hover:text-red-600 dark:hover:text-red-100 flex-shrink-0"
          aria-label="Dismiss error"
        >
          &#x2715; {/* Close icon */}
        </button>
      </div>
    </div>
  );
}