import { ModelViewer } from '../ModelViewer/ModelViewer'; // Import the viewer
import { ModelGenerationResult } from '../../hooks/useModelGeneration'; // Import result type

interface ResultSectionProps {
  result: ModelGenerationResult | null;
  isLoading: boolean; // Overall loading state
  onStartOver: () => void; // Function to reset the app
}

export function ResultSection({ result, isLoading, onStartOver }: ResultSectionProps) {

  if (!result) {
    // This section should ideally only be rendered when there's a result,
    // but we can add a fallback just in case.
    return (
        <div className="flex-grow flex items-center justify-center bg-gray-200 dark:bg-gray-800 rounded-md text-center p-4">
            <p className="text-gray-500 dark:text-gray-400">Waiting for generation result...</p>
        </div>
    );
  }

  const { fileUrl, fileType, heightmapUrl } = result;

  return (
    <div className="flex-grow flex flex-col min-h-0"> {/* Ensure flex container takes height */}
      <div className="mb-3 flex-shrink-0 flex flex-wrap gap-2">
        <a
          href={fileUrl}
          download // Suggest download
          className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 dark:focus:ring-offset-gray-800"
        >
          <svg className="-ml-1 mr-2 h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
            <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L10 11.586l2.293-2.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clipRule="evenodd" />
            <path fillRule="evenodd" d="M10 3a1 1 0 011 1v6a1 1 0 11-2 0V4a1 1 0 011-1z" clipRule="evenodd" />
          </svg>
          Download Model ({fileType.toUpperCase()})
        </a>
        {/* Add Heightmap Download Button */}
        {heightmapUrl && (
          <a
            href={heightmapUrl}
            download // Suggest download
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 dark:focus:ring-offset-gray-800"
          >
            <svg className="-ml-1 mr-2 h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
              <path fillRule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z" clipRule="evenodd" /> {/* Simple image icon */}
            </svg>
            Download Heightmap (PNG)
          </a>
        )}
        <button
          onClick={onStartOver}
          disabled={isLoading} // Disable if any loading is happening
          className="inline-flex items-center px-4 py-2 border border-gray-300 dark:border-gray-600 text-sm font-medium rounded-md shadow-sm text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 dark:focus:ring-offset-gray-800 disabled:opacity-50"
        >
          Start Over
        </button>
      </div>
      {/* Model Viewer Container */}
      {/* Key forces remount on resultUrl change if needed */}
      <div className="flex-grow border border-gray-300 dark:border-gray-600 rounded-lg overflow-hidden bg-gray-200 dark:bg-gray-800 min-h-0">
        <ModelViewer key={fileUrl} fileUrl={fileUrl} fileType={fileType} />
      </div>
    </div>
  );
}