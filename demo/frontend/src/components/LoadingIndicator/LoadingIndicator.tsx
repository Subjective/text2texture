interface LoadingIndicatorProps {
  message?: string; // Optional message to display
  size?: 'sm' | 'md' | 'lg'; // Optional size control
}

export function LoadingIndicator({ message = "Processing...", size = 'md' }: LoadingIndicatorProps) {
  const sizeClasses = {
    sm: 'h-5 w-5',
    md: 'h-8 w-8',
    lg: 'h-12 w-12',
  };

  return (
    <div className="flex items-center justify-center p-4 bg-gray-100 dark:bg-gray-700 rounded-lg shadow">
      <svg
        className={`animate-spin mr-3 text-blue-600 dark:text-blue-400 ${sizeClasses[size]}`}
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 24 24"
      >
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
      </svg>
      <span className="text-gray-700 dark:text-gray-300">{message} Please wait.</span>
    </div>
  );
}