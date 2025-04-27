import React, { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios'; // Use axios for consistency
import { v4 as uuidv4 } from 'uuid'; // Import uuid for unique IDs
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader";

// --- Constants ---
// Define backend URLs
const GENERATE_IMAGE_URL = 'http://127.0.0.1:5000/api/generate_image'; // Endpoint to generate image from text
const PREDICT_MASK_URL = 'http://127.0.0.1:5000/api/predict_mask'; // Endpoint for SAM2-like mask prediction
const GENERATE_MODEL_URL = 'http://127.0.0.1:5000/api/generate_model'; // Endpoint for final 3D model generation

// --- Types ---
// Basic Point type
type Point = {
  x: number;
  y: number;
};

// Structure for storing saved masks
type SavedMask = {
  id: string; // Unique identifier
  name: string; // User-editable name
  maskB64Png: string; // Base64 PNG string
  points: Point[]; // Points used to generate this mask
  isActive: boolean; // Whether the mask should be displayed/used
  loadedImage: HTMLImageElement | null; // Loaded image element for drawing
  score?: number; // Optional score from backend
};

// --- Helper Functions ---
// Helper to convert Base64 string to Blob (needed for FormData)
const base64ToBlob = (base64: string, contentType = '', sliceSize = 512): Blob => {
  const byteCharacters = atob(base64);
  const byteArrays = [];

  for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {
    const slice = byteCharacters.slice(offset, offset + sliceSize);
    const byteNumbers = new Array(slice.length);
    for (let i = 0; i < slice.length; i++) {
      byteNumbers[i] = slice.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    byteArrays.push(byteArray);
  }

  return new Blob(byteArrays, { type: contentType });
};


// --- ModelViewer Component ---
function ModelViewer({ fileUrl, fileType }: { fileUrl: string | null; fileType: string | null }) {
  const [geometry, setGeometry] = useState<any>(null); // Use 'any' for simplicity with different loader types
  const [loadingError, setLoadingError] = useState<string | null>(null);

  useEffect(() => {
    if (!fileUrl || !fileType) {
      setGeometry(null);
      setLoadingError(null);
      return;
    };

    setGeometry(null); // Reset previous geometry
    setLoadingError(null); // Reset error

    let loader: STLLoader | PLYLoader;
    if (fileType.toLowerCase() === "stl") {
      loader = new STLLoader();
    } else if (fileType.toLowerCase() === "ply") {
      loader = new PLYLoader();
    } else {
      console.error("Unsupported file type for viewer:", fileType);
      setLoadingError(`Cannot display file type: ${fileType}`);
      return;
    }

    loader.load(
      fileUrl,
      (geom) => {
        if (fileType.toLowerCase() === "ply") {
          // Ensure geometry is BufferGeometry and compute normals
          if ('computeVertexNormals' in geom) {
            geom.computeVertexNormals(); // Important for lighting on PLY
          } else {
            console.warn("Loaded PLY geometry does not have computeVertexNormals method.");
          }
        }
        // Center the geometry before setting state
        if ('center' in geom) {
          geom.center();
        }
        setGeometry(geom);
      },
      undefined, // onProgress callback (optional)
      (err) => { // onError callback
        console.error("Error loading model:", err);
        setLoadingError("Failed to load 3D model preview.");
      }
    );

    // Cleanup function (optional, loaders might not need explicit disposal)
    return () => {
      // Potential cleanup if needed
    };

  }, [fileUrl, fileType]);

  if (loadingError) {
    return (
      <div className="flex items-center justify-center h-full text-lg font-medium text-red-600 bg-red-50 dark:bg-gray-800 dark:text-red-400 p-4 rounded-md">
        Error: {loadingError}
      </div>
    );
  }

  if (!geometry) {
    return (
      <div className="flex items-center justify-center h-full text-lg font-medium text-gray-600 dark:text-gray-300">
        Loading model preview...
      </div>
    );
  }

  return (
    <Canvas className="w-full h-full" camera={{ position: [150, 150, 150], fov: 50 }}> {/* Adjusted fov */}
      <ambientLight intensity={0.7} /> {/* Slightly increased ambient light */}
      <directionalLight intensity={1.0} position={[50, 100, 150]} /> {/* Stronger main light */}
      <directionalLight intensity={0.5} position={[-80, -40, -100]} /> {/* Back/fill light */}
      <mesh geometry={geometry} scale={1}> {/* Ensure scale is appropriate, geometry is centered */}
        {/* Primitive is used when geometry is loaded directly */}
        <primitive object={geometry} attach="geometry" />
        {fileType === "ply" && geometry.attributes.color ? (
          // Use vertexColors if they exist in the PLY
          <meshStandardMaterial vertexColors={true} roughness={0.7} metalness={0.1} />
        ) : (
          // Default material for STL or PLY without color
          <meshStandardMaterial color="#cccccc" roughness={0.6} metalness={0.2} />
        )}
      </mesh>
      <OrbitControls />
    </Canvas>
  );
}


// --- Main App Component ---
function App() {
  // --- State Variables ---

  // Input state
  const [inputMethod, setInputMethod] = useState<"upload" | "text">("upload");
  const [imageFile, setImageFile] = useState<File | null>(null); // Holds the File object for upload
  const [textPrompt, setTextPrompt] = useState<string>(""); // Holds the text prompt
  const [imageSrc, setImageSrc] = useState<string | null>(null); // Base64 Data URL of the image for display/masking

  // Masking state (from SAM2)
  const [points, setPoints] = useState<Point[]>([]); // Current points for next mask
  const [savedMasks, setSavedMasks] = useState<SavedMask[]>([]); // List of saved masks

  // 3D Model parameters state (from heightmap_to_3d)
  const [blockWidth, setBlockWidth] = useState<number>(100);
  const [blockLength, setBlockLength] = useState<number>(100);
  const [blockThickness, setBlockThickness] = useState<number>(10);
  const [depth, setDepth] = useState<number>(25);
  const [baseHeight, setBaseHeight] = useState<number>(0);
  const [mode, setMode] = useState<"protrude" | "carve">("protrude");
  const [invert, setInvert] = useState<boolean>(false);
  const [includeColor, setIncludeColor] = useState<boolean>(false); // Determines output format (STL vs PLY)

  // Result state
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [resultFileType, setResultFileType] = useState<string | null>(null);

  // UI Flow and Loading/Error state
  type Step = "input" | "masking" | "params" | "generating" | "result";
  const [currentStep, setCurrentStep] = useState<Step>("input");
  const [isLoadingMask, setIsLoadingMask] = useState<boolean>(false); // Loading for mask prediction
  const [isLoadingModel, setIsLoadingModel] = useState<boolean>(false); // Loading for 3D generation / text-to-image
  const [error, setError] = useState<string | null>(null); // Combined error state

  // Refs
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null); // Ref to the loaded image element for dimensions/drawing
  const originalDimensionsRef = useRef<{ width: number; height: number } | null>(null);

  // --- Image Handling Callbacks ---

  // Clear relevant state when input changes significantly
  const resetForNewImage = () => {
    setImageSrc(null);
    setPoints([]);
    setSavedMasks([]);
    setResultUrl(null);
    setResultFileType(null);
    setError(null);
    imageRef.current = null;
    originalDimensionsRef.current = null;
    setCurrentStep("input"); // Go back to input step
  };

  // Handle file selection
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      resetForNewImage(); // Clear previous state
      setImageFile(file); // Store the file object
      setInputMethod("upload"); // Ensure input method is correct

      const reader = new FileReader();
      reader.onload = (e) => {
        const result = e.target?.result as string;
        const img = new Image();
        img.onload = () => {
          imageRef.current = img;
          originalDimensionsRef.current = { width: img.width, height: img.height };
          setImageSrc(result); // Set Base64 source for canvas
          setCurrentStep("masking"); // Move to masking step
        };
        img.onerror = () => {
          console.error("Failed to load image.");
          setError("Failed to load the selected image file.");
          resetForNewImage();
        };
        img.src = result;
      };
      reader.onerror = () => {
        console.error("Failed to read file.");
        setError("Failed to read the selected file.");
        resetForNewImage();
      };
      reader.readAsDataURL(file); // Read as Data URL for imageSrc
    } else {
      // Handle case where user cancels file selection
      // If imageFile was already set, don't reset unless necessary
      if (!imageFile) {
        resetForNewImage();
      }
    }
  };

  // Handle text prompt input change
  const handlePromptChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setTextPrompt(e.target.value);
    // Optionally clear file if text is entered and method is upload
    // if (inputMethod === 'upload') {
    //   setImageFile(null);
    // }
  };

  // Handle generating image from text prompt
  const handleGenerateImageFromText = async () => {
    if (!textPrompt.trim()) {
      setError("Please enter a text prompt.");
      return;
    }
    resetForNewImage(); // Clear previous state
    setIsLoadingModel(true); // Use model loading indicator
    setError(null);
    setCurrentStep("generating"); // Indicate generation process

    try {
      // Assume backend returns JSON with base64 image data: { image_b64: "..." }
      const response = await axios.post(GENERATE_IMAGE_URL, {
        prompt: textPrompt.trim(),
      });

      if (response.data && response.data.image_b64) {
        const base64ImageData = response.data.image_b64;
        const fullImageSrc = `data:image/png;base64,${base64ImageData}`; // Assuming PNG

        const img = new Image();
        img.onload = () => {
          imageRef.current = img;
          originalDimensionsRef.current = { width: img.width, height: img.height };
          setImageSrc(fullImageSrc); // Set Base64 source for canvas
          // Convert Base64 back to a File object if needed for final submission consistency
          const blob = base64ToBlob(base64ImageData, 'image/png');
          const generatedFile = new File([blob], "generated_image.png", { type: 'image/png' });
          setImageFile(generatedFile); // Store the generated image as a File object
          setCurrentStep("masking"); // Move to masking step
        };
        img.onerror = () => {
          setError("Failed to load the generated image.");
          resetForNewImage(); // Reset state on error
        };
        img.src = fullImageSrc;

      } else {
        throw new Error("Backend did not return valid image data.");
      }
    } catch (err: any) {
      console.error("Error generating image from text:", err);
      setError(err.response?.data?.error || err.message || "Failed to generate image from text.");
      resetForNewImage(); // Reset state on error
    } finally {
      setIsLoadingModel(false);
    }
  };


  // --- Mask Loading Effect (from SAM2) ---
  useEffect(() => {
    savedMasks.forEach((mask) => {
      // If mask has data but isn't loaded yet
      if (mask.maskB64Png && !mask.loadedImage) {
        const maskImg = new Image();
        maskImg.onload = () => {
          // Update the specific mask object in the array immutably
          setSavedMasks(currentMasks =>
            currentMasks.map(m =>
              m.id === mask.id ? { ...m, loadedImage: maskImg } : m
            )
          );
        };
        maskImg.onerror = () => {
          console.error(`Failed to load mask image ${mask.id} from Base64.`);
          // Optionally update the specific mask state to indicate loading error
          setError(prev => prev ? `${prev}\nFailed to load saved mask ${mask.id}.` : `Failed to load saved mask ${mask.id}.`);
        };
        // Ensure the src is a valid Data URL
        maskImg.src = mask.maskB64Png.startsWith('data:image')
          ? mask.maskB64Png
          : `data:image/png;base64,${mask.maskB64Png}`;
      }
    });
  }, [savedMasks]); // Rerun when savedMasks array changes


  // --- Canvas Drawing Logic ---
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    const mainImg = imageRef.current;

    if (!canvas || !ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!mainImg || !imageSrc) {
      // Optionally draw a placeholder if imageSrc is null but canvas exists
      ctx.fillStyle = '#f0f0f0'; // Light grey background
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#a0a0a0';
      ctx.font = '16px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No image loaded', canvas.width / 2, canvas.height / 2);
      return;
    };

    // Calculate scaling to fit image within canvas while maintaining aspect ratio
    const canvasWidth = canvas.width;
    const canvasHeight = canvas.height;
    const imgWidth = mainImg.naturalWidth; // Use naturalWidth for original size
    const imgHeight = mainImg.naturalHeight;

    if (imgWidth === 0 || imgHeight === 0) return; // Avoid division by zero if image not loaded properly

    const scale = Math.min(canvasWidth / imgWidth, canvasHeight / imgHeight);
    const scaledWidth = imgWidth * scale;
    const scaledHeight = imgHeight * scale;

    // Center the image on the canvas
    const offsetX = (canvasWidth - scaledWidth) / 2;
    const offsetY = (canvasHeight - scaledHeight) / 2;

    // 1. Draw the main image
    ctx.drawImage(mainImg, offsetX, offsetY, scaledWidth, scaledHeight);

    // 2. Draw all *active* saved masks
    ctx.globalAlpha = 0.55; // Semi-transparent masks
    savedMasks.forEach(mask => {
      if (mask.isActive && mask.loadedImage) {
        // Ensure mask image dimensions match the main image for correct overlay
        ctx.drawImage(mask.loadedImage, offsetX, offsetY, scaledWidth, scaledHeight);
      }
    });
    ctx.globalAlpha = 1.0; // Reset global alpha

    // 3. Draw the *current* points being selected (for the next mask)
    ctx.fillStyle = 'rgba(0, 255, 255, 0.9)'; // Cyan with slight transparency
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.9)'; // Black outline
    ctx.lineWidth = 1.5;
    points.forEach(point => {
      // Convert original image coordinates back to canvas coordinates
      const canvasX = offsetX + point.x * scale;
      const canvasY = offsetY + point.y * scale;
      ctx.beginPath();
      ctx.arc(canvasX, canvasY, 5, 0, 2 * Math.PI); // Point radius
      ctx.fill();
      ctx.stroke();
    });

  }, [points, savedMasks, imageSrc]); // Depend on points, masks, and image source

  // Effect to redraw canvas when image, points, or masks change
  useEffect(() => {
    // Only draw if we are in a step where the canvas should be visible
    if (currentStep === 'masking' || currentStep === 'params') {
      drawCanvas();
    }
  }, [imageSrc, points, savedMasks, drawCanvas, currentStep]);

  // Effect for handling canvas resizing and initial draw
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || currentStep !== 'masking') return; // Only run if canvas exists and is relevant

    let animationFrameId: number | null = null;

    // Use ResizeObserver to detect parent container size changes
    const resizeObserver = new ResizeObserver(entries => {
      if (!animationFrameId) {
        animationFrameId = requestAnimationFrame(() => {
          for (const entry of entries) {
            const { width, height } = entry.contentRect;
            // Check if canvas dimensions actually changed to avoid redundant redraws
            if (canvas.width !== Math.round(width) || canvas.height !== Math.round(height)) {
              // Update canvas resolution to match display size for crispness
              canvas.width = Math.round(width);
              canvas.height = Math.round(height);
              // Redraw with new dimensions
              drawCanvas();
            }
          }
          animationFrameId = null;
        });
      }
    });

    // Observe the canvas's parent element for size changes
    const parentElement = canvas.parentElement;
    if (parentElement) {
      resizeObserver.observe(parentElement);
      // Initial size setting and draw
      const { width, height } = parentElement.getBoundingClientRect();
      if (canvas.width !== Math.round(width) || canvas.height !== Math.round(height)) {
        canvas.width = Math.round(width);
        canvas.height = Math.round(height);
        // Use setTimeout to ensure initial draw happens after potential layout shifts
        const initialDrawTimeout = setTimeout(drawCanvas, 50);
        return () => clearTimeout(initialDrawTimeout); // Clear timeout on cleanup
      } else {
        const initialDrawTimeout = setTimeout(drawCanvas, 50);
        return () => clearTimeout(initialDrawTimeout);
      }
    }


    // Cleanup function
    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
      if (parentElement) {
        resizeObserver.unobserve(parentElement);
      }
    };
  }, [drawCanvas, currentStep]); // Rerun when drawCanvas changes or step becomes 'masking'


  // --- Point Selection ---
  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const img = imageRef.current; // Use the ref containing the loaded Image object
    // No need for originalDimensionsRef here if we use img.naturalWidth/Height

    if (isLoadingMask || isLoadingModel || !canvas || !img || img.naturalWidth === 0) return; // Don't allow clicks while loading or if image invalid

    const rect = canvas.getBoundingClientRect(); // Get canvas position on screen
    const clickX = event.clientX - rect.left;
    const clickY = event.clientY - rect.top;

    // Calculate image scaling and offset *at the time of click*
    const canvasWidth = canvas.width;
    const canvasHeight = canvas.height;
    const imgWidth = img.naturalWidth;
    const imgHeight = img.naturalHeight;
    const scale = Math.min(canvasWidth / imgWidth, canvasHeight / imgHeight);
    const scaledWidth = imgWidth * scale;
    const scaledHeight = imgHeight * scale;
    const offsetX = (canvasWidth - scaledWidth) / 2;
    const offsetY = (canvasHeight - scaledHeight) / 2;

    // Check if the click is within the bounds of the *displayed* image
    if (clickX >= offsetX && clickX <= offsetX + scaledWidth &&
      clickY >= offsetY && clickY <= offsetY + scaledHeight) {

      // Convert canvas click coordinates back to *original* image coordinates
      const originalX = (clickX - offsetX) / scale;
      const originalY = (clickY - offsetY) / scale;

      // Add the point (using original image coordinates) to the current selection
      setPoints(prevPoints => [...prevPoints, { x: originalX, y: originalY }]);
      setError(null); // Clear any previous errors
    }
  };

  // --- Mask Prediction ---
  const getMaskFromBackend = async () => {
    if (!imageSrc || points.length === 0) {
      setError("Please select at least one point on the image to predict a mask.");
      return;
    }
    setIsLoadingMask(true);
    setError(null);

    try {
      // Ensure imageSrc is just the Base64 data
      const base64Image = imageSrc.includes(',') ? imageSrc.split(',')[1] : imageSrc;

      const response = await axios.post(PREDICT_MASK_URL, {
        image: base64Image, // Send Base64 image data
        points: points, // Send the current points (in original image coordinates)
      });

      // Assuming backend returns: { status: 'success', mask_b64png: '...', score: 0.95 }
      if (response.data && response.data.status === 'success' && response.data.mask_b64png) {
        if (typeof response.data.mask_b64png === 'string') {
          // Create a new mask object and add it to the saved list
          const newMask: SavedMask = {
            id: uuidv4(), // Generate unique ID
            name: `Mask ${savedMasks.length + 1}`, // Default name
            maskB64Png: response.data.mask_b64png, // Store raw base64
            points: [...points], // Store points used for this mask
            isActive: true, // Make it active by default
            loadedImage: null, // Will be loaded by useEffect
            score: response.data.score,
          };
          setSavedMasks(prevMasks => [...prevMasks, newMask]);
          setPoints([]); // Clear points, ready for next selection
          console.log(`Saved new mask ${newMask.id}`);
        } else {
          throw new Error('Received mask data is not in the expected format.');
        }
      } else {
        throw new Error(response.data.message || 'Mask prediction failed on the backend.');
      }
    } catch (err: any) {
      console.error("Error predicting mask:", err);
      setError(err.response?.data?.error || err.message || "An unknown error occurred during mask prediction.");
      // Don't clear saved masks on error, but maybe clear current points?
      // setPoints([]);
    } finally {
      setIsLoadingMask(false);
    }
  };

  // --- Mask Management Callbacks ---

  // Toggle the active state of a specific saved mask
  const handleToggleMaskActive = (id: string) => {
    setSavedMasks(prevMasks =>
      prevMasks.map(mask =>
        mask.id === id ? { ...mask, isActive: !mask.isActive } : mask
      )
    );
  };

  // Handle renaming a mask
  const handleRenameMask = (id: string, newName: string) => {
    setSavedMasks(prevMasks =>
      prevMasks.map(mask =>
        mask.id === id ? { ...mask, name: newName || `Mask ${prevMasks.findIndex(m => m.id === id) + 1}` } : mask // Use default if name is empty
      )
    );
  };

  // Delete a specific mask
  const handleDeleteMask = (id: string) => {
    setSavedMasks(prevMasks => prevMasks.filter(mask => mask.id !== id));
  };


  // Resets only the currently selected points
  const handleResetCurrentPoints = () => {
    setPoints([]);
    setError(null); // Clear point-related errors
  };

  // Clears all saved masks
  const handleClearSavedMasks = () => {
    setSavedMasks([]);
    setPoints([]); // Also clear points that might have been selected
    setError(null);
  };

  // Downloads all *active* masks as individual PNGs
  const handleDownloadActiveMasks = () => {
    const activeMasks = savedMasks.filter(mask => mask.isActive);
    if (activeMasks.length === 0) {
      setError("No active masks selected to download.");
      return;
    }
    setError(null);

    activeMasks.forEach((mask) => {
      const link = document.createElement('a');
      // Ensure the href is a valid Data URL
      link.href = mask.maskB64Png.startsWith('data:image')
        ? mask.maskB64Png
        : `data:image/png;base64,${mask.maskB64Png}`;
      // Use mask name for filename, sanitize it
      const safeName = mask.name.replace(/[^a-z0-9]/gi, '_').toLowerCase();
      link.download = `mask_${safeName}_${mask.id.substring(0, 6)}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });
  };


  // --- 3D Model Generation ---
  const handleGenerate3DModel = async () => {
    setError(""); // Clear previous errors
    setIsLoadingModel(true);
    setResultUrl(null);
    setResultFileType(null);
    setCurrentStep("generating"); // Indicate generation process

    // --- Validation ---
    if (inputMethod === "upload" && !imageFile) {
      setError("Source image file is missing. Please re-upload.");
      setIsLoadingModel(false);
      setCurrentStep("input"); // Go back if image is lost
      return;
    }
    if (inputMethod === "text" && !imageFile) { // Check imageFile even for text prompt (should be set after generation)
      setError("Source image from text prompt is missing. Please regenerate.");
      setIsLoadingModel(false);
      setCurrentStep("input"); // Go back if image is lost
      return;
    }
    // --- End Validation ---

    // Build the form data
    const formData = new FormData();

    // --- Append Image Source ---
    // Always append the imageFile, which should exist for both upload and text-gen paths
    if (imageFile) {
      formData.append("color_image", imageFile);
    } else {
      // This case should ideally be caught by validation, but as a fallback:
      setError("Critical error: Image source not available for generation.");
      setIsLoadingModel(false);
      setCurrentStep("input");
      return;
    }
    // Note: We don't need to send the text prompt again if the image was generated from it,
    // the backend uses the provided image.

    // --- Append Active Masks ---
    const activeMasks = savedMasks.filter(m => m.isActive);
    activeMasks.forEach((mask, index) => {
      try {
        // Convert Base64 mask data to a Blob/File and append
        const blob = base64ToBlob(mask.maskB64Png, 'image/png');
        const safeName = mask.name.replace(/[^a-z0-9]/gi, '_').toLowerCase();
        // Backend needs to know how to interpret these files (e.g., by filename pattern)
        formData.append(`mask_${index}`, blob, `mask_${safeName}_${mask.id.substring(0, 6)}.png`);
      } catch (e) {
        console.error(`Failed to process mask ${mask.id} for upload:`, e);
        setError(`Error processing mask "${mask.name}" for upload. Please try removing it or generating again.`);
        setIsLoadingModel(false);
        setCurrentStep("params"); // Go back to params step
        return; // Stop the process if a mask fails
      }
    });
    // Optionally, send mask metadata (like names or points) as JSON string if backend needs it
    const maskMetadata = activeMasks.map(m => ({ id: m.id, name: m.name /*, points: m.points */ }));
    formData.append("mask_metadata", JSON.stringify(maskMetadata));


    // --- Append Model Parameters ---
    formData.append("block_width", String(blockWidth));
    formData.append("block_length", String(blockLength));
    formData.append("block_thickness", String(blockThickness));
    formData.append("depth", String(depth));
    formData.append("base_height", String(baseHeight));
    formData.append("mode", mode);
    formData.append("invert", String(invert));
    // The backend should infer file type based on include_color
    formData.append("include_color", String(includeColor));
    // --- End Append Model Parameters ---

    try {
      // Post to the final generation endpoint
      const response = await axios.post(GENERATE_MODEL_URL, formData, {
        headers: {
          // Content-Type is set automatically by browser for FormData
          // 'Content-Type': 'multipart/form-data', // Don't set manually with FormData
        },
        // TODO: Consider add a timeout for long generations
        // timeout: 300000, // e.g., 5 minutes
      });

      // Assuming backend returns { fileUrl: "...", fileType: "stl" | "ply" }
      if (response.data && response.data.fileUrl && response.data.fileType) {
        setResultUrl(response.data.fileUrl);
        setResultFileType(response.data.fileType);
        setCurrentStep("result"); // Move to result step
      } else {
        throw new Error("Invalid response received from model generation endpoint.");
      }
    } catch (err: any) {
      console.error("Error generating 3D model:", err);
      // Display user-friendly error from backend if available
      setError(err.response?.data?.error || err.message || "An unexpected error occurred during 3D model generation.");
      setCurrentStep("params"); // Go back to params step on error
    } finally {
      setIsLoadingModel(false);
    }
  };

  // --- Navigation Functions ---
  const goToMasking = () => {
    if (inputMethod === 'text' && !imageSrc) {
      handleGenerateImageFromText(); // Generate image first if needed
    } else if (imageSrc) {
      setCurrentStep("masking");
    } else {
      setError("Please upload an image or provide a text prompt first.");
    }
  };

  const goToParams = () => {
    if (!imageSrc) {
      setError("Cannot proceed without an image.");
      setCurrentStep("input");
    } else {
      setCurrentStep("params");
    }
  };


  // --- Render Logic ---
  return (
    <div className="max-w-6xl mx-auto p-4 md:p-6 text-gray-900 dark:text-gray-100 font-sans">
      {/* Page Header */}
      <header className="text-center mb-6 md:mb-8">
        <h1 className="text-3xl md:text-4xl font-bold">Text2Texture</h1>
        <p className="text-md md:text-lg text-gray-600 dark:text-gray-400 mt-2">
          Generate 3D models from images or text prompts.
        </p>
      </header>

      {/* Main Content Grid */}
      <main className="grid grid-cols-1 lg:grid-cols-2 gap-6 md:gap-8">

        {/* Left Column: Input, Masking, Parameters */}
        <div className="space-y-6 md:space-y-8">

          {/* === Step 1: Input === */}
          <section id="input-section" className="bg-white dark:bg-gray-800 p-4 md:p-6 rounded-lg shadow-md">
            <h2 className="text-xl md:text-2xl font-semibold mb-4 border-b pb-2 dark:border-gray-600">1. Input Image</h2>

            {/* Input Method Selection */}
            <fieldset className="mb-4">
              <legend className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-300">
                Choose Input Method:
              </legend>
              <div className="flex items-center space-x-6">
                {/* Upload Option */}
                <div className="flex items-center">
                  <input
                    type="radio" id="uploadMethod" name="inputMethod" value="upload"
                    checked={inputMethod === "upload"}
                    onChange={(e) => { setInputMethod(e.target.value as "upload" | "text"); /* resetForNewImage(); */ }} // Resetting here might be too aggressive
                    className="h-4 w-4 text-blue-600 border-gray-300 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700"
                    disabled={isLoadingModel || isLoadingMask}
                  />
                  <label htmlFor="uploadMethod" className="ml-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
                    Upload Image
                  </label>
                </div>
                {/* Text Prompt Option */}
                <div className="flex items-center">
                  <input
                    type="radio" id="textMethod" name="inputMethod" value="text"
                    checked={inputMethod === "text"}
                    onChange={(e) => { setInputMethod(e.target.value as "upload" | "text"); /* resetForNewImage(); */ }}
                    className="h-4 w-4 text-blue-600 border-gray-300 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700"
                    disabled={isLoadingModel || isLoadingMask}
                  />
                  <label htmlFor="textMethod" className="ml-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
                    Generate from Text
                  </label>
                </div>
              </div>
            </fieldset>

            {/* Conditional Inputs */}
            {inputMethod === "upload" && (
              <div>
                <label htmlFor="imageUpload" className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                  Select Image File:
                </label>
                <input
                  type="file" id="imageUpload" accept="image/*"
                  onChange={handleFileChange}
                  className="mt-1 block w-full text-sm file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 dark:file:bg-gray-600 dark:file:text-gray-200 dark:hover:file:bg-gray-500 text-gray-900 dark:text-gray-300 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                  disabled={isLoadingModel || isLoadingMask}
                />
                {imageFile && <p className="text-xs mt-1 text-gray-500 dark:text-gray-400">Selected: {imageFile.name}</p>}
              </div>
            )}

            {inputMethod === "text" && (
              <div>
                <label htmlFor="textPromptInput" className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                  Enter Text Prompt:
                </label>
                <textarea
                  id="textPromptInput" value={textPrompt} onChange={handlePromptChange}
                  placeholder="e.g., A detailed topographic map of a fantasy island"
                  rows={3}
                  className="mt-1 block w-full p-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white disabled:opacity-50"
                  disabled={isLoadingModel || isLoadingMask}
                />
                <button
                  onClick={handleGenerateImageFromText}
                  disabled={isLoadingModel || isLoadingMask || !textPrompt.trim()}
                  className="mt-2 w-full sm:w-auto px-4 py-2 bg-indigo-600 text-white font-semibold rounded-lg shadow hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-opacity-50 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isLoadingModel && currentStep === 'generating' ? 'Generating Image...' : 'Generate Image'}
                </button>
              </div>
            )}
          </section>


          {/* === Step 2: Mask Selection (Conditional) === */}
          {currentStep === 'masking' && imageSrc && (
            <section id="masking-section" className="bg-white dark:bg-gray-800 p-4 md:p-6 rounded-lg shadow-md">
              <h2 className="text-xl md:text-2xl font-semibold mb-4 border-b pb-2 dark:border-gray-600">2. Select Masks (Optional)</h2>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">Click on the image to place points. Use 'Predict & Save Mask' to generate a mask from the points. You can create multiple masks.</p>

              {/* Canvas Area for Masking */}
              <div className="w-full aspect-[4/3] bg-gray-200 dark:bg-gray-700 rounded-lg shadow overflow-hidden relative mb-4 border dark:border-gray-600">
                <canvas
                  ref={canvasRef}
                  onClick={handleCanvasClick}
                  className={`w-full h-full block ${isLoadingMask ? 'cursor-wait' : 'cursor-crosshair'} ${(isLoadingModel || isLoadingMask) ? 'opacity-50' : ''}`}
                  style={{ imageRendering: 'pixelated' }} // Optional: for sharper pixels if needed
                />
                {isLoadingMask && (
                  <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center z-10">
                    <p className="text-white text-lg font-semibold animate-pulse">Predicting Mask...</p>
                  </div>
                )}
              </div>

              {/* Display Current Points */}
              {points.length > 0 && (
                <div className="mb-3 p-2 bg-blue-50 dark:bg-gray-700 border border-blue-200 dark:border-blue-800 rounded text-sm">
                  <p className="text-blue-800 dark:text-blue-200">
                    <span className="font-semibold">{points.length}</span> points selected for next mask.
                  </p>
                </div>
              )}

              {/* Masking Buttons */}
              <div className="flex flex-wrap justify-start gap-2 mb-4">
                <button
                  onClick={getMaskFromBackend}
                  disabled={isLoadingMask || isLoadingModel || points.length === 0}
                  className="px-4 py-2 bg-blue-600 text-white text-sm font-semibold rounded-lg shadow hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Predict & Save Mask
                </button>
                <button
                  onClick={handleResetCurrentPoints}
                  disabled={isLoadingMask || isLoadingModel || points.length === 0}
                  className="px-4 py-2 bg-yellow-500 text-white text-sm font-semibold rounded-lg shadow hover:bg-yellow-600 focus:outline-none focus:ring-2 focus:ring-yellow-400 focus:ring-opacity-50 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Reset Points
                </button>
              </div>

              {/* Saved Masks List */}
              {savedMasks.length > 0 && (
                <div className="mt-4 border-t pt-4 dark:border-gray-600">
                  <h3 className="text-lg font-semibold mb-2 text-gray-700 dark:text-gray-200">
                    Saved Masks ({savedMasks.filter(m => m.isActive).length} active)
                  </h3>
                  <ul className="space-y-2 max-h-48 overflow-y-auto pr-2 border rounded-md p-2 dark:border-gray-600">
                    {savedMasks.map((mask, index) => (
                      <li key={mask.id} className="flex items-center justify-between p-2 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700">
                        <input
                          type="text"
                          value={mask.name}
                          onChange={(e) => handleRenameMask(mask.id, e.target.value)}
                          onBlur={(e) => handleRenameMask(mask.id, e.target.value)} // Save on blur too
                          className="text-sm text-gray-800 dark:text-gray-200 bg-transparent border-b border-gray-300 dark:border-gray-500 focus:outline-none focus:border-blue-500 mr-2 flex-grow"
                          placeholder={`Mask ${index + 1}`}
                          disabled={isLoadingMask || isLoadingModel}
                        />
                        <span className="text-xs text-gray-500 dark:text-gray-400 mr-2">
                          (Score: {mask.score?.toFixed(2) ?? 'N/A'})
                        </span>
                        <div className="flex items-center space-x-3 flex-shrink-0">
                          <label className="flex items-center space-x-1 cursor-pointer" title="Toggle visibility/usage">
                            <input
                              type="checkbox"
                              checked={mask.isActive}
                              onChange={() => handleToggleMaskActive(mask.id)}
                              className="form-checkbox h-4 w-4 text-blue-600 rounded border-gray-300 focus:ring-blue-500 disabled:opacity-50"
                              disabled={isLoadingMask || isLoadingModel}
                            />
                            <span className="text-xs text-gray-700 dark:text-gray-300">Use</span>
                          </label>
                          <button
                            onClick={() => handleDeleteMask(mask.id)}
                            title="Delete Mask"
                            className="text-red-500 hover:text-red-700 disabled:opacity-50"
                            disabled={isLoadingMask || isLoadingModel}
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                              <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                            </svg>
                          </button>
                        </div>
                      </li>
                    ))}
                  </ul>
                  <div className="flex flex-wrap justify-start gap-2 mt-3">
                    <button
                      onClick={handleDownloadActiveMasks}
                      disabled={isLoadingMask || isLoadingModel || savedMasks.filter(m => m.isActive).length === 0}
                      className="px-3 py-1 bg-teal-600 text-white text-xs font-semibold rounded-lg shadow hover:bg-teal-700 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:ring-opacity-50 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Download Active Masks
                    </button>
                    <button
                      onClick={handleClearSavedMasks}
                      disabled={isLoadingMask || isLoadingModel || savedMasks.length === 0}
                      className="px-3 py-1 bg-red-600 text-white text-xs font-semibold rounded-lg shadow hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-opacity-50 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Clear All Masks
                    </button>
                  </div>
                </div>
              )}

              {/* Navigation Button */}
              <div className="mt-6 text-center">
                <button
                  onClick={goToParams}
                  disabled={isLoadingMask || isLoadingModel}
                  className="px-6 py-2 bg-green-600 text-white font-semibold rounded-lg shadow hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-50 disabled:opacity-50"
                >
                  Proceed to 3D Parameters &raquo;
                </button>
              </div>

            </section>
          )}


          {/* === Step 3: 3D Parameters (Conditional) === */}
          {currentStep === 'params' && imageSrc && (
            <section id="params-section" className="bg-white dark:bg-gray-800 p-4 md:p-6 rounded-lg shadow-md">
              <h2 className="text-xl md:text-2xl font-semibold mb-4 border-b pb-2 dark:border-gray-600">3. Configure 3D Model</h2>

              {/* Back Button */}
              <button onClick={() => setCurrentStep('masking')} className="text-sm text-blue-600 hover:underline mb-4 dark:text-blue-400">&laquo; Back to Mask Selection</button>

              {/* Parameter Form Fields */}
              <div className="space-y-4">
                {/* Two-column grid for numeric inputs */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {/* Block Width */}
                  <div>
                    <label htmlFor="blockWidth" className="block text-sm font-medium text-gray-700 dark:text-gray-200">
                      Block Width (mm)
                    </label>
                    <input
                      id="blockWidth" type="number" value={blockWidth} min="1" step="1"
                      onChange={(e) => setBlockWidth(Number(e.target.value))}
                      className="mt-1 block w-full p-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white disabled:opacity-50"
                      disabled={isLoadingModel}
                    />
                  </div>
                  {/* Block Length */}
                  <div>
                    <label htmlFor="blockLength" className="block text-sm font-medium text-gray-700 dark:text-gray-200">
                      Block Length (mm)
                    </label>
                    <input
                      id="blockLength" type="number" value={blockLength} min="1" step="1"
                      onChange={(e) => setBlockLength(Number(e.target.value))}
                      className="mt-1 block w-full p-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white disabled:opacity-50"
                      disabled={isLoadingModel}
                    />
                  </div>
                  {/* Block Thickness */}
                  <div>
                    <label htmlFor="blockThickness" className="block text-sm font-medium text-gray-700 dark:text-gray-200">
                      Block Thickness (mm)
                    </label>
                    <input
                      id="blockThickness" type="number" value={blockThickness} min="0.1" step="0.1"
                      onChange={(e) => setBlockThickness(Number(e.target.value))}
                      className="mt-1 block w-full p-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white disabled:opacity-50"
                      disabled={isLoadingModel}
                    />
                  </div>
                  {/* Depth */}
                  <div>
                    <label htmlFor="depth" className="block text-sm font-medium text-gray-700 dark:text-gray-200">
                      Max Depth/Height (mm)
                    </label>
                    <input
                      id="depth" type="number" value={depth} min="0.1" step="0.1"
                      onChange={(e) => setDepth(Number(e.target.value))}
                      className="mt-1 block w-full p-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white disabled:opacity-50"
                      disabled={isLoadingModel}
                    />
                  </div>
                  {/* Base Height */}
                  <div>
                    <label htmlFor="baseHeight" className="block text-sm font-medium text-gray-700 dark:text-gray-200">
                      Base Height (mm)
                    </label>
                    <input
                      id="baseHeight" type="number" value={baseHeight} min="0" step="0.1"
                      onChange={(e) => setBaseHeight(Number(e.target.value))}
                      className="mt-1 block w-full p-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white disabled:opacity-50"
                      disabled={isLoadingModel}
                    />
                  </div>
                  {/* Mode */}
                  <div>
                    <label htmlFor="mode" className="block text-sm font-medium text-gray-700 dark:text-gray-200">
                      Mode
                    </label>
                    <select
                      id="mode" value={mode}
                      onChange={(e) => setMode(e.target.value as "protrude" | "carve")}
                      className="mt-1 block w-full p-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white disabled:opacity-50"
                      disabled={isLoadingModel}
                    >
                      <option value="protrude">Protrude (lighter=higher)</option>
                      <option value="carve">Carve (lighter=lower)</option>
                    </select>
                  </div>
                </div>

                {/* Checkboxes */}
                <div className="flex flex-col sm:flex-row justify-between space-y-2 sm:space-y-0 sm:space-x-4 pt-2">
                  {/* Invert */}
                  <div className="flex items-center">
                    <input
                      id="invert" type="checkbox" checked={invert}
                      onChange={(e) => setInvert(e.target.checked)}
                      className="h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500 dark:border-gray-500 dark:bg-gray-600 disabled:opacity-50"
                      disabled={isLoadingModel}
                    />
                    <label htmlFor="invert" className="ml-2 text-sm font-medium text-gray-700 dark:text-gray-300">
                      Invert Heightmap (darker = higher/lower based on mode)
                    </label>
                  </div>
                  {/* Include Color */}
                  <div className="flex items-center">
                    <input
                      id="includeColor" type="checkbox" checked={includeColor}
                      onChange={(e) => setIncludeColor(e.target.checked)}
                      className="h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500 dark:border-gray-500 dark:bg-gray-600 disabled:opacity-50"
                      disabled={isLoadingModel}
                    />
                    <label htmlFor="includeColor" className="ml-2 text-sm font-medium text-gray-700 dark:text-gray-300">
                      Include Color (generates .PLY instead of .STL)
                    </label>
                  </div>
                </div>
              </div>

              {/* Final Generate Button */}
              <div className="mt-6 pt-4 border-t dark:border-gray-600">
                <button
                  onClick={handleGenerate3DModel}
                  disabled={isLoadingModel}
                  className="w-full py-3 px-4 bg-green-600 hover:bg-green-700 text-white font-semibold rounded-md shadow-sm transition duration-150 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isLoadingModel && currentStep === 'generating' ? "Generating 3D Model..." : "Generate 3D Model"}
                </button>
              </div>
            </section>
          )}

          {/* Loading Indicator during Generation Steps */}
          {(isLoadingModel || isLoadingMask || currentStep === 'generating') && (
            <div className="flex items-center justify-center p-4 bg-gray-100 dark:bg-gray-700 rounded-lg shadow">
              <svg className="animate-spin h-5 w-5 mr-3 text-blue-600 dark:text-blue-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
              </svg>
              <span className="text-gray-700 dark:text-gray-300">
                {isLoadingMask ? "Predicting mask..." :
                  isLoadingModel && currentStep === 'generating' && inputMethod === 'text' ? "Generating image..." :
                    isLoadingModel && currentStep === 'generating' ? "Generating 3D model..." : "Processing..."}
                Please wait.
              </span>
            </div>
          )}

          {/* Error Display Area */}
          {error && (
            <div className="mt-4 p-3 bg-red-100 border border-red-400 text-red-700 dark:bg-red-900 dark:border-red-700 dark:text-red-200 rounded-lg shadow">
              <p className="font-medium">Error:</p>
              <p className="text-sm whitespace-pre-wrap">{error}</p>
              <button onClick={() => setError(null)} className="mt-2 text-xs font-semibold text-red-800 dark:text-red-300 hover:underline">Dismiss</button>
            </div>
          )}

        </div> {/* End Left Column */}


        {/* Right Column: 3D Viewer / Result */}
        <div className="lg:sticky lg:top-6 h-[50vh] lg:h-[calc(100vh-4rem)]"> {/* Make viewer taller and sticky on large screens */}
          <section id="viewer-section" className="bg-gray-100 dark:bg-gray-900 p-3 md:p-4 rounded-lg shadow-md h-full flex flex-col">
            <h2 className="text-lg md:text-xl font-semibold mb-3 text-gray-800 dark:text-gray-100 flex-shrink-0">
              {resultUrl ? 'Generated Model Preview' : '3D Preview Area'}
            </h2>

            {/* Loading/Generating State for Viewer */}
            {isLoadingModel && currentStep === 'generating' && (
              <div className="flex-grow flex items-center justify-center bg-gray-200 dark:bg-gray-800 rounded-md">
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
            {currentStep === 'result' && resultUrl && resultFileType && (
              <div className="flex-grow flex flex-col min-h-0"> {/* Ensure flex container takes height */}
                <div className="mb-3 flex-shrink-0">
                  <a
                    href={resultUrl}
                    download // Suggest download
                    className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 dark:focus:ring-offset-gray-800"
                  >
                    <svg className="-ml-1 mr-2 h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                      <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L10 11.586l2.293-2.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clipRule="evenodd" />
                      <path fillRule="evenodd" d="M10 3a1 1 0 011 1v6a1 1 0 11-2 0V4a1 1 0 011-1z" clipRule="evenodd" />
                    </svg>
                    Download Model ({resultFileType.toUpperCase()})
                  </a>
                  <button
                    onClick={resetForNewImage}
                    className="ml-3 inline-flex items-center px-4 py-2 border border-gray-300 dark:border-gray-600 text-sm font-medium rounded-md shadow-sm text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 dark:focus:ring-offset-gray-800"
                  >
                    Start Over
                  </button>
                </div>
                {/* Model Viewer Container */}
                <div className="flex-grow border border-gray-300 dark:border-gray-600 rounded-lg overflow-hidden bg-gray-200 dark:bg-gray-800 min-h-0"> {/* Ensure viewer takes remaining space */}
                  <ModelViewer fileUrl={resultUrl} fileType={resultFileType} />
                </div>
              </div>
            )}

            {/* Placeholder when no result and not loading */}
            {!isLoadingModel && currentStep !== 'result' && currentStep !== 'generating' && (
              <div className="flex-grow flex items-center justify-center bg-gray-200 dark:bg-gray-800 rounded-md text-center p-4">
                <p className="text-gray-500 dark:text-gray-400">
                  {currentStep === 'input' ? 'Upload or generate an image to begin.' :
                    currentStep === 'masking' ? 'Select masks or proceed to parameters.' :
                      currentStep === 'params' ? 'Configure parameters and click "Generate 3D Model".' :
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
