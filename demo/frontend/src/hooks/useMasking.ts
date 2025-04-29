import { useState, useEffect, useRef, useCallback, MouseEvent } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { Point, SavedMask, BoundingBox } from '../types/app.types';
import { predictMaskFromPrompt, generateMasksAutomatically } from '../services/api';

export type MaskingMode = 'point' | 'box';

export interface UseMaskingReturn {
  points: Point[];
  savedMasks: SavedMask[];
  selectedBox: BoundingBox | null; // Expose selected box
  maskingMode: MaskingMode; // Expose current mode
  isLoadingMask: boolean; // Prediction loading (points or box)
  isLoadingAutoMask: boolean; // Automatic generation loading
  maskingError: string | null; // Error state specific to masking
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  imageRef: React.RefObject<HTMLImageElement | null>;
  setPoints: React.Dispatch<React.SetStateAction<Point[]>>;
  setSavedMasks: React.Dispatch<React.SetStateAction<SavedMask[]>>;
  setMaskingMode: (mode: MaskingMode) => void; // Function to change mode
  // Canvas interaction handlers
  handleMouseDown: (event: MouseEvent<HTMLCanvasElement>) => void;
  handleMouseMove: (event: MouseEvent<HTMLCanvasElement>) => void;
  handleMouseUp: (event: MouseEvent<HTMLCanvasElement>) => void;
  // Prediction triggers
  triggerPrediction: () => Promise<void>;
  handleGenerateMasksAutomatically: (imageSrc: string | null) => Promise<void>;
  // Mask list management
  handleToggleMaskActive: (id: string) => void;
  handleRenameMask: (id: string, newName: string) => void;
  handleDeleteMask: (id: string) => void;
  // Reset/Clear actions
  handleResetCurrentPoints: () => void;
  handleClearSavedMasks: () => void;
  handleDownloadActiveMasks: () => void;
  clearMaskingError: () => void;
  // Expose drawCanvas if needed externally
  drawCanvas: () => void;
}

// Helper to convert canvas event coords to original image coords
const getCoordsFromEvent = (
  event: MouseEvent<HTMLCanvasElement>,
  canvas: HTMLCanvasElement,
  img: HTMLImageElement | null
): Point | null => {
  if (!img || img.naturalWidth === 0) return null;

  const rect = canvas.getBoundingClientRect();
  const clickX = event.clientX - rect.left;
  const clickY = event.clientY - rect.top;

  const canvasWidth = canvas.width;
  const canvasHeight = canvas.height;
  const imgWidth = img.naturalWidth;
  const imgHeight = img.naturalHeight;
  const scale = Math.min(canvasWidth / imgWidth, canvasHeight / imgHeight);
  const scaledWidth = imgWidth * scale;
  const scaledHeight = imgHeight * scale;
  const offsetX = (canvasWidth - scaledWidth) / 2;
  const offsetY = (canvasHeight - scaledHeight) / 2;

  // Check if click is within image bounds
  if (clickX < offsetX || clickX > offsetX + scaledWidth || clickY < offsetY || clickY > offsetY + scaledHeight) {
    return null; // Click outside image
  }

  // Convert canvas click coordinates back to *original* image coordinates
  const originalX = (clickX - offsetX) / scale;
  const originalY = (clickY - offsetY) / scale;

  return { x: originalX, y: originalY };
};


export function useMasking(): UseMaskingReturn {
  const [points, setPoints] = useState<Point[]>([]);
  const [savedMasks, setSavedMasks] = useState<SavedMask[]>([]);
  const [isLoadingMask, setIsLoadingMask] = useState<boolean>(false); // Used for both point/box prediction
  const [isLoadingAutoMask, setIsLoadingAutoMask] = useState<boolean>(false);
  const [maskingError, setMaskingError] = useState<string | null>(null);
  const [maskingMode, setMaskingModeInternal] = useState<MaskingMode>('point');
  const [isDrawingBox, setIsDrawingBox] = useState<boolean>(false);
  const [startPoint, setStartPoint] = useState<Point | null>(null); // Start point for box drawing (image coords)
  const [currentBox, setCurrentBox] = useState<BoundingBox | null>(null); // Box being drawn (image coords)
  const [selectedBox, setSelectedBox] = useState<BoundingBox | null>(null); // Finalized box (image coords)


  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);

  const clearMaskingError = useCallback(() => {
    setMaskingError(null);
  }, []);

  const setMaskingMode = useCallback((mode: MaskingMode) => {
    setMaskingModeInternal(mode);
    // Clear interactive elements when switching modes
    setPoints([]);
    setSelectedBox(null);
    setCurrentBox(null);
    setIsDrawingBox(false);
    setStartPoint(null);
    clearMaskingError();
  }, [clearMaskingError]);

  // --- Mask Image Loading Effect ---
  useEffect(() => {
    // Load images for newly added masks or masks without loaded images
    savedMasks.forEach((mask) => {
      if (mask.maskB64Png && !mask.loadedImage) {
        const maskImg = new Image();
        maskImg.onload = () => {
          // Update only the specific mask's loadedImage property
          setSavedMasks(currentMasks =>
            currentMasks.map(m =>
              m.id === mask.id ? { ...m, loadedImage: maskImg } : m
            )
          );
        };
        maskImg.onerror = () => {
          console.error(`Failed to load mask image ${mask.id} from Base64.`);
          setMaskingError(prev => (prev ? `${prev}\n` : '') + `Failed to load display for saved mask '${mask.name}'.`);
          // TODO: remove the broken mask or mark it as errored
          // setSavedMasks(currentMasks => currentMasks.filter(m => m.id !== mask.id));
        };
        // Ensure the src is a valid Data URL
        maskImg.src = mask.maskB64Png.startsWith('data:image')
          ? mask.maskB64Png
          : `data:image/png;base64,${mask.maskB64Png}`;
      }
    });
    // Cleanup: Revoke object URLs if mask images are removed? Not applicable here as we use Data URLs.
  }, [savedMasks]); // Rerun when savedMasks array changes identity

  // --- Canvas Drawing Logic ---
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    // Ensure canvas resolution matches its displayed size
    const rect = canvas.getBoundingClientRect();
    const w = Math.round(rect.width);
    const h = Math.round(rect.height);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    const ctx = canvas.getContext('2d');
    const mainImg = imageRef.current;

    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!mainImg || mainImg.naturalWidth === 0 || !mainImg.src) {
      // Draw placeholder if no valid image ref or src
      ctx.fillStyle = '#e0e0e0';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#808080';
      ctx.font = '16px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('Waiting for image...', canvas.width / 2, canvas.height / 2);
      return;
    };

    // Calculate scaling and offset based on the image element's natural dimensions
    const canvasWidth = canvas.width;
    const canvasHeight = canvas.height;
    const imgWidth = mainImg.naturalWidth;
    const imgHeight = mainImg.naturalHeight;
    const scale = Math.min(canvasWidth / imgWidth, canvasHeight / imgHeight);
    const scaledWidth = imgWidth * scale;
    const scaledHeight = imgHeight * scale;
    const offsetX = (canvasWidth - scaledWidth) / 2;
    const offsetY = (canvasHeight - scaledHeight) / 2;

    // Helper to convert image coords to canvas coords
    const toCanvasCoords = (imgX: number, imgY: number): Point => ({
      x: offsetX + imgX * scale,
      y: offsetY + imgY * scale,
    });

    // 1. Draw the main image
    ctx.drawImage(mainImg, offsetX, offsetY, scaledWidth, scaledHeight);

    // 2. Draw all *active* saved masks
    ctx.globalAlpha = 0.55; // Semi-transparent masks
    savedMasks.forEach(mask => {
      if (mask.isActive && mask.loadedImage) {
        ctx.drawImage(mask.loadedImage, offsetX, offsetY, scaledWidth, scaledHeight);
      }
    });
    ctx.globalAlpha = 1.0; // Reset global alpha

    // 3. Draw interaction elements based on mode
    if (maskingMode === 'point') {
      // Draw the *current* points being selected
      ctx.fillStyle = 'rgba(0, 255, 255, 0.9)'; // Cyan points
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.9)'; // Black outline
      ctx.lineWidth = 1.5;
      points.forEach(point => {
        const canvasPoint = toCanvasCoords(point.x, point.y);
        ctx.beginPath();
        ctx.arc(canvasPoint.x, canvasPoint.y, 5, 0, 2 * Math.PI); // Point radius
        ctx.fill();
        ctx.stroke();
      });
    } else if (maskingMode === 'box') {
      // Draw the box being drawn (currentBox)
      if (currentBox) {
        const [x1, y1, x2, y2] = currentBox;
        const startCanvas = toCanvasCoords(x1, y1);
        const endCanvas = toCanvasCoords(x2, y2);
        ctx.strokeStyle = 'rgba(255, 255, 0, 0.9)'; // Yellow dashed line
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.strokeRect(startCanvas.x, startCanvas.y, endCanvas.x - startCanvas.x, endCanvas.y - startCanvas.y);
        ctx.setLineDash([]); // Reset line dash
      }
      // Draw the finalized selected box
      if (selectedBox) {
        const [x1, y1, x2, y2] = selectedBox;
        const startCanvas = toCanvasCoords(x1, y1);
        const endCanvas = toCanvasCoords(x2, y2);
        ctx.strokeStyle = 'rgba(0, 255, 0, 0.9)'; // Green solid line
        ctx.lineWidth = 2;
        ctx.strokeRect(startCanvas.x, startCanvas.y, endCanvas.x - startCanvas.x, endCanvas.y - startCanvas.y);
      }
    }

  }, [points, savedMasks, maskingMode, currentBox, selectedBox]); // Dependencies

  // --- Canvas Interaction Handlers ---
  const handleMouseDown = useCallback((event: MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (maskingMode === 'point') {
      // Handle point click directly on mouse down for simplicity
      handleCanvasClick(event);
      return;
    }
    // Box mode logic
    if (!canvas || !img) return;
    const coords = getCoordsFromEvent(event, canvas, img);
    if (coords) {
      setIsDrawingBox(true);
      setStartPoint(coords);
      setCurrentBox(null); // Clear previous drawing box
      setSelectedBox(null); // Clear previous selection
      clearMaskingError();
    }
  }, [maskingMode, clearMaskingError]); // Removed handleCanvasClick dependency

  const handleMouseMove = useCallback((event: MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (maskingMode !== 'box' || !isDrawingBox || !startPoint || !canvas || !img) return;

    const currentCoords = getCoordsFromEvent(event, canvas, img);
    if (currentCoords) {
      // Ensure x1 < x2 and y1 < y2 for drawing rect
      const x1 = Math.min(startPoint.x, currentCoords.x);
      const y1 = Math.min(startPoint.y, currentCoords.y);
      const x2 = Math.max(startPoint.x, currentCoords.x);
      const y2 = Math.max(startPoint.y, currentCoords.y);
      setCurrentBox([x1, y1, x2, y2]);
    }
  }, [maskingMode, isDrawingBox, startPoint]);

  const handleMouseUp = useCallback((event: MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    // Only process if we were drawing a box
    if (maskingMode !== 'box' || !isDrawingBox || !startPoint || !canvas || !img) {
      setIsDrawingBox(false); // Ensure drawing state is reset even if conditions not met
      setStartPoint(null);
      return;
    }

    const finalCoords = getCoordsFromEvent(event, canvas, img);
    if (finalCoords) {
      // Ensure x1 < x2 and y1 < y2 for the final box
      const x1 = Math.min(startPoint.x, finalCoords.x);
      const y1 = Math.min(startPoint.y, finalCoords.y);
      const x2 = Math.max(startPoint.x, finalCoords.x);
      const y2 = Math.max(startPoint.y, finalCoords.y);

      // Prevent tiny boxes (optional threshold)
      if (Math.abs(x2 - x1) > 5 && Math.abs(y2 - y1) > 5) {
        setSelectedBox([x1, y1, x2, y2]);
      } else {
        setSelectedBox(null); // Box too small
        console.log("Selected box too small, clearing selection.");
      }
    } else {
      setSelectedBox(null); // Mouse up outside image
      console.log("Mouse up outside image, clearing selection.");
    }

    setIsDrawingBox(false);
    setStartPoint(null);
    setCurrentBox(null); // Clear the drawing feedback box
  }, [maskingMode, isDrawingBox, startPoint]);

  // Internal handler for point clicks (called from handleMouseDown)
  const handleCanvasClick = useCallback((event: MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    const isLoading = isLoadingMask || isLoadingAutoMask;

    // This function is only for adding points, called by handleMouseDown in point mode
    if (isLoading || !canvas || !img) return;

    const coords = getCoordsFromEvent(event, canvas, img);
    if (coords) {
      setPoints(prevPoints => [...prevPoints, coords]);
      clearMaskingError();
    }
  }, [isLoadingMask, isLoadingAutoMask, clearMaskingError]);


  // --- Mask Prediction (Triggered by UI) ---
  const triggerPrediction = useCallback(async () => {
    const img = imageRef.current;
    if (!img || !img.src) {
      setMaskingError("Please load an image first.");
      return;
    }

    const predictionPayload: { image: string; points?: Point[]; box?: BoundingBox } = { image: img.src };
    let canPredict = false;
    let namePrefix = "Mask";

    if (maskingMode === 'point') {
      if (points.length === 0) {
        setMaskingError("Please select at least one point to predict a mask.");
        return;
      }
      predictionPayload.points = points;
      canPredict = true;
      namePrefix = `Point Mask ${savedMasks.length + 1}`;
    } else if (maskingMode === 'box') {
      if (!selectedBox) {
        setMaskingError("Please draw a bounding box to predict a mask.");
        return;
      }
      predictionPayload.box = selectedBox;
      canPredict = true;
      namePrefix = `Box Mask ${savedMasks.length + 1}`;
    }

    if (!canPredict) return; // Should not happen if checks above are correct

    setIsLoadingMask(true);
    clearMaskingError();

    try {
      // Call the unified API service function
      const { maskB64Png, score } = await predictMaskFromPrompt(predictionPayload);

      const newMask: SavedMask = {
        id: uuidv4(),
        name: namePrefix,
        maskB64Png: maskB64Png,
        points: maskingMode === 'point' ? [...points] : [], // Store points only if point mode
        // box: maskingMode === 'box' ? selectedBox : undefined, // Optionally store the box
        isActive: true,
        loadedImage: null, // Will be loaded by useEffect
        score: score,
      };
      setSavedMasks(prevMasks => [...prevMasks, newMask]);

      // Clear the prompt used for this prediction
      if (maskingMode === 'point') {
        setPoints([]);
      } else {
        setSelectedBox(null);
      }
      console.log(`Saved new mask ${newMask.id} from ${maskingMode} prompt`);

    } catch (err) {
      console.error(`Error predicting mask from ${maskingMode}:`, err);
      const message = err instanceof Error ? err.message : String(err);
      setMaskingError(message || `An unknown error occurred during mask prediction (${maskingMode}).`);
    } finally {
      setIsLoadingMask(false);
    }
  }, [maskingMode, points, selectedBox, savedMasks.length, clearMaskingError]);

  // --- Automatic Mask Generation ---
  const handleGenerateMasksAutomatically = useCallback(async (imageSrc: string | null) => {
    if (!imageSrc) { // Check if imageSrc is provided
      setMaskingError("No image available for automatic mask generation.");
      return;
    }
    setIsLoadingAutoMask(true);
    clearMaskingError();
    setPoints([]); // Clear any manual points

    try {
      const autoMasksData = await generateMasksAutomatically(imageSrc); // Use provided imageSrc

      if (autoMasksData.length === 0) {
        setMaskingError("Automatic generation did not find any masks."); // Use error state
      } else {
        const newMasks: SavedMask[] = autoMasksData.map((maskData, index) => ({
          id: uuidv4(),
          name: maskData.label || `Auto Mask ${index + 1}`,
          maskB64Png: maskData.mask_b64png, // Assume backend sends with data URI prefix
          points: [],
          isActive: true,
          loadedImage: null, // Will be loaded by useEffect
          score: maskData.score,
        }));
        setSavedMasks(newMasks); // Replace existing masks
        console.log(`Automatically generated and saved ${newMasks.length} masks.`);
      }
    } catch (err) {
      console.error("Error generating masks automatically:", err);
      const message = err instanceof Error ? err.message : String(err);
      setMaskingError(message || "An unknown error occurred during automatic mask generation.");
    } finally {
      setIsLoadingAutoMask(false);
    }
  }, [clearMaskingError]); // Dependencies

  // --- Mask Management Callbacks ---
  const handleToggleMaskActive = useCallback((id: string) => {
    setSavedMasks(prevMasks =>
      prevMasks.map(mask =>
        mask.id === id ? { ...mask, isActive: !mask.isActive } : mask
      )
    );
  }, []);

  const handleRenameMask = useCallback((id: string, newName: string) => {
    setSavedMasks(prevMasks =>
      prevMasks.map(mask =>
        mask.id === id ? { ...mask, name: newName.trim() || `Mask ${prevMasks.findIndex(m => m.id === id) + 1}` } : mask
      )
    );
    console.log(`Renamed mask ${id} to ${newName}`);
  }, []);

  const handleDeleteMask = useCallback((id: string) => {
    setSavedMasks(prevMasks => prevMasks.filter(mask => mask.id !== id));
  }, []);

  const handleResetCurrentPoints = useCallback(() => { // Resets points OR box selection
    setPoints([]);
    setSelectedBox(null);
    setCurrentBox(null);
    setIsDrawingBox(false);
    setStartPoint(null);
    clearMaskingError();
  }, [clearMaskingError]);

  const handleClearSavedMasks = useCallback(() => {
    setSavedMasks([]);
    // Also clear current interaction state
    handleResetCurrentPoints();
  }, [handleResetCurrentPoints]);

  const handleDownloadActiveMasks = useCallback(() => {
    const activeMasks = savedMasks.filter(mask => mask.isActive);
    if (activeMasks.length === 0) {
      setMaskingError("No active masks selected to download.");
      return;
    }
    clearMaskingError();

    activeMasks.forEach((mask) => {
      const link = document.createElement('a');
      link.href = mask.maskB64Png.startsWith('data:image')
        ? mask.maskB64Png
        : `data:image/png;base64,${mask.maskB64Png}`;
      const safeName = mask.name.replace(/[^a-z0-9]/gi, '_').toLowerCase() || `mask_${mask.id.substring(0, 6)}`;
      link.download = `${safeName}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });
  }, [savedMasks, clearMaskingError]);

  // Effect to draw canvas whenever dependencies change
  useEffect(() => {
    // Debounce or throttle drawing if performance becomes an issue
    drawCanvas();
  }, [drawCanvas, points, savedMasks]); // Redraw when points or masks change


  return {
    points, // Current points (point mode)
    savedMasks, // List of generated masks
    selectedBox, // Currently selected box (box mode)
    maskingMode, // Current interaction mode
    isLoadingMask, // Loading state for point/box prediction
    isLoadingAutoMask, // Loading state for auto-generation
    maskingError, // Any error message
    canvasRef, // Ref for the canvas element
    imageRef, // Ref for the image element
    setPoints, // Setter for points (rarely needed externally)
    setSavedMasks, // Setter for masks (rarely needed externally)
    setMaskingMode, // Function to change mode
    // Canvas interaction handlers
    handleMouseDown,
    handleMouseMove,
    handleMouseUp,
    // Prediction triggers
    triggerPrediction, // Use this to predict from points or box
    handleGenerateMasksAutomatically, // Trigger auto-generation
    // Mask list management
    handleToggleMaskActive,
    handleRenameMask,
    handleDeleteMask,
    // Reset/Clear actions
    handleResetCurrentPoints, // Clears points OR box selection
    handleClearSavedMasks, // Clears all saved masks and current selection
    handleDownloadActiveMasks, // Downloads active masks
    clearMaskingError, // Clears any displayed error
    drawCanvas, // Function to redraw canvas (rarely needed externally)
  };
}
