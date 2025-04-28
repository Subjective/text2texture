import { useState } from 'react';

export type ModelMode = "protrude" | "carve";

export interface ModelParamsState {
  blockWidth: number;
  blockLength: number;
  blockThickness: number;
  depth: number; // Max height/depth variation
  baseHeight: number;
  mode: ModelMode;
  invert: boolean;
  includeColor: boolean; // Determines output format (STL vs PLY)
}

export interface UseModelParamsReturn extends ModelParamsState {
  setBlockWidth: React.Dispatch<React.SetStateAction<number>>;
  setBlockLength: React.Dispatch<React.SetStateAction<number>>;
  setBlockThickness: React.Dispatch<React.SetStateAction<number>>;
  setDepth: React.Dispatch<React.SetStateAction<number>>;
  setBaseHeight: React.Dispatch<React.SetStateAction<number>>;
  setMode: React.Dispatch<React.SetStateAction<ModelMode>>;
  setInvert: React.Dispatch<React.SetStateAction<boolean>>;
  setIncludeColor: React.Dispatch<React.SetStateAction<boolean>>;
  // Function to get all current params, useful for API calls
  getParams: () => ModelParamsState;
}

// Initial default values based on original App.tsx state
const initialParams: ModelParamsState = {
  blockWidth: 100,
  blockLength: 100,
  blockThickness: 10,
  depth: 25,
  baseHeight: 0,
  mode: "protrude",
  invert: false,
  includeColor: false,
};

export function useModelParams(defaults: Partial<ModelParamsState> = {}): UseModelParamsReturn {
  const mergedDefaults = { ...initialParams, ...defaults };

  const [blockWidth, setBlockWidth] = useState<number>(mergedDefaults.blockWidth);
  const [blockLength, setBlockLength] = useState<number>(mergedDefaults.blockLength);
  const [blockThickness, setBlockThickness] = useState<number>(mergedDefaults.blockThickness);
  const [depth, setDepth] = useState<number>(mergedDefaults.depth);
  const [baseHeight, setBaseHeight] = useState<number>(mergedDefaults.baseHeight);
  const [mode, setMode] = useState<ModelMode>(mergedDefaults.mode);
  const [invert, setInvert] = useState<boolean>(mergedDefaults.invert);
  const [includeColor, setIncludeColor] = useState<boolean>(mergedDefaults.includeColor);

  const getParams = (): ModelParamsState => ({
    blockWidth,
    blockLength,
    blockThickness,
    depth,
    baseHeight,
    mode,
    invert,
    includeColor,
  });

  return {
    blockWidth,
    blockLength,
    blockThickness,
    depth,
    baseHeight,
    mode,
    invert,
    includeColor,
    setBlockWidth,
    setBlockLength,
    setBlockThickness,
    setDepth,
    setBaseHeight,
    setMode,
    setInvert,
    setIncludeColor,
    getParams,
  };
}