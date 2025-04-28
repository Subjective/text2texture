import { useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
import * as THREE from 'three'; // Import THREE for BufferGeometry

interface ModelViewerProps {
  fileUrl: string | null;
  fileType: string | null;
}

export function ModelViewer({ fileUrl, fileType }: ModelViewerProps) {
  // Use THREE.BufferGeometry for type safety, though loader might return Geometry
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null);
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
    const lowerFileType = fileType.toLowerCase();

    if (lowerFileType === "stl") {
      loader = new STLLoader();
    } else if (lowerFileType === "ply") {
      loader = new PLYLoader();
    } else {
      console.error("Unsupported file type for viewer:", fileType);
      setLoadingError(`Cannot display file type: ${fileType}`);
      return;
    }

    console.log(`Loading ${fileType.toUpperCase()} model from: ${fileUrl}`); // Debug log

    loader.load(
      fileUrl,
      (loadedGeom) => { // Rename to avoid conflict with state variable
        console.log(`${fileType.toUpperCase()} model loaded successfully.`); // Debug log

        // Ensure we have BufferGeometry - loaders should provide this directly
        if (!(loadedGeom instanceof THREE.BufferGeometry)) {
          console.error("Loaded geometry is not an instance of THREE.BufferGeometry!");
          setLoadingError("Failed to process loaded 3D model data (unexpected format).");
          return;
        }

        const bufferGeom = loadedGeom; // Assign directly

        if (lowerFileType === "ply") {
          // Compute normals if they don't exist
          if (!bufferGeom.attributes.normal) {
            bufferGeom.computeVertexNormals(); // Important for lighting on PLY
            console.log("Computed vertex normals for PLY."); // Debug log
          } else {
            console.log("PLY geometry already has normals.");
          }
        }

        // Center the geometry before setting state
        bufferGeom.center();
        console.log("Centered geometry."); // Debug log

        setGeometry(bufferGeom);
      },
      (xhr) => { // onProgress callback
        console.log(`Model loading progress: ${(xhr.loaded / xhr.total * 100).toFixed(2)}%`);
      },
      (err) => { // onError callback
        console.error("Error loading model:", err);
        setLoadingError(`Failed to load 3D model preview. Check console for details. Error: ${(err instanceof Error ? err.message : String(err))}`);
      }
    );

    // Cleanup function
    return () => {
      console.log("Cleaning up ModelViewer for:", fileUrl); // Debug log
      // Dispose geometry when component unmounts or fileUrl changes
      geometry?.dispose();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fileUrl, fileType]); // Keep geometry out of deps to avoid loop on dispose/set

  if (loadingError) {
    return (
      <div className="flex items-center justify-center h-full text-lg font-medium text-red-600 bg-red-50 dark:bg-gray-800 dark:text-red-400 p-4 rounded-md">
        Error loading preview: {loadingError}
      </div>
    );
  }

  if (!geometry) {
    return (
      <div className="flex items-center justify-center h-full text-lg font-medium text-gray-600 dark:text-gray-300">
        Loading 3D model preview...
      </div>
    );
  }

  return (
    <Canvas className="w-full h-full" camera={{ position: [150, 150, 150], fov: 50 }}> {/* Adjusted fov */}
      <ambientLight intensity={0.7} /> {/* Slightly increased ambient light */}
      <directionalLight intensity={1.0} position={[50, 100, 150]} castShadow /> {/* Stronger main light */}
      <directionalLight intensity={0.5} position={[-80, -40, -100]} /> {/* Back/fill light */}
      <mesh geometry={geometry} scale={1} castShadow receiveShadow> {/* Ensure scale is appropriate, geometry is centered */}
        {/* Primitive is generally not needed if you pass geometry directly to mesh */}
        {/* <primitive object={geometry} attach="geometry" /> */}
        {fileType?.toLowerCase() === "ply" && geometry.attributes.color ? (
          // Use vertexColors if they exist in the PLY
          <meshStandardMaterial vertexColors={true} roughness={0.7} metalness={0.1} />
        ) : (
          // Default material for STL or PLY without color
          <meshStandardMaterial color="#cccccc" roughness={0.6} metalness={0.2} />
        )}
      </mesh>
      {/* TODO: Add a ground plane */}
      {/* <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -geometry?.boundingBox?.min?.y || -50, 0]} receiveShadow>
            <planeGeometry args={[500, 500]} />
            <shadowMaterial opacity={0.3} />
        </mesh> */}
      <OrbitControls />
    </Canvas>
  );
}
