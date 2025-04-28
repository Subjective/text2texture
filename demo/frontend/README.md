# Demo Frontend README

This directory contains the React + TypeScript frontend for the Text2Texture demo application, built using Vite.

## Functionality

The frontend provides a user interface for interacting with the Text2Texture backend APIs:

1.  **Image Generation:** Allows users to input text prompts to generate images via the backend's DALL-E integration.
2.  **Masking Interface:**
    *   Displays generated images.
    *   Supports automatic mask generation requests to the backend.
    *   Allows users to click points on the image to generate specific masks via the backend's SAM 2 integration.
    *   Manages and displays generated masks.
3.  **3D Model Generation:** Sends the generated image (and selected masks) to the backend to create a 3D model.
4.  **Model Viewing:** Displays the generated 3D model (STL/PLY) using a suitable viewer component.
5.  **Parameter Control:** Provides UI elements to adjust parameters for various backend processes (though specific parameters might be handled implicitly or explicitly).

## Project Structure

```text
demo/frontend/
├── public/                 # Static assets
│   └── vite.svg
├── src/                    # Source code
│   ├── assets/             # Image assets (e.g., react.svg)
│   ├── components/         # Reusable React components
│   │   ├── ErrorMessage/
│   │   ├── InputSection/
│   │   ├── LoadingIndicator/
│   │   ├── MaskingSection/
│   │   ├── MaskList/
│   │   ├── ModelViewer/
│   │   ├── ParamsSection/
│   │   └── ResultSection/
│   ├── config/             # Configuration files (e.g., API endpoints)
│   │   └── api.ts
│   ├── hooks/              # Custom React hooks for state management and logic
│   │   ├── useAppWorkflow.ts
│   │   ├── useImageInput.ts
│   │   ├── useMasking.ts
│   │   ├── useModelGeneration.ts
│   │   └── useModelParams.ts
│   ├── services/           # API interaction logic
│   │   └── api.ts
│   ├── types/              # TypeScript type definitions
│   │   └── app.types.ts
│   ├── utils/              # Utility functions
│   │   └── blobUtils.ts
│   ├── App.css             # Main application styles
│   ├── App.tsx             # Root application component
│   ├── index.css           # Global styles
│   ├── main.tsx            # Application entry point
│   └── vite-env.d.ts       # Vite environment types
├── .gitignore              # Git ignore rules
├── .mise.toml              # Mise configuration (for Node.js version)
├── eslint.config.js        # ESLint configuration
├── index.html              # Main HTML entry point
├── package-lock.json       # Exact dependency versions
├── package.json            # Project metadata and dependencies
├── README.md               # This file
├── tsconfig.app.json       # TypeScript config for the app
├── tsconfig.json           # Base TypeScript config
├── tsconfig.node.json      # TypeScript config for Node scripts (like Vite config)
└── vite.config.ts          # Vite build configuration
```

## Setup

1.  **Prerequisites:**
    *   Node.js (Version specified in `.mise.toml`)
    *   npm (usually comes with Node.js)
    *   (Optional but Recommended) `mise` for automatic Node.js version management.

2.  **Environment & Dependencies (Using Mise - Recommended):**
    *   If you have `mise` installed (see [mise.jdx.dev](https://mise.jdx.dev/)), simply navigate to the `demo/frontend` directory in your terminal and run:
        ```bash
        mise install
        # or just `mise i`
        ```
    *   This command will automatically:
        *   Read the `.mise.toml` file.
        *   Install the correct Node.js version specified.
        *   Install all dependencies listed in `package.json` using `npm install`.
    *   `mise` will automatically use the correct Node version whenever you `cd` into this directory.

3.  **Environment & Dependencies (Manual):**
    *   Ensure you have the correct Node.js version installed (check `.mise.toml` or `package.json` engines field if present). You can use tools like `nvm` to manage Node versions.
    *   Navigate to the `demo/frontend` directory.
    *   Install dependencies:
        ```bash
        npm install
        ```

4.  **Configure Backend URL:**
    *   The frontend needs to know where the backend server is running. This is typically configured in `src/config/api.ts` or via environment variables.
    *   By default, it assumes the backend is running on `http://127.0.0.1:5000`. If your backend runs on a different address or port, update the configuration accordingly. Check `src/config/api.ts` or look for `.env` file usage.

## Running the Frontend

1.  Navigate to the `demo/frontend` directory in your terminal.
2.  If using `mise`, the correct Node version should be active. If managing manually, ensure the correct Node version is selected.
3.  Start the Vite development server:
    ```bash
    npm run dev
    ```
4.  This will typically start the frontend application and make it accessible in your web browser, often at `http://localhost:5173` (Vite's default). The terminal output will provide the exact URL.
5.  Ensure the backend server is also running for the frontend to function correctly.
