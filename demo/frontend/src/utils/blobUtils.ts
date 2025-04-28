// src/utils/blobUtils.ts

/**
 * Converts a Base64 string (with or without Data URI prefix) to a Blob.
 * Throws an error if decoding fails.
 *
 * @param base64 The Base64 encoded string.
 * @param contentType The content type of the resulting Blob (e.g., 'image/png').
 * @param sliceSize The size of chunks to process the string in.
 * @returns A Blob object.
 */
export const base64ToBlob = (base64: string, contentType = '', sliceSize = 512): Blob => {
  // Remove Data URI prefix if present
  const base64Data = base64.includes(',') ? base64.split(',')[1] : base64;
  try {
    const byteCharacters = atob(base64Data);
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
  } catch (e) {
    console.error("Error decoding base64 string:", e, base64.substring(0, 50) + "..."); // Log error and part of string
    // Re-throw to indicate failure clearly.
    throw new Error("Failed to decode base64 string.");
  }
};