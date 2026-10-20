/**
 * FileLoader - File System Access API for Context Folder Loading
 *
 * Loads context files from a user-selected folder.
 * Uses File System Access API (Chrome/Edge) with fallback to
 * <input webkitdirectory> (Firefox/Safari).
 *
 * Features:
 * - Feature detection for File System Access API
 * - Recursive folder reading
 * - File extension filtering
 * - Fallback for unsupported browsers
 * - File size limits
 *
 * Quality fixes addressed:
 * - HIGH: Handles permission denial gracefully
 * - HIGH: Handles empty folders
 *
 * Usage:
 *   if (FileLoader.isFileSystemAccessSupported()) {
 *     const files = await FileLoader.pickFolder();
 *   } else {
 *     const input = FileLoader.createFallbackInput();
 *     const files = await FileLoader.readFromInput(input);
 *   }
 *
 * @module file-loader
 */

/**
 * Default supported file extensions for context files.
 */
const DEFAULT_EXTENSIONS = ['.md', '.txt', '.ucl', '.json', '.yaml', '.yml', '.js', '.ts'];

/**
 * Maximum file size in bytes (1MB).
 */
const MAX_FILE_SIZE = 1 * 1024 * 1024;

/**
 * @typedef {Object} LoadedFile
 * @property {string} name - File name (relative path from folder root)
 * @property {string} content - Text content of the file
 * @property {number} size - File size in bytes
 */

class FileLoader {
    /**
     * Check if the File System Access API is supported.
     * @returns {boolean}
     */
    static isFileSystemAccessSupported() {
        return typeof window !== 'undefined' && 'showDirectoryPicker' in window;
    }

    /**
     * Open a folder picker and read all supported files.
     * Only works in Chrome/Edge with File System Access API.
     *
     * @param {string[]} [extensions=DEFAULT_EXTENSIONS] - Allowed file extensions
     * @returns {Promise<LoadedFile[]>}
     * @throws {Error} If API not supported or user denies permission
     */
    static async pickFolder(extensions = DEFAULT_EXTENSIONS) {
        if (!FileLoader.isFileSystemAccessSupported()) {
            throw new Error('File System Access API not supported in this browser');
        }

        let dirHandle;
        try {
            dirHandle = await window.showDirectoryPicker({ mode: 'read' });
        } catch (error) {
            if (error.name === 'AbortError') {
                throw new Error('Folder selection was cancelled');
            }
            throw new Error(`Failed to open folder picker: ${error.message}`);
        }

        const files = [];
        await FileLoader._readDirectoryRecursive(dirHandle, '', files, extensions);

        if (files.length === 0) {
            throw new Error('No supported files found in the selected folder');
        }

        // Sort by name for consistent ordering
        files.sort((a, b) => a.name.localeCompare(b.name));
        return files;
    }

    /**
     * Create a fallback <input> element for browsers without File System Access API.
     * The caller must append it to the DOM, click it, and then call readFromInput().
     *
     * @param {string[]} [extensions=DEFAULT_EXTENSIONS] - Allowed extensions (as accept attribute)
     * @returns {HTMLInputElement}
     */
    static createFallbackInput(extensions = DEFAULT_EXTENSIONS) {
        const input = document.createElement('input');
        input.type = 'file';
        input.webkitdirectory = true;
        input.multiple = true;
        // Convert extensions to accept format: '.md,.txt,.ucl,...'
        input.accept = extensions.join(',');
        return input;
    }

    /**
     * Read files from a fallback <input webkitdirectory> element.
     *
     * @param {HTMLInputElement} inputElement - The input element after file selection
     * @param {string[]} [extensions=DEFAULT_EXTENSIONS]
     * @returns {Promise<LoadedFile[]>}
     */
    static async readFromInput(inputElement, extensions = DEFAULT_EXTENSIONS) {
        const fileList = inputElement.files;
        if (!fileList || fileList.length === 0) {
            throw new Error('No files selected');
        }

        const files = [];
        for (const file of fileList) {
            const ext = FileLoader._getExtension(file.name);
            if (!extensions.includes(ext)) continue;
            if (file.size > MAX_FILE_SIZE) {
                console.warn(`[FileLoader] Skipping ${file.name}: exceeds ${MAX_FILE_SIZE} bytes`);
                continue;
            }

            try {
                const content = await file.text();
                files.push({
                    name: file.webkitRelativePath || file.name,
                    content,
                    size: file.size,
                });
            } catch (error) {
                console.warn(`[FileLoader] Failed to read ${file.name}:`, error.message);
            }
        }

        if (files.length === 0) {
            throw new Error('No supported files found in selection');
        }

        files.sort((a, b) => a.name.localeCompare(b.name));
        return files;
    }

    /**
     * Filter files to only those with supported extensions.
     *
     * @param {LoadedFile[]} files
     * @param {string[]} [extensions=DEFAULT_EXTENSIONS]
     * @returns {LoadedFile[]}
     */
    static filterFiles(files, extensions = DEFAULT_EXTENSIONS) {
        return files.filter(f => extensions.includes(FileLoader._getExtension(f.name)));
    }

    /**
     * Format file size for display.
     * @param {number} bytes
     * @returns {string}
     */
    static formatFileSize(bytes) {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    }

    /**
     * Get the total token estimate for a set of loaded files.
     * Rough estimate: total chars / 4.
     *
     * @param {LoadedFile[]} files
     * @returns {number}
     */
    static estimateTotalTokens(files) {
        const totalChars = files.reduce((sum, f) => sum + f.content.length, 0);
        return Math.ceil(totalChars / 4);
    }

    // ── Private ───────────────────────────────────────────────────────────

    /**
     * Recursively read a directory using File System Access API.
     * @private
     * @param {FileSystemDirectoryHandle} dirHandle
     * @param {string} path - Relative path prefix
     * @param {LoadedFile[]} results - Accumulator
     * @param {string[]} extensions
     */
    static async _readDirectoryRecursive(dirHandle, path, results, extensions) {
        for await (const entry of dirHandle.values()) {
            const entryPath = path ? `${path}/${entry.name}` : entry.name;

            if (entry.kind === 'directory') {
                // Skip hidden directories
                if (entry.name.startsWith('.')) continue;
                await FileLoader._readDirectoryRecursive(entry, entryPath, results, extensions);
            } else if (entry.kind === 'file') {
                const ext = FileLoader._getExtension(entry.name);
                if (!extensions.includes(ext)) continue;

                try {
                    const file = await entry.getFile();
                    if (file.size > MAX_FILE_SIZE) {
                        console.warn(`[FileLoader] Skipping ${entryPath}: exceeds size limit`);
                        continue;
                    }
                    const content = await file.text();
                    results.push({
                        name: entryPath,
                        content,
                        size: file.size,
                    });
                } catch (error) {
                    console.warn(`[FileLoader] Failed to read ${entryPath}:`, error.message);
                }
            }
        }
    }

    /**
     * Get file extension (lowercase, with dot).
     * @private
     */
    static _getExtension(filename) {
        const idx = filename.lastIndexOf('.');
        if (idx === -1) return '';
        return filename.slice(idx).toLowerCase();
    }
}

// ── Exports ────────────────────────────────────────────────────────────────
export { FileLoader, DEFAULT_EXTENSIONS, MAX_FILE_SIZE };
