import { useState, useRef, useCallback, DragEvent, ChangeEvent } from 'react';
import axios, { AxiosProgressEvent } from 'axios';
import './FileUpload.css';

interface FileUploadProps {
  onUploadComplete: (taskId: string) => void;
  onError: (error: string) => void;
}

interface SelectedFile {
  file: File;
  name: string;
  size: number;
  error?: string;
}

interface UploadProgress {
  loaded: number;
  total: number;
  percentage: number;
}

// Validation constants
const MAX_FILE_SIZE = 200 * 1024 * 1024; // 200MB in bytes
const MAX_FILES = 200;
const ALLOWED_EXTENSION = '.pdf';

// Format file size for display
const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

// Validate a single file
const validateFile = (file: File): string | null => {
  // Check extension
  if (!file.name.toLowerCase().endsWith(ALLOWED_EXTENSION)) {
    return `文件 "${file.name}" 不是 PDF 格式，仅支持 .pdf 文件`;
  }
  
  // Check file size
  if (file.size > MAX_FILE_SIZE) {
    return `文件 "${file.name}" 大小超过 200MB 限制 (当前: ${formatFileSize(file.size)})`;
  }
  
  return null;
};

export const FileUpload: React.FC<FileUploadProps> = ({ onUploadComplete, onError }) => {
  const [selectedFiles, setSelectedFiles] = useState<SelectedFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<UploadProgress | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Process files and validate them
  const processFiles = useCallback((files: FileList | File[]) => {
    const fileArray = Array.from(files);
    
    // Check total file count
    if (fileArray.length > MAX_FILES) {
      onError(`最多只能上传 ${MAX_FILES} 个文件，当前选择了 ${fileArray.length} 个`);
      return;
    }
    
    const processedFiles: SelectedFile[] = fileArray.map(file => ({
      file,
      name: file.name,
      size: file.size,
      error: validateFile(file) || undefined,
    }));
    
    setSelectedFiles(processedFiles);
  }, [onError]);

  // Handle drag events
  const handleDragEnter = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      processFiles(files);
    }
  }, [processFiles]);

  // Handle file input change
  const handleFileInputChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      processFiles(files);
    }
  }, [processFiles]);

  // Handle click on drop zone
  const handleDropZoneClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  // Remove a file from selection
  const handleRemoveFile = useCallback((index: number) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  }, []);

  // Clear all selected files
  const handleClearAll = useCallback(() => {
    setSelectedFiles([]);
    setUploadProgress(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, []);

  // Upload files
  const handleUpload = useCallback(async () => {
    // Check if there are valid files to upload
    const validFiles = selectedFiles.filter(f => !f.error);
    if (validFiles.length === 0) {
      onError('没有有效的文件可以上传');
      return;
    }

    setIsUploading(true);
    setUploadProgress({ loaded: 0, total: 100, percentage: 0 });

    try {
      const formData = new FormData();
      validFiles.forEach(({ file }) => {
        formData.append('files', file);
      });

      const response = await axios.post<{ task_id: string }>('/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent: AxiosProgressEvent) => {
          if (progressEvent.total) {
            const percentage = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setUploadProgress({
              loaded: progressEvent.loaded,
              total: progressEvent.total,
              percentage,
            });
          }
        },
      });

      // Success - call the callback with task_id
      onUploadComplete(response.data.task_id);
      
      // Clear the form after successful upload
      handleClearAll();
    } catch (error) {
      let errorMessage = '上传失败';
      if (axios.isAxiosError(error)) {
        if (error.response?.data?.detail) {
          errorMessage = error.response.data.detail;
        } else if (error.message) {
          errorMessage = error.message;
        }
      } else if (error instanceof Error) {
        errorMessage = error.message;
      }
      onError(errorMessage);
    } finally {
      setIsUploading(false);
    }
  }, [selectedFiles, onUploadComplete, onError, handleClearAll]);

  // Check if upload button should be disabled
  const hasValidFiles = selectedFiles.some(f => !f.error);
  const hasErrors = selectedFiles.some(f => f.error);

  return (
    <div className="file-upload-container">
      {/* Drop Zone */}
      <div
        className={`drop-zone ${isDragging ? 'dragging' : ''} ${isUploading ? 'disabled' : ''}`}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={isUploading ? undefined : handleDropZoneClick}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          multiple
          onChange={handleFileInputChange}
          className="file-input"
          disabled={isUploading}
        />
        <div className="drop-zone-content">
          <div className="drop-zone-icon">📄</div>
          <p className="drop-zone-text">
            {isDragging ? '释放文件以上传' : '拖拽 PDF 文件到此处，或点击选择文件'}
          </p>
          <p className="drop-zone-hint">
            支持多文件上传，单个文件最大 200MB，最多 200 个文件
          </p>
        </div>
      </div>

      {/* Selected Files List */}
      {selectedFiles.length > 0 && (
        <div className="selected-files">
          <div className="selected-files-header">
            <h3>已选择的文件 ({selectedFiles.length})</h3>
            <button 
              className="clear-all-btn" 
              onClick={handleClearAll}
              disabled={isUploading}
            >
              清除全部
            </button>
          </div>
          <ul className="file-list">
            {selectedFiles.map((file, index) => (
              <li key={`${file.name}-${index}`} className={`file-item ${file.error ? 'has-error' : ''}`}>
                <div className="file-info">
                  <span className="file-name">{file.name}</span>
                  <span className="file-size">{formatFileSize(file.size)}</span>
                </div>
                {file.error && (
                  <div className="file-error">{file.error}</div>
                )}
                <button
                  className="remove-file-btn"
                  onClick={() => handleRemoveFile(index)}
                  disabled={isUploading}
                  aria-label={`移除 ${file.name}`}
                >
                  ✕
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Upload Progress */}
      {isUploading && uploadProgress && (
        <div className="upload-progress">
          <div className="progress-bar-container">
            <div 
              className="progress-bar" 
              style={{ width: `${uploadProgress.percentage}%` }}
            />
          </div>
          <div className="progress-text">
            上传中... {uploadProgress.percentage}%
            {uploadProgress.total > 0 && (
              <span className="progress-size">
                ({formatFileSize(uploadProgress.loaded)} / {formatFileSize(uploadProgress.total)})
              </span>
            )}
          </div>
        </div>
      )}

      {/* Error Summary */}
      {hasErrors && !isUploading && (
        <div className="error-summary">
          ⚠️ 部分文件存在问题，这些文件将不会被上传
        </div>
      )}

      {/* Upload Button */}
      {selectedFiles.length > 0 && (
        <button
          className="upload-btn"
          onClick={handleUpload}
          disabled={isUploading || !hasValidFiles}
        >
          {isUploading ? '上传中...' : `上传 ${selectedFiles.filter(f => !f.error).length} 个文件`}
        </button>
      )}
    </div>
  );
};

export default FileUpload;
