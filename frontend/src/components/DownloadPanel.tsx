import { useState, useCallback } from 'react';
import './DownloadPanel.css';

interface DownloadPanelProps {
  taskId: string | null;
  isCompleted: boolean;
}

// Output file types matching backend OutputFileType enum
type OutputFileType = 'original_md' | 'original_docx';

interface FileTypeInfo {
  type: OutputFileType;
  label: string;
  icon: string;
  extension: string;
}

// File type configuration with Chinese labels
const FILE_TYPES: FileTypeInfo[] = [
  { type: 'original_md', label: '原文 Markdown', icon: '📄', extension: '.md' },
  { type: 'original_docx', label: '原文 DOCX', icon: '📝', extension: '.docx' },
];

// Track download state for each file type
interface DownloadState {
  isDownloading: boolean;
  error: string | null;
}

export const DownloadPanel: React.FC<DownloadPanelProps> = ({
  taskId,
  isCompleted,
}) => {
  // Track download state for each file type
  const [downloadStates, setDownloadStates] = useState<Record<OutputFileType, DownloadState>>({
    original_md: { isDownloading: false, error: null },
    original_docx: { isDownloading: false, error: null },
  });

  // Update download state for a specific file type
  const updateDownloadState = useCallback((fileType: OutputFileType, state: Partial<DownloadState>) => {
    setDownloadStates(prev => ({
      ...prev,
      [fileType]: { ...prev[fileType], ...state },
    }));
  }, []);

  // Handle file download
  const handleDownload = useCallback(async (fileType: OutputFileType, fileInfo: FileTypeInfo) => {
    if (!taskId) return;

    // Set downloading state
    updateDownloadState(fileType, { isDownloading: true, error: null });

    try {
      const downloadUrl = `/api/download/${taskId}/${fileType}`;
      
      // Fetch the file
      const response = await fetch(downloadUrl);
      
      if (!response.ok) {
        // Try to get error message from response
        let errorMessage = `下载失败 (${response.status})`;
        try {
          const errorData = await response.json();
          if (errorData.detail) {
            errorMessage = errorData.detail;
          }
        } catch {
          // If response is not JSON, use status text
          errorMessage = `下载失败: ${response.statusText || response.status}`;
        }
        throw new Error(errorMessage);
      }

      // Get the blob from response
      const blob = await response.blob();
      
      // Create download link
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      
      // Try to get filename from Content-Disposition header
      const contentDisposition = response.headers.get('Content-Disposition');
      let filename = `${taskId}_${fileType}${fileInfo.extension}`;
      
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
        if (filenameMatch && filenameMatch[1]) {
          filename = filenameMatch[1].replace(/['"]/g, '');
        }
      }
      
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      
      // Cleanup
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      
      // Clear downloading state
      updateDownloadState(fileType, { isDownloading: false });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '下载失败，请重试';
      updateDownloadState(fileType, { isDownloading: false, error: errorMessage });
    }
  }, [taskId, updateDownloadState]);

  // Clear error for a specific file type
  const clearError = useCallback((fileType: OutputFileType) => {
    updateDownloadState(fileType, { error: null });
  }, [updateDownloadState]);

  // Don't render if no taskId or task is not completed
  if (!taskId || !isCompleted) {
    return null;
  }

  return (
    <div className="download-panel">
      <div className="download-panel-header">
        <h3>📥 下载输出文件</h3>
        <p className="download-panel-subtitle">处理完成，请下载您需要的文件</p>
      </div>

      <div className="download-buttons">
        {FILE_TYPES.map((fileInfo) => {
          const state = downloadStates[fileInfo.type];
          
          return (
            <div key={fileInfo.type} className="download-item">
              <button
                className={`download-btn ${state.isDownloading ? 'downloading' : ''}`}
                onClick={() => handleDownload(fileInfo.type, fileInfo)}
                disabled={state.isDownloading}
              >
                <span className="download-btn-icon">
                  {state.isDownloading ? '⏳' : fileInfo.icon}
                </span>
                <span className="download-btn-label">
                  {state.isDownloading ? '下载中...' : fileInfo.label}
                </span>
                <span className="download-btn-arrow">↓</span>
              </button>
              
              {state.error && (
                <div className="download-error">
                  <span className="download-error-text">{state.error}</span>
                  <button 
                    className="download-error-dismiss"
                    onClick={() => clearError(fileInfo.type)}
                    aria-label="关闭错误提示"
                  >
                    ✕
                  </button>
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="download-panel-footer">
        <span className="task-label">任务 ID:</span>
        <span className="task-id">{taskId}</span>
      </div>
    </div>
  );
};

export default DownloadPanel;
