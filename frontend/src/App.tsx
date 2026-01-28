import { useState, useCallback } from 'react';
import FileUpload from './components/FileUpload';
import ProgressPanel from './components/ProgressPanel';
import DownloadPanel from './components/DownloadPanel';
import './App.css';

// Application state type
type AppState = 'upload' | 'processing' | 'completed';

// Error notification interface
interface ErrorNotification {
  id: number;
  message: string;
  timestamp: Date;
}

function App() {
  // Core state
  const [appState, setAppState] = useState<AppState>('upload');
  const [taskId, setTaskId] = useState<string | null>(null);
  
  // Error notifications
  const [errors, setErrors] = useState<ErrorNotification[]>([]);
  const [nextErrorId, setNextErrorId] = useState(1);

  // Handle successful file upload
  const handleUploadComplete = useCallback((newTaskId: string) => {
    setTaskId(newTaskId);
    setAppState('processing');
    // Clear any previous errors when starting new task
    setErrors([]);
  }, []);

  // Handle task completion
  const handleTaskComplete = useCallback(() => {
    setAppState('completed');
  }, []);

  // Handle errors from any component
  const handleError = useCallback((message: string) => {
    const newError: ErrorNotification = {
      id: nextErrorId,
      message,
      timestamp: new Date(),
    };
    setErrors(prev => [...prev, newError]);
    setNextErrorId(prev => prev + 1);

    // Auto-dismiss error after 10 seconds
    setTimeout(() => {
      setErrors(prev => prev.filter(e => e.id !== newError.id));
    }, 10000);
  }, [nextErrorId]);

  // Dismiss a specific error
  const dismissError = useCallback((errorId: number) => {
    setErrors(prev => prev.filter(e => e.id !== errorId));
  }, []);

  // Start a new task (reset state)
  const handleStartNewTask = useCallback(() => {
    setTaskId(null);
    setAppState('upload');
    setErrors([]);
  }, []);

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <h1 className="app-title">📊 研报处理系统</h1>
        <p className="app-subtitle">Research Report Processor</p>
        <p className="app-description">
          上传 PDF 研究报告，自动转换为 Markdown/DOCX 格式，并生成中英/中日双语对照译文和总结
        </p>
      </header>

      {/* Error Notifications */}
      {errors.length > 0 && (
        <div className="error-notifications">
          {errors.map(error => (
            <div key={error.id} className="error-notification">
              <span className="error-icon">⚠️</span>
              <span className="error-message">{error.message}</span>
              <button
                className="error-dismiss"
                onClick={() => dismissError(error.id)}
                aria-label="关闭错误提示"
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Main Content */}
      <main className="app-main">
        {/* File Upload - Show when in upload state */}
        {appState === 'upload' && (
          <section className="section-upload">
            <FileUpload
              onUploadComplete={handleUploadComplete}
              onError={handleError}
            />
          </section>
        )}

        {/* Progress Panel - Show when processing */}
        {(appState === 'processing' || appState === 'completed') && (
          <section className="section-progress">
            <ProgressPanel
              taskId={taskId}
              onComplete={handleTaskComplete}
              onError={handleError}
            />
          </section>
        )}

        {/* Download Panel - Show when completed */}
        {appState === 'completed' && (
          <section className="section-download">
            <DownloadPanel
              taskId={taskId}
              isCompleted={true}
            />
          </section>
        )}

        {/* New Task Button - Show when completed */}
        {appState === 'completed' && (
          <section className="section-new-task">
            <button
              className="new-task-btn"
              onClick={handleStartNewTask}
            >
              <span className="new-task-icon">➕</span>
              <span className="new-task-text">处理新文件</span>
            </button>
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p className="footer-text">
          支持 PDF 文件上传 • 单文件最大 200MB • 最多 200 个文件
        </p>
      </footer>
    </div>
  );
}

export default App;
