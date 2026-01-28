import { useState, useEffect, useRef, useCallback } from 'react';
import './ProgressPanel.css';

interface ProgressPanelProps {
  taskId: string | null;
  onComplete: () => void;
  onError: (error: string) => void;
}

interface ProgressUpdate {
  task_id: string;
  stage: 'uploading' | 'parsing' | 'downloading' | 'translating' | 'summarizing' | 'generating' | 'completed' | 'failed';
  progress: number;
  total: number;
  percentage: number;
  message: string;
  timestamp?: string;
}

// Stage labels in Chinese
const STAGE_LABELS: Record<string, string> = {
  uploading: '上传中',
  parsing: '解析中',
  downloading: '下载中',
  translating: '翻译中',
  summarizing: '总结中',
  generating: '生成中',
  completed: '已完成',
  failed: '失败',
};

// Stage order for progress visualization
const STAGE_ORDER: string[] = [
  'uploading',
  'parsing',
  'downloading',
  'translating',
  'summarizing',
  'generating',
  'completed',
];

// Get stage index for comparison
const getStageIndex = (stage: string): number => {
  const index = STAGE_ORDER.indexOf(stage);
  return index >= 0 ? index : -1;
};

// Get stage icon
const getStageIcon = (stage: string, currentStage: string): string => {
  const stageIndex = getStageIndex(stage);
  const currentIndex = getStageIndex(currentStage);
  
  if (currentStage === 'failed') {
    return stage === currentStage ? '❌' : '⚪';
  }
  
  if (stageIndex < currentIndex) {
    return '✅'; // Completed
  } else if (stageIndex === currentIndex) {
    return '🔄'; // In progress
  } else {
    return '⚪'; // Pending
  }
};

export const ProgressPanel: React.FC<ProgressPanelProps> = ({
  taskId,
  onComplete,
  onError,
}) => {
  const [progress, setProgress] = useState<ProgressUpdate | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const reconnectAttemptsRef = useRef<number>(0);
  const maxReconnectAttempts = 5;

  // Clean up WebSocket connection
  const cleanupWebSocket = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  // Connect to WebSocket
  const connectWebSocket = useCallback((taskIdToConnect: string) => {
    // Clean up existing connection
    cleanupWebSocket();
    
    setConnectionStatus('connecting');
    setErrorMessage(null);
    
    // Determine WebSocket URL based on current location
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws/${taskIdToConnect}`;
    
    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;
      
      ws.onopen = () => {
        setConnectionStatus('connected');
        reconnectAttemptsRef.current = 0;
        console.log(`WebSocket connected for task ${taskIdToConnect}`);
      };
      
      ws.onmessage = (event) => {
        try {
          const data: ProgressUpdate = JSON.parse(event.data);
          setProgress(data);
          
          // Handle completion
          if (data.stage === 'completed') {
            onComplete();
          }
          
          // Handle failure
          if (data.stage === 'failed') {
            setErrorMessage(data.message || '处理失败');
            onError(data.message || '处理失败');
          }
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };
      
      ws.onerror = (event) => {
        console.error('WebSocket error:', event);
        setConnectionStatus('error');
      };
      
      ws.onclose = (event) => {
        console.log(`WebSocket closed: code=${event.code}, reason=${event.reason}`);
        setConnectionStatus('disconnected');
        
        // Attempt to reconnect if not a normal close and task is still active
        if (event.code !== 1000 && reconnectAttemptsRef.current < maxReconnectAttempts) {
          const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 30000);
          reconnectAttemptsRef.current += 1;
          
          console.log(`Attempting to reconnect in ${delay}ms (attempt ${reconnectAttemptsRef.current})`);
          
          reconnectTimeoutRef.current = window.setTimeout(() => {
            if (taskIdToConnect) {
              connectWebSocket(taskIdToConnect);
            }
          }, delay);
        }
      };
    } catch (e) {
      console.error('Failed to create WebSocket:', e);
      setConnectionStatus('error');
      setErrorMessage('无法建立 WebSocket 连接');
    }
  }, [cleanupWebSocket, onComplete, onError]);

  // Effect to manage WebSocket connection based on taskId
  useEffect(() => {
    if (taskId) {
      // Reset state for new task
      setProgress(null);
      setErrorMessage(null);
      reconnectAttemptsRef.current = 0;
      
      connectWebSocket(taskId);
    } else {
      cleanupWebSocket();
      setConnectionStatus('disconnected');
      setProgress(null);
      setErrorMessage(null);
    }
    
    // Cleanup on unmount or taskId change
    return () => {
      cleanupWebSocket();
    };
  }, [taskId, connectWebSocket, cleanupWebSocket]);

  // Don't render if no taskId
  if (!taskId) {
    return null;
  }

  const currentStage = progress?.stage || 'uploading';
  const isCompleted = currentStage === 'completed';
  const isFailed = currentStage === 'failed';

  return (
    <div className={`progress-panel ${isFailed ? 'has-error' : ''} ${isCompleted ? 'completed' : ''}`}>
      <div className="progress-panel-header">
        <h3>处理进度</h3>
        <div className={`connection-status ${connectionStatus}`}>
          <span className="status-dot"></span>
          <span className="status-text">
            {connectionStatus === 'connecting' && '连接中...'}
            {connectionStatus === 'connected' && '已连接'}
            {connectionStatus === 'disconnected' && '已断开'}
            {connectionStatus === 'error' && '连接错误'}
          </span>
        </div>
      </div>

      {/* Overall Progress Bar */}
      <div className="overall-progress">
        <div className="progress-bar-container">
          <div 
            className={`progress-bar ${isFailed ? 'error' : ''} ${isCompleted ? 'completed' : ''}`}
            style={{ width: `${progress?.percentage || 0}%` }}
          />
        </div>
        <div className="progress-percentage">
          {Math.round(progress?.percentage || 0)}%
        </div>
      </div>

      {/* Stage Indicators */}
      <div className="stage-indicators">
        {STAGE_ORDER.filter(s => s !== 'completed').map((stage) => {
          const stageIndex = getStageIndex(stage);
          const currentIndex = getStageIndex(currentStage);
          const isActive = stage === currentStage && !isFailed;
          const isComplete = stageIndex < currentIndex && !isFailed;
          const isPending = stageIndex > currentIndex || isFailed;
          
          return (
            <div 
              key={stage}
              className={`stage-item ${isActive ? 'active' : ''} ${isComplete ? 'complete' : ''} ${isPending ? 'pending' : ''}`}
            >
              <div className="stage-icon">
                {getStageIcon(stage, currentStage)}
              </div>
              <div className="stage-label">
                {STAGE_LABELS[stage]}
              </div>
              {isActive && progress && progress.total > 0 && (
                <div className="stage-progress">
                  {progress.progress}/{progress.total}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Current Status Message */}
      {progress?.message && !isFailed && (
        <div className="status-message">
          <span className="message-icon">ℹ️</span>
          <span className="message-text">{progress.message}</span>
        </div>
      )}

      {/* Error Message */}
      {(isFailed || errorMessage) && (
        <div className="error-message">
          <span className="error-icon">⚠️</span>
          <span className="error-text">{errorMessage || progress?.message || '处理过程中发生错误'}</span>
        </div>
      )}

      {/* Completion Message */}
      {isCompleted && (
        <div className="completion-message">
          <span className="completion-icon">🎉</span>
          <span className="completion-text">处理完成！您可以下载输出文件了。</span>
        </div>
      )}

      {/* Task ID Display */}
      <div className="task-info">
        <span className="task-label">任务 ID:</span>
        <span className="task-id">{taskId}</span>
      </div>
    </div>
  );
};

export default ProgressPanel;
