# Research Report Processor

一个用于处理 PDF 研究报告的 Web 应用：MinerU 解析 +（可选）AI 翻译/总结。

## 功能特性

- PDF 文档上传与解析（基于 MinerU API）
- （可选）将解析后的 Markdown 整篇翻译为中文
- （可选）基于解析后的 Markdown 生成中文总结
- 实时处理进度展示（WebSocket）
- 多种格式输出下载（Markdown、DOCX）

## 技术栈

- **后端**: Python + FastAPI + Redis
- **前端**: React + TypeScript + Vite
- **部署**: Docker Compose

## 快速开始

### 1. 环境配置

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入必要的配置：

```env
MINERU_API_TOKEN=your_mineru_token
AI_API_ENDPOINT=https://api.openai.com/v1
AI_API_KEY=your_api_key
STORAGE_PATH=/app/storage
```

### 2. Docker 部署（推荐）

```bash
docker compose up -d
```

服务启动后访问 http://localhost:8765

### 3. 本地开发

**后端：**
```bash
pip install -e .
uvicorn backend.api.main:app --host 0.0.0.0 --port 8765 --reload
```

**前端：**
```bash
cd frontend
npm install
npm run dev
```

开发模式访问 http://localhost:8765

## 端口说明

| 服务 | 端口 | 说明 |
|------|------|------|
| 前端 | 8765 | Web 界面（对外暴露） |
| 后端 | 8765 | API 服务（Docker 内部） |
| Redis | 6379 | 内部使用 |

## API 接口

- `POST /api/upload` - 上传 PDF 文件
- `GET /api/tasks/{task_id}` - 查询任务状态
- `GET /api/download/{task_id}/{file_type}` - 下载输出文件
- `WS /ws/{task_id}` - WebSocket 进度推送
- `GET /health` - 健康检查

## 环境变量

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| MINERU_API_TOKEN | ✓ | - | MinerU API Token |
| AI_API_ENDPOINT | | - | AI 服务端点（开启翻译/总结） |
| AI_API_KEY | | - | AI 服务密钥（开启翻译/总结） |
| STORAGE_PATH | ✓ | - | 文件存储路径 |
| SERVER_PORT | | 8765 | 后端服务端口 |
| FRONTEND_PORT | | 8765 | 前端服务端口 |
| REDIS_URL | | redis://localhost:6379/0 | Redis 连接地址 |
| LOG_LEVEL | | INFO | 日志级别 |

## License

MIT
