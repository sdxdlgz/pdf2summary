"""
Task Manager for the Research Report Processor.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from redis.asyncio import Redis
from fastapi import UploadFile

from backend.models import (
    BatchResultResponse,
    FileInfo,
    MineruTaskState,
    OutputFileType,
    ProgressUpdate,
    TaskStage,
    TaskStatus,
    generate_data_id,
    generate_task_id,
)


logger = logging.getLogger(__name__)


INITIAL_POLL_INTERVAL = 2
MAX_POLL_INTERVAL = 60
BACKOFF_MULTIPLIER = 2

STAGE_WEIGHTS = {
    TaskStage.UPLOADING: 10,
    TaskStage.PARSING: 30,
    TaskStage.DOWNLOADING: 10,
    TaskStage.TRANSLATING: 30,
    TaskStage.SUMMARIZING: 15,
    TaskStage.GENERATING: 5,
}

STAGE_CUMULATIVE_WEIGHTS = {
    TaskStage.UPLOADING: 0,
    TaskStage.PARSING: 10,
    TaskStage.DOWNLOADING: 40,
    TaskStage.TRANSLATING: 50,
    TaskStage.SUMMARIZING: 80,
    TaskStage.GENERATING: 95,
    TaskStage.COMPLETED: 100,
    TaskStage.FAILED: 0,
}

TASK_KEY_PREFIX = "task:"
PROGRESS_CHANNEL_PREFIX = "progress:"


def calculate_backoff_interval(attempt: int) -> float:
    interval = INITIAL_POLL_INTERVAL * (BACKOFF_MULTIPLIER ** attempt)
    return min(interval, MAX_POLL_INTERVAL)


def is_terminal_state(state: MineruTaskState) -> bool:
    return state in (MineruTaskState.DONE, MineruTaskState.FAILED)


def should_continue_polling(state: MineruTaskState) -> bool:
    return not is_terminal_state(state)


class TaskManagerError(Exception):
    def __init__(self, message: str, task_id: Optional[str] = None):
        self.message = message
        self.task_id = task_id
        super().__init__(self.message)


class TaskManager:
    def __init__(
        self,
        redis_client: Redis,
        mineru_client: Optional[Any] = None,
        file_storage: Optional[Any] = None,
        ai_client: Optional[Any] = None,
        document_processor: Optional[Any] = None,
    ):
        self.redis_client = redis_client
        self.mineru_client = mineru_client
        self.file_storage = file_storage
        self.ai_client = ai_client
        self.document_processor = document_processor
    
    def _get_task_key(self, task_id: str) -> str:
        return f"{TASK_KEY_PREFIX}{task_id}"
    
    def _get_progress_channel(self, task_id: str) -> str:
        return f"{PROGRESS_CHANNEL_PREFIX}{task_id}"

    async def create_task(self, files: list[UploadFile]) -> str:
        if not files:
            raise TaskManagerError("No files provided for processing")
        task_id = generate_task_id()
        file_infos: list[FileInfo] = []
        for file in files:
            data_id = generate_data_id()
            if hasattr(file, 'size') and file.size is not None:
                size = file.size
            else:
                content = await file.read()
                size = len(content)
                await file.seek(0)
            file_info = FileInfo(name=file.filename or "unknown", data_id=data_id, size=size)
            file_infos.append(file_info)
        now = datetime.now()
        initial_progress = ProgressUpdate(
            task_id=task_id, stage=TaskStage.UPLOADING, progress=0, total=len(files),
            percentage=0.0, message="Task created", timestamp=now,
        )
        task_status = TaskStatus(
            task_id=task_id, stage=TaskStage.UPLOADING, files=file_infos,
            progress=initial_progress, outputs=None, error=None, created_at=now, updated_at=now,
        )
        task_key = self._get_task_key(task_id)
        try:
            await self.redis_client.set(task_key, task_status.model_dump_json())
        except Exception as e:
            raise TaskManagerError(message=f"Failed to store task: {e}", task_id=task_id) from e
        return task_id

    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        task_key = self._get_task_key(task_id)
        try:
            task_data = await self.redis_client.get(task_key)
            if task_data:
                return TaskStatus.model_validate_json(task_data)
            return None
        except Exception as e:
            logger.error("Failed to get task status: %s", e)
            return None

    def calculate_overall_percentage(self, stage: TaskStage, stage_progress: float) -> float:
        if stage == TaskStage.COMPLETED:
            return 100.0
        if stage == TaskStage.FAILED:
            return STAGE_CUMULATIVE_WEIGHTS.get(stage, 0.0)
        cumulative = STAGE_CUMULATIVE_WEIGHTS.get(stage, 0.0)
        stage_weight = STAGE_WEIGHTS.get(stage, 0.0)
        percentage = cumulative + (stage_progress * stage_weight)
        return max(0.0, min(100.0, percentage))

    async def update_progress(self, task_id: str, stage: TaskStage, progress: int, total: int, message: str) -> None:
        stage_progress = progress / total if total > 0 else 0.0
        percentage = self.calculate_overall_percentage(stage, stage_progress)
        now = datetime.now()
        progress_update = ProgressUpdate(
            task_id=task_id, stage=stage, progress=progress, total=total,
            percentage=percentage, message=message, timestamp=now,
        )
        task_key = self._get_task_key(task_id)
        try:
            task_data = await self.redis_client.get(task_key)
            if task_data:
                task_status = TaskStatus.model_validate_json(task_data)
                task_status.stage = stage
                task_status.progress = progress_update
                task_status.updated_at = now
                await self.redis_client.set(task_key, task_status.model_dump_json())
        except Exception as e:
            logger.warning("Failed to update task status: %s", e)
        channel = self._get_progress_channel(task_id)
        try:
            await self.redis_client.publish(channel, progress_update.model_dump_json())
        except Exception as e:
            logger.warning("Failed to broadcast progress: %s", e)

    async def _broadcast_error(self, task_id: str, error_msg: str) -> None:
        now = datetime.now()
        progress_update = ProgressUpdate(
            task_id=task_id, stage=TaskStage.FAILED, progress=0, total=0,
            percentage=0.0, message=error_msg, timestamp=now,
        )
        task_key = self._get_task_key(task_id)
        try:
            task_data = await self.redis_client.get(task_key)
            if task_data:
                task_status = TaskStatus.model_validate_json(task_data)
                task_status.stage = TaskStage.FAILED
                task_status.progress = progress_update
                task_status.error = error_msg
                task_status.updated_at = now
                await self.redis_client.set(task_key, task_status.model_dump_json())
        except Exception as e:
            logger.warning("Failed to update task status: %s", e)
        channel = self._get_progress_channel(task_id)
        try:
            await self.redis_client.publish(channel, progress_update.model_dump_json())
        except Exception as e:
            logger.warning("Failed to broadcast error: %s", e)

    async def poll_mineru_status(self, task_id: str, batch_id: str) -> BatchResultResponse:
        if self.mineru_client is None:
            raise TaskManagerError(message="MinerU client not configured", task_id=task_id)
        attempt = 0
        while True:
            try:
                batch_result = await self.mineru_client.get_batch_results(batch_id)
            except Exception as e:
                raise TaskManagerError(message=f"Failed to get batch results: {e}", task_id=task_id) from e
            all_done = True
            any_failed = False
            failed_errors: list[str] = []
            total_extracted = 0
            total_pages = 0
            for result in batch_result.extract_result:
                state = result.state
                if not is_terminal_state(state):
                    all_done = False
                if state == MineruTaskState.FAILED:
                    any_failed = True
                    if result.err_msg:
                        failed_errors.append(f"{result.file_name}: {result.err_msg}")
                if state == MineruTaskState.RUNNING:
                    if result.extracted_pages is not None:
                        total_extracted += result.extracted_pages
                    if result.total_pages is not None:
                        total_pages += result.total_pages
            if total_pages > 0:
                await self.update_progress(
                    task_id=task_id, stage=TaskStage.PARSING, progress=total_extracted,
                    total=total_pages, message=f"Parsing: {total_extracted}/{total_pages} pages",
                )
            if any_failed:
                error_msg = "; ".join(failed_errors) if failed_errors else "Unknown error"
                await self._broadcast_error(task_id, error_msg)
                raise TaskManagerError(message=f"MinerU parsing failed: {error_msg}", task_id=task_id)
            if all_done:
                return batch_result
            interval = calculate_backoff_interval(attempt)
            await asyncio.sleep(interval)
            attempt += 1

    async def process_task(self, task_id: str) -> None:
        logger.info("Task %s: processing started", task_id)
        try:
            task_status = await self.get_task_status(task_id)
            if task_status is None:
                raise TaskManagerError(message=f"Task {task_id} not found", task_id=task_id)
            await self.update_progress(task_id=task_id, stage=TaskStage.UPLOADING, progress=0, total=len(task_status.files), message="Uploading files")
            batch_id = await self._upload_files_to_mineru(task_id, task_status)
            logger.info("Task %s: uploaded to MinerU (batch_id=%s)", task_id, batch_id)
            await self.update_progress(task_id=task_id, stage=TaskStage.UPLOADING, progress=len(task_status.files), total=len(task_status.files), message="Files uploaded")
            await self.update_progress(task_id=task_id, stage=TaskStage.PARSING, progress=0, total=100, message="Waiting for MinerU")
            batch_result = await self.poll_mineru_status(task_id, batch_id)
            logger.info("Task %s: MinerU parsing finished", task_id)
            await self.update_progress(task_id=task_id, stage=TaskStage.DOWNLOADING, progress=0, total=len(batch_result.extract_result), message="Downloading results")
            await self._download_results(task_id, batch_result)
            await self.update_progress(task_id=task_id, stage=TaskStage.DOWNLOADING, progress=len(batch_result.extract_result), total=len(batch_result.extract_result), message="Results downloaded")

            if self.file_storage is None:
                raise TaskManagerError(message="File storage not configured", task_id=task_id)

            extracted_md_path = self.file_storage.get_extracted_markdown(task_id)
            if extracted_md_path is None:
                raise TaskManagerError(message="Extracted Markdown not found", task_id=task_id)

            original_markdown = Path(extracted_md_path).read_text(
                encoding="utf-8",
                errors="replace",
            )

            extracted_docx_path = self.file_storage.get_extracted_docx(task_id)
            extracted_docx_bytes: bytes | None = None
            if extracted_docx_path is not None:
                extracted_docx_bytes = Path(extracted_docx_path).read_bytes()

            if self.ai_client is None:
                raise TaskManagerError(message="AI client not configured", task_id=task_id)

            source_lang = await self.ai_client.detect_language(original_markdown)

            # Translation (chunk progress is reported via callback)
            from backend.services.ai_client import split_into_chunks  # local import

            total_chunks = max(1, len(split_into_chunks(original_markdown)))
            await self.update_progress(
                task_id=task_id,
                stage=TaskStage.TRANSLATING,
                progress=0,
                total=total_chunks,
                message="Starting translation",
            )

            def translation_progress_callback(completed: int, total: int) -> None:
                asyncio.create_task(
                    self.update_progress(
                        task_id=task_id,
                        stage=TaskStage.TRANSLATING,
                        progress=completed,
                        total=total,
                        message=f"Translating: {completed}/{total} chunks",
                    )
                )

            bilingual_markdown = await self.ai_client.translate_document(
                original_markdown,
                source_lang,
                progress_callback=translation_progress_callback,
            )

            await self.update_progress(
                task_id=task_id,
                stage=TaskStage.TRANSLATING,
                progress=total_chunks,
                total=total_chunks,
                message="Translation complete",
            )

            # Summarization
            await self.update_progress(
                task_id=task_id,
                stage=TaskStage.SUMMARIZING,
                progress=0,
                total=1,
                message="Generating summary",
            )
            summary_markdown = await self.ai_client.summarize(original_markdown, source_lang)
            await self.update_progress(
                task_id=task_id,
                stage=TaskStage.SUMMARIZING,
                progress=1,
                total=1,
                message="Summary complete",
            )

            if self.document_processor is None:
                raise TaskManagerError(message="Document processor not configured", task_id=task_id)

            # Generate and save output files
            await self.update_progress(
                task_id=task_id,
                stage=TaskStage.GENERATING,
                progress=0,
                total=5,
                message="Generating output files",
            )

            output_basename = self._get_output_basename(task_id, task_status)
            outputs: dict[str, str] = {}

            original_md_filename = f"{output_basename}.md"
            original_docx_filename = f"{output_basename}.docx"
            bilingual_md_filename = f"{output_basename}_bilingual.md"
            bilingual_docx_filename = f"{output_basename}_bilingual.docx"
            summary_filename = f"{output_basename}_summary.md"

            outputs[OutputFileType.ORIGINAL_MD.value] = self.file_storage.save_output(
                task_id,
                OutputFileType.ORIGINAL_MD.value,
                original_md_filename,
                original_markdown.encode("utf-8"),
            )
            await self.update_progress(
                task_id=task_id,
                stage=TaskStage.GENERATING,
                progress=1,
                total=5,
                message="Saved original Markdown",
            )

            if extracted_docx_bytes is None:
                extracted_docx_bytes = await self.document_processor.markdown_to_docx(
                    original_markdown
                )

            outputs[OutputFileType.ORIGINAL_DOCX.value] = self.file_storage.save_output(
                task_id,
                OutputFileType.ORIGINAL_DOCX.value,
                original_docx_filename,
                extracted_docx_bytes,
            )
            await self.update_progress(
                task_id=task_id,
                stage=TaskStage.GENERATING,
                progress=2,
                total=5,
                message="Saved original DOCX",
            )

            outputs[OutputFileType.BILINGUAL_MD.value] = self.file_storage.save_output(
                task_id,
                OutputFileType.BILINGUAL_MD.value,
                bilingual_md_filename,
                bilingual_markdown.encode("utf-8"),
            )
            await self.update_progress(
                task_id=task_id,
                stage=TaskStage.GENERATING,
                progress=3,
                total=5,
                message="Saved bilingual Markdown",
            )

            bilingual_docx_bytes = await self.document_processor.markdown_to_docx(bilingual_markdown)
            outputs[OutputFileType.BILINGUAL_DOCX.value] = self.file_storage.save_output(
                task_id,
                OutputFileType.BILINGUAL_DOCX.value,
                bilingual_docx_filename,
                bilingual_docx_bytes,
            )
            await self.update_progress(
                task_id=task_id,
                stage=TaskStage.GENERATING,
                progress=4,
                total=5,
                message="Saved bilingual DOCX",
            )

            outputs[OutputFileType.SUMMARY.value] = self.file_storage.save_output(
                task_id,
                OutputFileType.SUMMARY.value,
                summary_filename,
                summary_markdown.encode("utf-8"),
            )
            await self.update_progress(
                task_id=task_id,
                stage=TaskStage.GENERATING,
                progress=5,
                total=5,
                message="Saved summary",
            )

            await self._update_task_outputs(task_id, outputs)
            await self.update_progress(
                task_id=task_id,
                stage=TaskStage.COMPLETED,
                progress=100,
                total=100,
                message="Processing complete",
            )
            logger.info("Task %s: processing completed", task_id)
        except TaskManagerError:
            raise
        except Exception as e:
            error_msg = str(e)
            logger.exception("Task %s: processing failed: %s", task_id, error_msg)
            await self._broadcast_error(task_id, error_msg)
            raise TaskManagerError(message=f"Task processing failed: {error_msg}", task_id=task_id) from e

    def _get_output_basename(self, task_id: str, task_status: TaskStatus) -> str:
        if task_status.files:
            name = (task_status.files[0].name or "").strip()
            if name:
                stem = Path(name).stem.strip()
                if stem:
                    return stem
        return task_id

    async def _update_task_outputs(self, task_id: str, outputs: dict[str, str]) -> None:
        task_key = self._get_task_key(task_id)
        try:
            task_data = await self.redis_client.get(task_key)
            if not task_data:
                return
            task_status = TaskStatus.model_validate_json(task_data)
            task_status.outputs = outputs
            task_status.updated_at = datetime.now()
            await self.redis_client.set(task_key, task_status.model_dump_json())
        except Exception as e:
            logger.warning("Failed to update outputs for task %s: %s", task_id, e)

    async def _upload_files_to_mineru(self, task_id: str, task_status: TaskStatus) -> str:
        if self.mineru_client is None:
            raise TaskManagerError(message="MinerU client not configured", task_id=task_id)
        if self.file_storage is None:
            raise TaskManagerError(message="File storage not configured", task_id=task_id)
        try:
            upload_response = await self.mineru_client.request_upload_urls(task_status.files)
        except Exception as e:
            raise TaskManagerError(message=f"Failed to get upload URLs: {e}", task_id=task_id) from e
        for i, (file_info, upload_url) in enumerate(zip(task_status.files, upload_response.file_urls)):
            file_path = self.file_storage.get_upload_path(task_id, file_info.name)
            if file_path is None:
                raise TaskManagerError(message=f"File not found: {file_info.name}", task_id=task_id)
            with open(file_path, "rb") as f:
                content = f.read()
            try:
                await self.mineru_client.upload_file(upload_url, content)
            except Exception as e:
                raise TaskManagerError(message=f"Failed to upload {file_info.name}: {e}", task_id=task_id) from e
            await self.update_progress(task_id=task_id, stage=TaskStage.UPLOADING, progress=i + 1, total=len(task_status.files), message=f"Uploaded {file_info.name}")
        return upload_response.batch_id

    async def _download_results(self, task_id: str, batch_result: BatchResultResponse) -> None:
        if self.file_storage is None:
            raise TaskManagerError(message="File storage not configured", task_id=task_id)
        for i, result in enumerate(batch_result.extract_result):
            if result.full_zip_url:
                try:
                    await self.file_storage.download_and_extract_zip(result.full_zip_url, task_id)
                except Exception as e:
                    raise TaskManagerError(message=f"Failed to download result for {result.file_name}: {e}", task_id=task_id) from e
            await self.update_progress(task_id=task_id, stage=TaskStage.DOWNLOADING, progress=i + 1, total=len(batch_result.extract_result), message=f"Downloaded {result.file_name}")
