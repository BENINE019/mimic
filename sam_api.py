# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import base64
import json
import mimetypes
import os
from collections import defaultdict
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from contextlib import aclosing
from pathlib import Path
from typing import Any

from meta_sam_parser import (
    ImageSegmentationResult,
    ParsedResponsesStream,
    ResponsesEventLike,
    SegmentationResult,
    VideoSegmentationResult,
    VideoSegmentationSnapshot,
    decode_mask_to_raster,
    decode_mask_to_svg_path,
    image_segmentation_format,
    parse_responses_stream,
    video_segmentation_format,
)
from openai import AsyncOpenAI

BASE_URL = "https://api.meta.ai/v1"
MODEL = "sam-3.1"


def client() -> AsyncOpenAI:
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        raise RuntimeError(
            "Set MODEL_API_KEY before calling the Segment Anything Model API."
        )
    return AsyncOpenAI(base_url=BASE_URL, api_key=key)


def media_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed is None:
        raise ValueError(f"Could not infer a media type for {path}.")
    return guessed


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type(path)};base64,{encoded}"


def response_input(prompt: str, media: dict[str, str]) -> list[dict[str, Any]]:
    phrase = prompt.strip()
    if not phrase:
        raise ValueError("The concept must be a short noun phrase.")
    return [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": phrase}, media],
        }
    ]


async def stream_events(stream: AsyncIterable[Any]) -> AsyncIterator[ResponsesEventLike]:
    """Convert the SDK's typed events to the plain Responses-event dicts the parser
    consumes, and finalize each output-text lane. The API streams
    `response.content_part.done` but not `response.output_text.done`, which the
    parser uses to close a lane, so synthesize it from the content part."""
    completed_lanes: set[tuple[object, object, object]] = set()
    async for event in stream:
        data = event.model_dump() if hasattr(event, "model_dump") else dict(event)
        lane = (
            data.get("item_id"),
            data.get("output_index"),
            data.get("content_index"),
        )
        if data.get("type") == "response.output_text.done":
            if lane not in completed_lanes:
                completed_lanes.add(lane)
                yield data
            continue
        yield data
        part = data.get("part")
        if (
            data.get("type") == "response.content_part.done"
            and lane not in completed_lanes
            and isinstance(part, dict)
            and part.get("type") == "output_text"
            and isinstance(part.get("text"), str)
        ):
            completed_lanes.add(lane)
            yield {
                "type": "response.output_text.done",
                "item_id": data.get("item_id"),
                "output_index": data.get("output_index"),
                "content_index": data.get("content_index"),
                "text": part["text"],
            }


async def parse_image_events(
    events: AsyncIterable[ResponsesEventLike],
) -> ImageSegmentationResult:
    return await parse_responses_stream(
        events, image_segmentation_format()
    ).final_result()


async def segment_image(path: Path, concept: str) -> ImageSegmentationResult:
    # Stream the response and let the parser decode the segmentation grammar.
    async with client() as api:
        stream = await api.responses.create(
            model=MODEL,
            stream=True,
            input=response_input(
                concept, {"type": "input_image", "image_url": image_data_url(path)}
            ),
        )
        # The parser returns on the terminal event, so stream_events is still
        # suspended over the response body. Close it here, before the response,
        # or the loop's asyncgen finalizer aclose()s a generator that is running.
        async with stream, aclosing(stream_events(stream)) as events:
            return await parse_image_events(events)


async def upload_video(path: Path) -> str:
    async with client() as api:
        uploaded = await api.files.create(
            file=(path.name, path.read_bytes(), media_type(path)),
            purpose="user_data",
        )
    file_id = uploaded.id
    if not isinstance(file_id, str) or not file_id.startswith("file-"):
        raise ValueError("The Files API returned an invalid file ID.")
    return file_id


async def report_snapshots(
    parsed: ParsedResponsesStream[VideoSegmentationSnapshot, VideoSegmentationResult],
) -> None:
    async for snapshot in parsed:
        if snapshot.revision == 1 or snapshot.revision % 30 == 0:
            print(
                f"revision={snapshot.revision} records={len(snapshot.records)}",
                flush=True,
            )


async def segment_video(file_id: str, concept: str) -> VideoSegmentationResult:
    if not file_id.startswith("file-"):
        raise ValueError("Expected a Files API ID beginning with 'file-'.")
    async with client() as api:
        stream = await api.responses.create(
            model=MODEL,
            stream=True,
            input=response_input(concept, {"type": "input_video", "file_id": file_id}),
        )
        # See segment_image: close the event generator before the response.
        async with stream, aclosing(stream_events(stream)) as events:
            parsed = parse_responses_stream(events, video_segmentation_format())
            # Claim final_result() inside the context manager: aclose() rejects
            # the result of a stream the consumer has not finished draining.
            async with parsed:
                await report_snapshots(parsed)
                return await parsed.final_result()


def validate_result(result: SegmentationResult) -> None:
    if result.outcome.status != "completed":
        reason = getattr(result.outcome, "reason", "unknown")
        detail = getattr(result.outcome, "detail", None)
        raise RuntimeError(f"Segmentation was incomplete: {reason} ({detail})")
    errors = [item for item in result.diagnostics if item.severity == "error"]
    if errors:
        messages = "; ".join(f"line {item.line}: {item.message}" for item in errors)
        raise RuntimeError(f"The parser rejected records: {messages}")


def summarize_result(result: SegmentationResult, concept: str) -> dict[str, Any]:
    boxes = [record for record in result.records if record.kind == "box"]
    masks = [record for record in result.records if record.kind == "mask"]
    tracks: dict[str, list[int]] = defaultdict(list)
    for record in (*boxes, *masks):
        if record.frame is not None:
            tracks[record.object_id].append(record.frame.frame_index)

    return {
        "concept": concept,
        "media": result.media,
        "outcome": result.outcome.status,
        "records": len(result.records),
        "boxes": [
            {
                "object_id": record.object_id,
                "frame": None if record.frame is None else record.frame.frame_index,
                "left": record.left,
                "top": record.top,
                "right": record.right,
                "bottom": record.bottom,
            }
            for record in boxes
        ],
        "masks": [
            {
                "object_id": record.object_id,
                "frame": None if record.frame is None else record.frame.frame_index,
                "encoding": record.mask.encoding,
                "width": record.mask.width,
                "height": record.mask.height,
                "foreground_pixels": sum(decode_mask_to_raster(record.mask)),
            }
            for record in masks
        ],
        "tracks": [
            {
                "object_id": object_id,
                "first_frame": min(frames),
                "last_frame": max(frames),
                "visible_frames": len(set(frames)),
            }
            for object_id, frames in sorted(
                tracks.items(), key=lambda item: int(item[0])
            )
        ],
        "diagnostics": [
            {
                "severity": item.severity,
                "code": item.code,
                "line": item.line,
                "message": item.message,
            }
            for item in result.diagnostics
        ],
    }


def write_outputs(
    output: Path,
    results: Iterable[tuple[str, SegmentationResult]],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    mask_index = 0
    for concept, result in results:
        validate_result(result)
        summaries.append(summarize_result(result, concept))
        for record in result.records:
            if record.kind != "mask":
                continue
            svg_path = decode_mask_to_svg_path(record.mask)
            mask_path = output / f"mask-{mask_index:04d}.svg"
            mask_path.write_text(
                (
                    f'<svg xmlns="http://www.w3.org/2000/svg" '
                    f'viewBox="0 0 {record.mask.width} {record.mask.height}">\n'
                    f'  <path d="{svg_path}" fill="black"/>\n'
                    "</svg>\n"
                ),
                encoding="utf-8",
            )
            mask_index += 1

    (output / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(summaries)} result(s) and {mask_index} mask(s) to {output}")
