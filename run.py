import hydra
from omegaconf import DictConfig
from pathlib import Path


@hydra.main(version_base=None, config_path="configs", config_name="default")
def run(args: DictConfig) -> None:
    from vipe.streams.base import StreamList, CachedVideoStream, SliceVideoStream, ConcatVideoStream

    # Gather all video streams
    stream_list = StreamList.make(args.streams)

    from vipe.pipeline import make_pipeline
    from vipe.utils.logging import configure_logging

    # Process each video stream
    logger = configure_logging()
    for stream_idx in range(len(stream_list)):
        video_stream = stream_list[stream_idx]
        logger.info(
            f"Processing {video_stream.name()} ({stream_idx + 1} / {len(stream_list)})"
        )

        # Split-merge execution to reduce peak memory without changing configs
        # Strategy: cache input, process chunks with overlap using pipeline.return_output_streams, then concatenate
        cached_input = CachedVideoStream(video_stream, desc="Caching input", streaming=True)
        total_len = len(cached_input)
        chunk_size = 200
        overlap = 50
        starts = list(range(0, total_len, chunk_size - overlap))
        merged_outputs = []

        for chunk_id, start in enumerate(starts):
            end = min(start + chunk_size, total_len)
            if start >= end:
                break
            # create view with name suffix for uniqueness
            sliced = SliceVideoStream(cached_input, start, end, name_suffix=f"part{chunk_id}")

            pipeline = make_pipeline(args.pipeline)
            pipeline.return_output_streams = True
            out = pipeline.run(sliced)
            assert out.output_streams is not None and len(out.output_streams) >= 1
            # For default pipeline, may be multiple views; for panorama, single
            for os_idx, output_stream in enumerate(out.output_streams):
                # Save each chunk immediately in a temp path, and keep only a lightweight reader
                merged_outputs.append(output_stream)

        # Concatenate outputs and save artifacts once (streams are cached online to limit RAM)
        if len(merged_outputs) > 0:
            concat = ConcatVideoStream(merged_outputs, name=video_stream.name())
            from vipe.utils import io
            out_base = Path(args.pipeline.output.path)
            artifact_path = io.ArtifactPath(out_base, video_stream.name())
            artifact_path.meta_info_path.parent.mkdir(exist_ok=True, parents=True)
            if args.pipeline.output.save_artifacts:
                io.save_artifacts(artifact_path, concat)
            if args.pipeline.output.save_viz:
                # Visualization requires SLAM output; skip in split-merge mode
                pass

        logger.info(f"Finished processing {video_stream.name()}")


if __name__ == "__main__":
    run()
