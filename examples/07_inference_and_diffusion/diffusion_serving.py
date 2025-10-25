# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
# Adapted from GVM project

import argparse
import logging
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from diffusers import StableDiffusion3Pipeline

LOG_LEVEL = "INFO"
DISABLE_PROGRESS_BAR = True


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored log levels."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"  # Reset color

    def format(self, record):
        # Extract filename and line number
        filename = record.filename
        lineno = record.lineno

        # Format timestamp as MM-DD HH:MM:SS
        timestamp = self.formatTime(record, "%m-%d %H:%M:%S")

        # Color the log level
        level_name = record.levelname
        if level_name in self.COLORS:
            colored_level = f"{self.COLORS[level_name]}{level_name}{self.RESET}"
        else:
            colored_level = level_name

        # Format: LEVEL MM-DD HH:MM:SS [filename:lineno] message
        log_message = (
            f"{colored_level} {timestamp} [{filename}:{lineno}] {record.getMessage()}"
        )

        return log_message


def setup_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)

    # Avoid adding multiple handlers if logger already exists
    if logger.handlers:
        return logger

    log_level = os.getenv("GVM_DIFFUSION_LOG_LEVEL", LOG_LEVEL)
    level = getattr(logging, log_level.upper())
    logger.setLevel(level)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Create colored formatter
    formatter = ColoredFormatter()
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    return logger


logger = setup_logger()

default_config = {
    "model_path": "stabilityai/stable-diffusion-3.5-medium",
    "batch_size": 1,
    "num_inference_steps": 20,
    "guidance_scale": 7.5,
    "output_dir": "diffusion_outputs",
}


class SDConfig:
    def __init__(
        self,
        model_path: str = default_config["model_path"],
        batch_size: int = default_config["batch_size"],
        num_inference_steps: int = default_config["num_inference_steps"],
        guidance_scale: float = default_config["guidance_scale"],
        output_dir: str = default_config["output_dir"],
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_path = model_path
        self.batch_size = batch_size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.output_dir = Path(output_dir)
        self.device = device
        self.torch_dtype = torch_dtype

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class InferenceRequest:
    request_id: str
    prompt: str
    arrival_time: float


@dataclass
class InferenceResult:
    request_id: str
    prompt: str
    arrival_time: float
    start_time: float
    end_time: float
    inference_duration: float
    queue_wait_time: float


class DiffusionInferenceServer:
    """Offline diffusion inference server that processes requests sequentially."""

    def __init__(self, config: SDConfig, save_images: bool = False):
        self.config = config
        self.save_images = save_images
        self.pipeline = None
        self.results = []
        self.shutdown_requested = False
        self.processed_requests = 0
        self.stream = torch.cuda.Stream()

    def init_pipeline(self):
        """Initialize the diffusion pipeline."""
        logger.info("Initializing diffusion pipeline...")
        self.pipeline: StableDiffusion3Pipeline = (
            StableDiffusion3Pipeline.from_pretrained(
                self.config.model_path, torch_dtype=self.config.torch_dtype
            )
        )
        self.pipeline.set_progress_bar_config(disable=DISABLE_PROGRESS_BAR)
        self.pipeline = self.pipeline.to(self.config.device)
        logger.info("Pipeline initialized successfully.")

    def load_requests_from_file(
        self, dataset_path: str, num_requests: int
    ) -> List[InferenceRequest]:
        """Load prompts from text file and create inference requests."""
        requests = []
        current_time = time.time()

        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    prompt = line.strip()
                    if prompt:  # Skip empty lines
                        # Convert \n escape sequences back to actual newlines
                        decoded_prompt = prompt.replace("\\n", "\n")
                        request = InferenceRequest(
                            request_id=f"req{idx + 1}",
                            prompt=decoded_prompt,
                            arrival_time=current_time,
                        )
                        requests.append(request)
                        if num_requests and len(requests) >= num_requests:
                            break

            logger.info(f"Loaded {len(requests)} requests from {dataset_path}")
            return requests

        except FileNotFoundError:
            logger.error(f"Request file '{dataset_path}' not found.")
            return []
        except Exception as e:
            logger.error(f"Error loading requests: {str(e)}")
            return []

    def process_request(self, request: InferenceRequest) -> InferenceResult:
        """Process a single inference request with detailed timing."""
        start_time = time.time()
        queue_wait_time = start_time - request.arrival_time

        logger.info(f"Processing {request.request_id}: {request.prompt[:50]}...")

        try:
            with torch.cuda.stream(self.stream):
                images = self.pipeline(
                    prompt=[request.prompt],
                    num_inference_steps=self.config.num_inference_steps,
                    guidance_scale=self.config.guidance_scale,
                ).images
            self.stream.synchronize()

            end_time = time.time()
            inference_duration = end_time - start_time

            # Save images if requested
            if self.save_images and images:
                for idx, image in enumerate(images):
                    output_path = (
                        self.config.output_dir / f"{request.request_id}_{idx}.png"
                    )
                    image.save(output_path)

            result = InferenceResult(
                request_id=request.request_id,
                prompt=request.prompt,
                arrival_time=request.arrival_time,
                start_time=start_time,
                end_time=end_time,
                inference_duration=inference_duration,
                queue_wait_time=queue_wait_time,
            )

            logger.info(f"Completed {request.request_id} in {inference_duration:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Error processing {request.request_id}: {str(e)}")
            # Return error result with timing data
            end_time = time.time()
            return InferenceResult(
                request_id=request.request_id,
                prompt=request.prompt,
                arrival_time=request.arrival_time,
                start_time=start_time,
                end_time=end_time,
                inference_duration=end_time - start_time,
                queue_wait_time=queue_wait_time,
            )

    def save_log(self, output_file: str):
        """Save timing results to CSV file."""
        if not self.results:
            logger.warning("No results to save.")
            return

        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Save log file in the output directory
        log_path = self.config.output_dir / output_file
        with open(log_path, "w", encoding="utf-8") as f:
            # Write simple text format: request_id inference_duration
            for result in self.results:
                f.write(f"{result.inference_duration:.3f}\n")

        logger.info(f"Timing log saved to {log_path}")

        # Print summary statistics
        total_requests = len(self.results)
        avg_inference_time = (
            sum(r.inference_duration for r in self.results) / total_requests
        )

        logger.info(f"{total_requests} requests processed")
        logger.info(f"Average inference time: {avg_inference_time:.2f}s")

    def signal_handler(self, signum, frame):
        logger.warning("\nReceived interrupt signal. Shutting down gracefully...")
        logger.info(f"Processed {self.processed_requests} requests so far.")
        self.shutdown_requested = True

    def run_server(self, dataset_path: str, num_requests: int, output_log: str):
        """Main server loop that processes all requests sequentially."""
        # Setup signal handler
        signal.signal(signal.SIGINT, self.signal_handler)

        # Initialize pipeline
        self.init_pipeline()

        # Load requests
        requests = self.load_requests_from_file(dataset_path, num_requests)
        if not requests:
            logger.error("No valid requests found. Exiting.")
            return

        logger.info(f"Starting to process {len(requests)} requests...")

        # Process requests sequentially
        for request in requests:
            if self.shutdown_requested:
                logger.warning("Shutdown requested. Stopping processing.")
                break

            result = self.process_request(request)
            self.results.append(result)
            self.processed_requests += 1

        # Save results
        logger.info("\nProcessing completed. Saving log...")
        self.save_log(output_log)

        if self.shutdown_requested:
            logger.info("Server shutdown gracefully.")
        else:
            logger.info("All requests processed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion Inference Server")

    # Required arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to text file containing prompts (one per line)",
    )
    parser.add_argument(
        "--num_requests",
        type=int,
        default=None,
        help="Number of requests to process",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="stats.txt",
        help="Output file for timing results",
    )

    # Model configuration arguments
    parser.add_argument("--model_path", type=str, default=default_config["model_path"])
    parser.add_argument(
        "--num_inference_steps", type=int, default=default_config["num_inference_steps"]
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=default_config["guidance_scale"]
    )
    parser.add_argument("--output_dir", type=str, default=default_config["output_dir"])
    parser.add_argument(
        "--save_images", action="store_true", help="Save generated images"
    )

    args = parser.parse_args()

    # Configure logging level
    setup_logger()

    # Initialize configuration
    config = SDConfig(
        model_path=args.model_path,
        batch_size=1,  # Always 1 for sequential processing
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        output_dir=args.output_dir,
    )

    # Run inference server
    server = DiffusionInferenceServer(config, save_images=args.save_images)
    server.run_server(args.dataset_path, args.num_requests, args.log_file)


if __name__ == "__main__":
    main()
