import argparse
import cv2
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO

from vision_assist.FrameProcessor import FrameProcessor
from vision_assist.MockCamera import MockCamera


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8n-seg.pt', help='path to weights file')
    parser.add_argument('--source', type=str, default='../videos/longer.MP4', help='video file path')
    parser.add_argument('--output', type=str, default="../results/", help='output directory')
    parser.add_argument('--process-fps', type=int, default=8, help='process frames per second')
    parser.add_argument('--verbose', action='store_true', default=False, help='print debug information')
    parser.add_argument('--debug', action='store_true', default=False, help='if debug enabled debug stuff happens')
    
    return parser.parse_args()


def main(weights: str | Path = 'yolov8n-seg.pt',
         source: str | Path = '../videos/longer.MP4',
         output: str | Path = '../results/',
         process_fps: int = 8,
         verbose: bool = False,
         debug: bool = False,
        ) -> None:
    """
    Main function to process the video and generate the output.
   
    Args:
        weights: Path to the weights file
        source: Path to the video file
        output: Output directory
        
    Returns:
        Nish
    """
    # Initialize the model and frame processor
    model = YOLO(f"{weights}").to("cuda")
    processor = FrameProcessor(model=model, verbose=verbose, debug=debug)
   
    # Mock camera for testing
    mock_cam = MockCamera(source, target_fps=30)
    
    # Setup output directory for saved frames
    save_dir = Path(output)
    frames_dir = save_dir / f"{Path(source).stem}_saved_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    frames_processed = 0
    frames_skipped = 0
    frames_saved = 0
    save_counter = 0  # Counter for multiple saves of the same frame
    processing_times = []
   
    try:
        frame_count = 0  # Initialize a counter for frames
        while mock_cam.isOpened():
            ret, frame = mock_cam.read()
            if not ret:
                break

            frame_count += 1  # Increment the frame counter

            # Process only every 15th frame
            if frame_count % 15 != 0:
                continue

            start_time = cv2.getTickCount()
            
            processed_frame = None
            instructions = None
            
            while processed_frame is None and instructions is None:
                if debug:
                    processed_frame, instructions = processor(frame)
                else:
                    instructions = processor(frame)
                    processed_frame = frame  # Use original frame in non-debug mode
                
                if instructions is None:
                    frames_skipped += 1
                    if verbose:
                        print(f"Frame {frames_processed} was skipped as it was too blurry, trying next frame")
                    ret, frame = mock_cam.read()
                    if not ret:
                        break

            if processed_frame is None or instructions is None:
                continue
                
            frames_processed += 1
            
            if debug:
                resized_processed_frame = cv2.resize(processed_frame, (576, 1024))
                
                # Display loop - stay on current frame until Enter is pressed
                while True:
                    cv2.imshow("Processed Frame", resized_processed_frame)
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('s'):
                        frame_path = frames_dir / f"frame_{frames_processed:04d}_{save_counter:02d}.png"
                        cv2.imwrite(str(frame_path), processed_frame)
                        frames_saved += 1
                        save_counter += 1
                        if verbose:
                            print(f"Saved frame to {frame_path}")
                    
                    elif key == 13:  # ASCII code for Enter
                        save_counter = 0  # Reset save counter for new frame
                        break
                    
                    elif key == ord('q'):
                        raise KeyboardInterrupt

            end_time = cv2.getTickCount()
            
            processing_time = (end_time - start_time) / cv2.getTickFrequency()
            # only append when we have outcomes, otherwise the processing time
            # will be lower than the actual time it takes to process the frame
            if instructions: processing_times.append(processing_time)
            print(f"Instructions: {instructions}")
            print(f"Processing time: {processing_time} seconds")
    except KeyboardInterrupt:
        if processing_times:
            avg_processing_time = sum(processing_times) / len(processing_times)
            print("\nProcessing summary:")
            print(f"Average processing time: {avg_processing_time} seconds")
            print(f"Frames processed: {frames_processed}")
            print(f"Frames skipped: {frames_skipped}")
            print(f"Frames saved: {frames_saved}")
        
        mock_cam.release()
        cv2.destroyAllWindows()
    finally:
        mock_cam.release()
        cv2.destroyAllWindows()
   

if __name__ == "__main__":
    opt = parse_opt()
    main(**vars(opt))