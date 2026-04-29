# Background Face Blurring & Privacy Masking

## Project Overview
This Python-based computer vision pipeline automates privacy protection in video sequences. The tool identifies and masks faces with a specialized focus on maintaining the visibility of primary subjects while ensuring background anonymity through configurable thresholds.

## Core Features
- Automated Face Detection: Optimized for varying lighting conditions.
- Threshold-Based Blurring: A logic-driven "Group Privacy" mode. If the system detects more than two faces in a frame, it triggers a global foreground blur to ensure comprehensive anonymity.
- Privacy Masking: Robust layers that isolate detected regions from the original frame.
- Environmental Adaptation: Tuned to address challenges like motion blur and lower-resolution frames common in mobile or robotic capture.

## Technical Constraints & Limitations
- Detection Efficacy: The face detector’s reliability is closely tied to image quality. In instances of significant motion blur or low-resolution frames, the detector may fail to register background faces.
- Masking Omissions: Some background faces may remain unblurred if they bypass the initial detection phase due to environmental factors.
- Threshold Logic: To prevent identity leaks in crowded scenes, the system applies a global blur once the foreground count exceeds 2.

## Project Structure
- data/       : Sample images and masks.
- src/        : Core detection and blurring scripts.
- results/    : Output video sequences and processed frames.
- .gitignore  : Configured to exclude large media and binaries.

## Usage
1. Install dependencies: pip install -r requirements.txt
2. Run the pipeline: python main.py --input path/to/video.mp4 --threshold 2

## Research Context
This tool is part of an ongoing exploration into autonomous navigation and scene reconstruction. Ensuring the privacy of researchers and bystanders within captured datasets is a critical requirement for the ethical deployment of these computer vision systems.
