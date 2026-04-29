# Background Face Blurring & Privacy Masking

## Project Overview
This Python-based computer vision pipeline automates privacy protection in video sequences. The tool identifies and masks faces with a specialized focus on maintaining the visibility of primary subjects while ensuring background anonymity through configurable thresholds.

## Core Features
- Automated Face Detection: Optimized for varying lighting conditions.
- Threshold-Based Blurring: A logic-driven "Group Privacy" mode. If the system detects more than two faces in a frame, it triggers a global foreground blur to ensure comprehensive anonymity.
- Privacy Masking: Robust layers that isolate detected regions from the original frame.
- Environmental Adaptation: Tuned to address challenges like motion blur and lower-resolution frames common in mobile or robotic capture.

