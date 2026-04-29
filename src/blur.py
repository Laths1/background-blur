import cv2 as cv
import glob
import masking
import face_detection as fd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
from tqdm import tqdm
import threading
import time
import torch, gc


def showImage(image, title="Image"):
    cv.namedWindow(title, cv.WINDOW_NORMAL)
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.imwrite(title, image)
    cv.destroyAllWindows()
    

def showVideo(video):
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        cv.imshow('Blurred Video', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv.destroyAllWindows()

def showBoundingBox(image, box, foreground=True):
    x, y, w, h = box
    if foreground:
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return image

def hashFrames(frames):
    hash_map = {}
    seq = 1

    while frames.isOpened():
        ret, frame = frames.read()
        if not ret:
            break

        hash_map[seq] = frame
        seq += 1

    frames.release()
    cv.destroyAllWindows()

    return hash_map

def blur(image, box):
    # 1. Extract the coordinates
    x, y, w, h = box
    
    # SAFETY: Ensure coordinates are within image boundaries
    # This prevents the "110 vs 102" mismatch if the box goes off-screen
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(image.shape[1], x + w), min(image.shape[0], y + h)
    
    # 2. Extract ROI based on clipped coordinates
    roi = image[y1:y2, x1:x2]
    
    # Get the ACTUAL height and width of the slice
    real_h, real_w = roi.shape[:2]
    
    # 3. Create mask based on ACTUAL ROI size
    mask = np.zeros((real_h, real_w), dtype=np.uint8)
    
    # 4. Draw the ellipse
    center = (real_w // 2, real_h // 2)
    axes = (real_w // 2, real_h // 2)
    cv.ellipse(mask, center, axes, 0, 0, 360, (255), -1)
    
    # 5. Create blurred version
    # Note: ksize must be odd. 51 is fine.
    blurred_roi = cv.GaussianBlur(roi, (21, 21), 30)
    
    # 6. Blend
    # Using cv.bitwise_and/or is often faster/safer than np.where for images
    mask_stack = cv.merge([mask, mask, mask])
    
    # np.where works fine now because shapes are guaranteed to match
    inv_mask = cv.bitwise_not(mask_stack)
    output_roi = cv.bitwise_and(blurred_roi, mask_stack) + cv.bitwise_and(roi, inv_mask)
    
    # 7. Place back
    image[y1:y2, x1:x2] = output_roi
    
    return image

def createAllMasks():
    img_folder = r'C:\Users\lathi\Documents\projects\background blur\crowd\*.jpg'
    i = 1
    all_images = glob.glob(img_folder) 
    num_of_img = len(all_images)
    for path in all_images:
        try:
            img = cv.imread(path)

            if img is None:
                continue
            
            print(f"Getting image {i}/{num_of_img} mask")
            mask = masking.Mask.modnet(img, path)

            masking.Mask.createImageMask(img, mask, path)

        except Exception as e:
            print(f"Error processing image {i}: {e}")

        i += 1

def maskBlur(img, msk, thresh=3, Foreground=True):
    # get masked image
    masked_image = masking.Mask.labelImage(img, msk, foreground=Foreground)

    if Foreground:
        boxes, num_of_faces = fd.FaceDetector.YuNet(masked_image, Foreground=Foreground)
    else:
        boxes = fd.FaceDetector.YuNet(masked_image)

    if boxes is not None:
        if Foreground:
            if num_of_faces >= thresh:
                for box in boxes:
                    img = blur(img, box)
                    img = showBoundingBox(img, box)
            else:
                pass
        else:
            for box in boxes:
                img = blur(img, box)
                img = showBoundingBox(img, box, foreground=False)

    return img, masked_image

def process_frame_batch(item):
    frame_id, frame = item
    processed_img, background_mask, foreground_mask = blurFrame(frame, masking.Mask.modnet) 
    return (frame_id, processed_img, background_mask, foreground_mask)

def blurFrame(image_path, model=masking.Mask.modnet):
    if isinstance(image_path, str):
        image_path = cv.imread(image_path)
    
    mask = model(image_path)

    # 2. handle background
    background_blurred, background_mask = maskBlur(image_path, mask, Foreground=False)

    # 3. handle foreground
    foreground_blurred, foreground_mask = maskBlur(background_blurred, mask)

    # 4. return final image
    return foreground_blurred, background_mask, foreground_mask
 

def blurVideo(video_path_str, file_name='output.mp4', back_mask='output_back_mask.mp4', fore_mask='output_fore_mask.mp4'):
    # 1. Drive & Folder Setup
    
    base_drive = '/content/drive/MyDrive/background_blur'

    # Create the full paths for Drive
    drive_final_path = os.path.join(base_drive, 'blurred', os.path.basename(file_name))
    drive_back_path = os.path.join(base_drive, 'back_mask', os.path.basename(back_mask))
    drive_fore_path = os.path.join(base_drive, 'fore_mask', os.path.basename(fore_mask))

    # os.makedirs(os.path.dirname(drive_final_path), exist_ok=True)
    # os.makedirs(os.path.dirname(drive_back_path), exist_ok=True)
    # os.makedirs(os.path.dirname(drive_fore_path), exist_ok=True)

    vid_path = cv.VideoCapture(video_path_str)
    if not vid_path.isOpened():
        print(f"Error: Could not open {video_path_str}")
        return

    fps = vid_path.get(cv.CAP_PROP_FPS) or 30.0
    width = int(vid_path.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(vid_path.get(cv.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(vid_path.get(cv.CAP_PROP_FRAME_COUNT))
    fourcc = cv.VideoWriter_fourcc(*'mp4v')

    out_final = cv.VideoWriter(drive_final_path, fourcc, fps, (width, height))
    out_back_mask = cv.VideoWriter(drive_back_path, fourcc, fps, (width, height))
    out_fore_mask = cv.VideoWriter(drive_fore_path, fourcc, fps, (width, height))

    batch_size = 100
    start_time = time.time()

    print(f"Processing: {os.path.basename(video_path_str)}")

    with ThreadPoolExecutor() as executor:
        while vid_path.isOpened():
            batch_frames = []
            for _ in range(batch_size):
                ret, frame = vid_path.read()
                if not ret: break
                batch_frames.append(frame)

            if not batch_frames: break

            results = list(tqdm(
            executor.map(process_frame_batch, enumerate(batch_frames)),
            total=len(batch_frames),
            desc="Processing Batch",
            leave=False
            ))
            results.sort(key=lambda x: x[0])

            for _, processed, b_mask, f_mask in results:
                if processed.shape[1] != width or processed.shape[0] != height:
                    processed = cv.resize(processed, (width, height))

                out_final.write(processed)
                out_back_mask.write(b_mask)
                out_fore_mask.write(f_mask)

            del batch_frames, results
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    vid_path.release()
    out_final.release()
    out_back_mask.release()
    out_fore_mask.release()

    total_duration = time.time() - start_time
    print(f"--- Finished: {os.path.basename(video_path_str)} ---")
    print(f"Saved to Drive: {drive_final_path}")
    print(f"Total time: {total_duration/60:.2f} minutes")
    

if __name__ == "__main__":
    # 1. get mask
    # image_path = r'C:\Users\lathi\Documents\projects\background blur\crowd\273271-1f249000e0cb4b12_jpg.rf.9da81cf89c89fdd750c81c89e6d528df.jpg'
    # image_path = r'C:\Users\lathi\Documents\projects\background blur\crowd\273271-2b427000e2a2b025_jpg.rf.7ffbf1d397b74587cf86058339119805.jpg'
    # image_path = r'C:\Users\lathi\Documents\projects\background blur\crowd\273271-2ca970007afb499f_jpg.rf.1e9d11a6bf44cc65c5fa5fe60992dc14.jpg'
    # image_path = r'C:\Users\lathi\Documents\projects\background blur\crowd\273271-2dcfd0007a9c41b7_jpg.rf.a5cb227a5fba4ada471aa3efc733a96e.jpg'
    # image_path = r'C:\Users\lathi\Documents\projects\background blur\crowd\273271-6e73000fd2de4d0_jpg.rf.899955e01abb0a769d5e919cb8c2448a.jpg'
    image_path = r'C:\Users\lathi\Documents\projects\background blur\non crowd\273278-87b6c00011235246_jpg.rf.16da59ffb0287280bc8a32622574333d.jpg'

    im = cv.imread(image_path)
    back, _, _ = blurFrame(im)
    # _, back2, _ = blurFrame(im, masking.Mask.mediapipe)
    # _, back3, _ = blurFrame(im, masking.Mask.isnet)

    showImage(back, title="girl.jpg")
    # showImage(back2, title="Mediapipe Mask")
    # showImage(back3, title="IsNet Mask")
    
    # video_path = r'C:\Users\lathi\Documents\projects\background blur\caz_cpt.mp4'
    # print("Inputting video path")
    # video_path = r'C:\Users\lathi\Documents\projects\background blur\caz_peru.mp4'
    # blurVideo(cv.VideoCapture(video_path))