import os
import glob
import shutil
import pandas as pd
from tqdm import tqdm
from image_extract import frame_extract
from utils import load_pretrained_model
from process import pre_process, prediction

# Boiler plate code

if __name__=='__main__':
    results = []
    log_file = "log.csv"
    model = load_pretrained_model('ir_50')
    ref_img_path = '/home/ajeet/Downloads/tom.png'#input("Enter the reference image path: ")
    vid_directory = '/home/ajeet/Downloads/vid'#input("Enter the vid directory")
    video_files = [f for f in os.listdir(vid_directory) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(vid_directory, video_file)
        if not os.path.exists(video_path):
            print(f"File does not exist, skipping: {video_file}")
            continue
        
        # Extract the frames and store the frames
        frame_extract(video_path, output_dir='./frames')
        
        # Preprocess the frames and do verification
        batch_input = pre_process(ref_img_path, frame_dir='./frames')
        matched, total_len, matched_count = prediction(model, batch_input)
        if matched:
            shutil.copy(video_path, './matched')
        else:
            shutil.copy(video_path, './unmatched')


        results.append({
                    "video": video_file,
                    "match": "Yes" if matched else "No",
                    "total_check": total_len,
                    "matched_frames": matched_count
                })
        # Clean the './frames'
        [os.remove(f) for f in glob.glob("frames/*.[jp][pn]g")]
    
    df = pd.DataFrame(results)
    df.to_csv(log_file, index=False)
    print(f"Log saved to: {log_file}")