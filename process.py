import os
import torch
from utils import to_input
from align import get_aligned_face


def pre_process(reference_img, frame_dir):
    batch_tensors = []
    ref_img = get_aligned_face(reference_img)
    ref_input = to_input(ref_img)
    batch_tensors.append(ref_input)
    
    for fname in sorted(os.listdir(frame_dir)):
        path = os.path.join(frame_dir, fname)

        try:
            aligned_rgb_img = get_aligned_face(path)
            bgr_tensor_input = to_input(aligned_rgb_img)
        except Exception as e:
            print(f"Skipping {path} due to error")
            continue
        batch_tensors.append(bgr_tensor_input)
    batch_input = torch.cat(batch_tensors, dim=0)
    return batch_input


def prediction(model, batch_input):
    features = []
    feature, _ = model(batch_input)
    features.append(feature)
    similarity_scores = torch.cat(features) @ torch.cat(features).T
    total_len = similarity_scores[0].numel()
    count_above_05 = (similarity_scores[0] > 0.5).sum().item()
    if count_above_05>=3:
        return True, total_len, count_above_05
    else:
        return False, total_len, 0