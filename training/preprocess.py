import os
import cv2 
import torch
import random
import argparse
import numpy as np
import multiprocessing as mp

from typing import Tuple
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from einops import rearrange
from torchvision.transforms import v2


from chanjing_avatar.models.vae import Wan2_2_VAE
from chanjing_avatar.preprocess import ensure_landmarks
from transformers import Wav2Vec2FeatureExtractor
from chanjing_avatar.audio.wav2vec2 import Wav2Vec2Model


import librosa
import pyloudnorm as pyln


def has_complete_samples(output_dir):
    output_dir = Path(output_dir)
    for latent in output_dir.glob("latents_*.pth"):
        suffix = latent.name.removeprefix("latents_")
        if (output_dir / f"audio_{suffix}").is_file() and (output_dir / f"latents_y_{suffix}").is_file():
            return True
    return False

class VideoAudioDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.frame_interval = args.frame_interval
        self.min_num_frames = args.num_frames
        self.num_frames = args.num_frames
        self.device = args.device
            
        self.frame_process = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.vae = Wan2_2_VAE(
            vae_pth=args.vae_path,
            device=self.device,
            )

        self.all_dataset = []  # 记录所有训练片段的索引

        self.wav2vec_processor, self.wav2vec = self.custom_init(self.device, args.audio_encoder_path)

    def custom_init(self, device, wav2vec):    
        audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec, local_files_only=True).to(device)
        audio_encoder.feature_extractor._freeze_parameters()
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec, local_files_only=True)
        return wav2vec_feature_extractor, audio_encoder

    def get_device_id(self):
        """
        从self.device中提取设备ID
        例如: 'cuda:0' -> '0', 'cuda:1' -> '1'
        """
        if 'cuda:' in self.device:
            return self.device.split(':')[1]
        else:
            return 0  # 默认使用GPU 0

    def loudness_norm(self, audio_array, sr=16000, lufs=-23):
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio_array)
        if abs(loudness) > 100:
            return audio_array
        normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
        return normalized_audio

    def get_audio_embedding(self, speech_array, fps, sr=16000):
        audio_duration = len(speech_array) / sr
        video_length = audio_duration * fps

        # wav2vec_feature_extractor
        audio_feature = np.squeeze(
            self.wav2vec_processor(speech_array, sampling_rate=sr).input_values
        )
        audio_feature = torch.from_numpy(audio_feature).float().to(self.device)
        audio_feature = audio_feature.unsqueeze(0)

        # audio encoder
        with torch.no_grad():
            embeddings = self.wav2vec(audio_feature, seq_len=int(video_length), output_hidden_states=True)

        if len(embeddings) == 0:
            print("Fail to extract audio embedding")
            return None

        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "b s d -> s b d")

        audio_emb = audio_emb.cpu().detach()
        return audio_emb
    
    def segment_bbox_robust(
        self,
        landmarks_seg: np.ndarray,
        q_low: float = 0.03,
        q_high: float = 0.97,
    ) -> Tuple[float, float, float, float]:
        xs = landmarks_seg[:, :, 0].reshape(-1)
        ys = landmarks_seg[:, :, 1].reshape(-1)

        xmin = np.quantile(xs, q_low)
        xmax = np.quantile(xs, q_high)
        ymin = np.quantile(ys, q_low)
        ymax = np.quantile(ys, q_high)

        return float(xmin), float(ymin), float(xmax), float(ymax)

    def temporal_smooth(self, landmarks, win=5):
        # landmarks: (T, N, 2)
        kernel = np.ones(win) / win
        out = landmarks.copy()
        for i in range(landmarks.shape[1]):
            for d in range(2):
                out[:, i, d] = np.convolve(
                    landmarks[:, i, d], kernel, mode="same"
                )
        return out

    def design_box2_for_segment(self,
        landmarks_seg,
        width: int,
        height: int,
        margin: int = 32,
    ) -> Tuple[float, float, float, float]:
        """
        为一个片段设计统一的 box2，返回 (x_min, y_min, x_max, y_max)。

        步骤：
        1. 对片段所有帧求聚合 bbox（片段级 box1）；
        2. 取最长边，构造正方形并放大一倍；
        3. 再在四周加入 margin，得到更大的正方形；
        4. 尝试在画面内平移以保持方形，不行则退化为矩形。
        """
        landmarks = self.temporal_smooth(landmarks_seg, win=5)
        pxmin, pymin, pxmax, pymax = self.segment_bbox_robust(landmarks)

        w = pxmax - pxmin
        h = pymax - pymin
        side_base = max(w, h)

        # 正方形放大一倍
        side = side_base * 2.0

        # 加上安全边距（四周各 margin）
        side += 2 * margin

        # 以片段中心为中心
        cx = (pxmin + pxmax) / 2.0
        cy = (pymin + pymax) / 2.0

        x_min = cx - side / 2.0
        x_max = cx + side / 2.0
        y_min = cy - side / 2.0
        y_max = cy + side / 2.0

        # 尝试保持正方形且不越界
        if side <= width:
            x_min = max(0.0, min(x_min, float(width) - side))
            x_max = x_min + side
        else:
            # 无法保持方形，改为 [0, width]
            x_min = 0.0
            x_max = float(width)

        if side <= height:
            y_min = max(0.0, min(y_min, float(height) - side))
            y_max = y_min + side
        else:
            y_min = 0.0
            y_max = float(height)

        # 最终 clip，避免浮点误差
        x_min = max(0.0, min(x_min, float(width)))
        x_max = max(0.0, min(x_max, float(width)))
        y_min = max(0.0, min(y_min, float(height)))
        y_max = max(0.0, min(y_max, float(height)))

        return int(x_min), int(y_min), int(x_max), int(y_max)  

    def design_box2_for_clips_landmarks(self,
        clips,
        landmarks,
        ref_clip_map, 
        ref_clip_map_list,
        width: int,
        height: int
    ):
        """
        为每一个片段设计统一的 box2，返回 (x_min, y_min, x_max, y_max)。
        """
        margin = 32
        box2_list = []
        for i in range(len(clips)):
            if i == 0 and len(clips) == 1: 
                start_idx = clips[i][0]
                end_idx = clips[i][1] + 1
            elif i == 0:
                start_idx = clips[i][0]
                end_idx = clips[i][1] + margin + 1
            elif i == len(clips) - 1 :
                start_idx = clips[i][0] - margin
                start_idx = max(0, start_idx)
                end_idx = clips[i][1] + 1
            else:
                start_idx = clips[i][0] - margin
                start_idx = max(0, start_idx)
                end_idx = clips[i][1] + margin + 1
            end_idx = min(end_idx, len(landmarks))

            min_idx = min(ref_clip_map_list[i].keys())
            max_idx = max(ref_clip_map_list[i].keys())
            if min_idx < start_idx or max_idx >= end_idx:
                return None, None, None

            box2 = self.design_box2_for_segment(landmarks[start_idx:end_idx], width, height)
            box2_list.append(box2)
            keys = list(ref_clip_map_list[i].keys())
            for key in keys:
                ref_clip_map_list[i][key]["box"] = box2 

            map_key = list(ref_clip_map[i].keys())[0]
            ref_clip_map[i][map_key]["box"] = box2

        return box2_list, ref_clip_map, ref_clip_map_list

    def split_frames_into_clips(self, total_frames: int, num_frames: int, frame_interval: int):
        """
        返回:
        - clips: List[Tuple[start, end]]，闭区间，0-based
        - clip_randoms: List[Tuple[int, int]]，每个 clip 对应两个随机帧索引

        clip_randoms 规则（按你的“完整说明”）：
        - 首段：
            - 如果有第二段：在“段后 25 帧内”随机取两个（即 [e+1, e+25] 与全局边界取交集）
            - 如果没有第二段：在段内随机取两个（[s, e]）
        - 末段：
            - 如果有前段：在“段前 25 帧内”随机取两个（即 [s-25, s-1] 与全局边界取交集）
            - 如果没有前段：在段内随机取两个（[s, e]）
        - 中间段：段前25取1个 + 段后25取1个
        """
        last_tag = False
        if total_frames < num_frames:
            return [], [], last_tag

        # 1) 构建 clips（按 stride=num_frames，不重叠分段）
        stride = num_frames
        clips = []
        start = 0
        while start + num_frames <= total_frames:
            end = start + num_frames - 1
            clips.append((start, end))
            start += stride

        # 2) 末尾补一段：剩余帧数 > frame_interval 时强制补最后 num_frames
        remaining = total_frames - (clips[-1][1] + 1) if clips else total_frames
        if remaining > frame_interval:
            tail = (max(0, total_frames - num_frames), total_frames - 1)
            if not clips or clips[-1] != tail:
                clips.append(tail)
                last_tag = True

        # 3) 采样工具：严格在给定区间与全局边界交集内采样
        def sample_k(lo: int, hi: int, k: int):
            lo = max(0, lo)
            hi = min(total_frames - 1, hi)
            if lo > hi:
                # 区间为空时的兜底：就钳到合法点
                p = max(0, min(total_frames - 1, lo))
                return [p] * k
            pool = list(range(lo, hi + 1))
            if len(pool) >= k:
                return random.sample(pool, k)
            if len(pool) == 1:
                return [pool[0]] * k
            return [random.choice(pool) for _ in range(k)]

        # 4) 构建 clip_randoms（严格按规则）
        clip_randoms = []
        n = len(clips)
        margin = 32

        for i, (s, e) in enumerate(clips):
            is_first = (i == 0)
            is_last = (i == n - 1)

            if n == 1:
                # 只有一个片段：段内 3 帧，复制 3 次得到 9 帧
                picks = sample_k(s, e, 3)

            elif is_first:
                # 首段且 n>1：有第二段，段内 3 帧 + 段后 40 帧内 6 帧
                in_seg = sample_k(s, e, 2)
                after_seg = sample_k(e + 1, e + margin, 3)
                picks = in_seg + after_seg

            elif is_last:
                # 末段且 n>1：有前段，段前 40 帧内 6 帧 + 段内 3 帧
                before_seg = sample_k(s - margin, s - 1, 3)
                in_seg = sample_k(s, e, 2)
                picks = before_seg + in_seg

            else:
                # 中间段：段前 3 帧 + 段内 3 帧 + 段后 3 帧
                before_seg = sample_k(s - margin, s - 1, 2)
                in_seg = sample_k(s, e, 1)
                after_seg = sample_k(e + 1, e + margin, 2)
                picks = before_seg + in_seg + after_seg

            if len(picks) != 5 and len(picks) !=3:
                print(f"picks must have 9 or 3 elements, got {len(picks)}" )
                return [], [], last_tag
            clip_randoms.append(tuple(picks))

        return clips, clip_randoms, last_tag


    def process_latent_x(self, frames, landmarks, box2):
        x_min, y_min, x_max, y_max = box2
        w = x_max - x_min
        h = y_max - y_min
        # 1. 选取与原裁剪后长宽比最接近的目标尺寸
        candidates = [(512, 512), (448, 576), (576, 448)]
        orig_ratio = w / h
        target_w, target_h = min(
            candidates,
            key=lambda wh: abs((wh[0] / wh[1]) - orig_ratio)
        )
        # 2. 按目标尺寸等比缩放，不超过目标宽/高
        scale = min(target_w / w, target_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2

        resize_frames = []
        resize_landmarks = []
        for frame, lmk in zip(frames, landmarks):
            # 3. 创建目标画布，高不够就上下填充，宽不够就左右填充（居中）
            resized = cv2.resize(frame, (new_w, new_h))
            canvas = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
            frame = canvas
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.frame_process(Image.fromarray(frame))
            resize_frames.append(frame)

            lmk_c = lmk.copy()
            # 先从整图坐标变成裁剪框内部坐标
            lmk_c[..., 0] = lmk_c[..., 0] - x_min
            lmk_c[..., 1] = lmk_c[..., 1] - y_min
            # 等比缩放
            lmk_c[..., 0] = lmk_c[..., 0] * scale
            lmk_c[..., 1] = lmk_c[..., 1] * scale
            # 再加上在目标画布中的偏移（居中）
            lmk_c[..., 0] = lmk_c[..., 0] + x_offset
            lmk_c[..., 1] = lmk_c[..., 1] + y_offset

            resize_landmarks.append(lmk_c) 
        
        resize_frames = torch.stack(resize_frames, dim=0)
        resize_frames = rearrange(resize_frames, "T C H W -> C T H W")
        resize_frames = resize_frames.to(dtype=torch.float32, device=self.device)
        resize_frames = resize_frames.unsqueeze(0)

        with torch.no_grad():
            latents = self.vae.encode(resize_frames)[0]
        return latents.cpu(), resize_landmarks

    def process_latent_y(self, frames):
        h, w = frames[0].shape[:2]
        # 1. 选取与原裁剪后长宽比最接近的目标尺寸
        candidates = [(512, 512), (448, 576), (576, 448)]
        orig_ratio = w / h
        target_w, target_h = min(
            candidates,
            key=lambda wh: abs((wh[0] / wh[1]) - orig_ratio)
        )
        # 2. 按目标尺寸等比缩放，不超过目标宽/高
        scale = min(target_w / w, target_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2

        resize_frames = []
        for frame in frames:
            # 3. 创建目标画布，高不够就上下填充，宽不够就左右填充（居中）
            resized = cv2.resize(frame, (new_w, new_h))
            canvas = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
            frame = canvas
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.frame_process(Image.fromarray(frame))
            resize_frames.append(frame)

        
        resize_frames = torch.stack(resize_frames, dim=0)
        resize_frames = rearrange(resize_frames, "T C H W -> C T H W")
        resize_frames = resize_frames.to(dtype=torch.float32, device=self.device)
        resize_frames = resize_frames.unsqueeze(0)

        with torch.no_grad():
            latents = self.vae.encode(resize_frames)[0]
        return latents.cpu()

    
    def process_audio(self, audio_path, clips, total_video_frames, fps, output_dir, sample_rate=16000):
        """
        按 clips(视频帧索引) 分段处理音频，并返回每段对应的音频特征(已剔除前后 pad)。

        需求实现：
        1) 每段音频按视频帧 pad：前后各多取 self.num_frames 帧，不足则取到边界
        2) 对每段音频做 loudness_norm
        3) loudness_norm 后用 get_audio_embedding 编码
        4) 编码后剔除前后 pad 对应的特征帧（多取多少剔除多少；若音频尾部不够，会自动按实际可用帧数剔除）
        """
        audio, sr = librosa.load(audio_path, sr=sample_rate)

        pad_frames = int(self.num_frames)  # 这里就是 81
        samples_per_frame = sample_rate / float(fps)  # 25fps 时=640.0

        audio_len = len(audio)
        indices = (torch.arange(2 * 2 + 1) - 2) * 1 
        
        save_index = 0 
        for (s, e) in clips:
            # 1) 计算需要 pad 的帧数（不足就少取）
            left_pad = min(pad_frames, s)
            right_pad = min(pad_frames, (total_video_frames - 1) - e)

            ext_s = s - left_pad
            ext_e = e + right_pad + 1

            # 2) 视频帧索引 -> 音频采样点索引（切片用半开区间 [start, end)）
            start_sample = int(round(ext_s * samples_per_frame))
            end_sample = int(round(ext_e * samples_per_frame))

            start_sample = max(0, min(start_sample, audio_len))
            end_sample = max(start_sample, min(end_sample, audio_len))

            seg_audio = audio[start_sample:end_sample]
            if seg_audio.size == 0:
                return None

            # 3) loudness_norm
            seg_audio = self.loudness_norm(seg_audio, sr=sample_rate)

            # 4) 编码
            audio_emb = self.get_audio_embedding(seg_audio, fps=fps, sr=sample_rate)
            if audio_emb is None:
                return None

            # 5) 根据“实际可用音频长度”推断这段 embedding 覆盖了多少帧，用它修正 right_pad（避免尾部音频不够导致过度裁剪）
            ext_frames_eff = int(len(seg_audio) / samples_per_frame)  # floor：这段音频最多对应多少完整视频帧
            ext_e_eff = ext_s + ext_frames_eff - 1
            right_pad_eff = max(0, ext_e_eff - e)
            left_pad_eff = left_pad  # ext_s 一定 >=0（因为 left_pad<=s），左侧不会因 clamp 变短

            T = audio_emb.shape[0]
            trim_left = min(left_pad_eff, T)
            trim_right = min(right_pad_eff, max(0, T - trim_left))
            trim_right = T - trim_right
            if trim_right - trim_left <= self.num_frames - 3:
                return None
            else:
                trim_right = trim_left + self.num_frames
            center_indices = torch.arange(
                    trim_left,
                    trim_right,
                    1,
                ).unsqueeze(
                    1
                ) + indices.unsqueeze(0)
            center_indices = torch.clamp(center_indices, min=0, max=audio_emb.shape[0]-1)
            audio_clip = audio_emb[center_indices][None,...].squeeze(0)

            tensor_filename = f"audio_{save_index+1:03d}.pth"
            tensor_path = os.path.join(output_dir, tensor_filename)
            torch.save(audio_clip.cpu(), tensor_path)
            save_index += 1
        return True

    def run_getitem(self, video_path, audio_path, lmk_path, output_dir):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")    
        fps = cap.get(cv2.CAP_PROP_FPS)
        if abs(fps - 25.0) > 0.01:
            raise ValueError(f"Video FPS must be 25, got {fps:.3f}: {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        landmarks = np.load(lmk_path)
        if len(landmarks) < self.num_frames:
            return False

        clips, clip_randoms, last_tag = self.split_frames_into_clips(len(landmarks), self.num_frames, self.frame_interval)
        
        ref_clip_map = {}
        ref_clip_map_list = {i:{} for i in range(len(clips))}

        ref_list = []
        ref_clip_list = []
        for i in range(len(clips)):
            clip_idx = clips[i]
            clip_rand_idx_list = list(clip_randoms[i])
            rand_idx_list = clip_rand_idx_list + [clip_idx[0]]
            ref_clip_idx = random.choices(rand_idx_list, k=1)[0]
            ref_list.append(ref_clip_idx)
            ref_clip_list += clip_rand_idx_list
            ref_clip_map.update({i:{ref_clip_idx:{"box": None, "frame": None}}})
            for r_idx in clip_rand_idx_list:
                ref_clip_map_list[i][r_idx] = {"box": None, "frame": None}
        
        ref_list = list(set(ref_list))
        ref_clip_list = list(set(ref_clip_list))

        box2_list, ref_clip_map, ref_clip_map_list = self.design_box2_for_clips_landmarks(clips, landmarks, ref_clip_map, ref_clip_map_list, width, height)

        if box2_list is None:
            return False

        frame_idx = 0
        clip_index = 0 
        
        l_frames = []
        m_frames = []
        if last_tag:
            last_start_idx = clips[-1][0]
            last_end_idx = clips[-2][1]
            last_box2 = box2_list[-1]
            last_box2_width = last_box2[2] - last_box2[0]
            last_box2_height = last_box2[3] - last_box2[1]
            if last_box2_width < 240 or last_box2_height < 240:
                print(f"Error: last_box2_width < 240 or last_box2_height < 240")
                return False
            last_x_min, last_y_min, last_x_max, last_y_max = last_box2
        
        start_idx, end_idx = clips[0][0], clips[0][1] + 1
        box2 = box2_list[0]
        box2_width = box2[2] - box2[0]
        box2_height = box2[3] - box2[1]
        if box2_width < 240 or box2_height < 240:
            print(f"Error: box2_width < 240 or box2_height < 240")
            return False
        
        x_min, y_min, x_max, y_max = box2
        save_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if last_tag:
                if frame_idx >= last_start_idx and frame_idx <= last_end_idx:
                    l_frames.append(frame[last_y_min:last_y_max, last_x_min:last_x_max])

            if frame_idx >= end_idx:
                m_frames = [] 
                clip_index += 1
                if clip_index >= len(clips):
                    break
                start_idx, end_idx = clips[clip_index][0], clips[clip_index][1] + 1
                box2 = box2_list[clip_index]
                box2_width = box2[2] - box2[0]
                box2_height = box2[3] - box2[1]
                if box2_width < 240 or box2_height < 240:
                    print(f"Error: box2_width < 240 or box2_height < 240")
                    return False
                x_min, y_min, x_max, y_max = box2
                

            m_frames.append(frame[y_min:y_max, x_min:x_max]) 

            
            if frame_idx in ref_list:
                for i in range(len(clips)):
                    if frame_idx in list(ref_clip_map[i].keys()):
                        ref_box = ref_clip_map[i][frame_idx]['box']
                        ref_x_min, ref_y_min, ref_x_max, ref_y_max = ref_box
                        ref_clip_map[i][frame_idx]['frame'] = frame[ref_y_min:ref_y_max, ref_x_min:ref_x_max]
            
            if frame_idx in ref_clip_list:
                for i in range(len(clips)):
                    if frame_idx in list(ref_clip_map_list[i].keys()):
                        ref_box = ref_clip_map_list[i][frame_idx]['box']
                        ref_x_min, ref_y_min, ref_x_max, ref_y_max = ref_box
                        ref_clip_map_list[i][frame_idx]['frame'] = frame[ref_y_min:ref_y_max, ref_x_min:ref_x_max]
            
            if len(m_frames) == self.num_frames:
                frames = m_frames
            elif len(m_frames) + len(l_frames) == self.num_frames and clip_index == len(clips) - 1:
                frames = l_frames + m_frames
            else:
                frames = []
            if len(frames) == self.num_frames:
                latents, lmk203_list = self.process_latent_x(frames, landmarks[start_idx:end_idx], box2) 
                tensor_filename = f"latents_{save_index+1:03d}.pth"
                tensor_path = os.path.join(output_dir, tensor_filename)
                torch.save(latents, tensor_path)

                lmk203_list_path = os.path.join(output_dir, f"lmk203_{save_index+1:03d}.npy")
                np.save(lmk203_list_path, lmk203_list)
                del frames
                m_frames =[]
                save_index += 1 

            frame_idx += 1
        cap.release()

        for i in range(len(clips)):
            keys = list(ref_clip_map_list[i].keys())
            if len(keys) != 5 and len(keys) != 3:
                print(f"ref_clip_frames 必须是5或3帧, got {len(keys)}" )
                return False
            random.shuffle(keys)
            if len(keys) == 3:
                keys = keys * 13 + keys[:2]
            else:
                keys = keys * 8 + keys[:1]
            random.shuffle(keys)
            clip_frames = []
            for key in keys:
                ref_frame = ref_clip_map_list[i][key]['frame']
                clip_frames.append(ref_frame)
            
            latents_y = self.process_latent_y(clip_frames)
            tensor_filename_y = f"latents_y_{i+1:03d}.pth"
            tensor_path_y = os.path.join(output_dir, tensor_filename_y)
            torch.save(latents_y, tensor_path_y)

        if not self.process_audio(audio_path, clips, len(landmarks), fps, output_dir):
            return False

        return True



def process_videos_on_gpu(gpu_id, video_paths, process_idx, args):
    """
    在指定GPU上处理视频文件的多进程函数
    
    Args:
        gpu_id: GPU ID
        video_paths: 要处理的视频路径列表
        process_idx: 进程索引
        args: 参数对象
    """

    # 设置当前设备
    torch.cuda.set_device(gpu_id)
    DEVICE = f'cuda:{gpu_id}'
    
    # 清空GPU缓存
    torch.cuda.empty_cache()
    print(f"[进程{process_idx}] GPU {gpu_id} (rank {gpu_id}) 可用: {torch.cuda.get_device_name(gpu_id)}")

    # 更新args中的device
    args.device = DEVICE
    
    # 重新创建数据集实例，确保每个进程独立
    dataset = VideoAudioDataset(args)
    
    failed = 0
    for video_path in tqdm(video_paths, desc=f"[GPU {gpu_id}][进程{process_idx}]"):
        try:
            video_path = str(video_path)
            audio_path = Path(video_path).with_suffix(".wav")
            
            if not os.path.exists(audio_path):
                print(f"[GPU {gpu_id}][进程{process_idx}] 音频文件不存在: {audio_path}")
                failed += 1
                continue

            relative_video = Path(video_path).resolve().relative_to(Path(args.data_dir).resolve())
            output_dir = Path(args.save_dir).resolve() / relative_video.with_suffix("")
            source_lmk_path = Path(video_path).with_suffix(".npy")
            lmk_path = source_lmk_path if source_lmk_path.is_file() else output_dir / "landmarks.npy"
            if not lmk_path.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
                ensure_landmarks(
                    args.landmark_model_path,
                    video_path,
                    str(lmk_path),
                    device_id=gpu_id,
                )
                
            # filename = os.path.basename(video_path)
            if has_complete_samples(output_dir):
                print(f"[GPU {gpu_id}][进程{process_idx}] 输出目录已存在，跳过: {output_dir}")
                continue
            
            os.makedirs(output_dir, exist_ok=True)
            success = dataset.run_getitem(str(video_path), str(audio_path), str(lmk_path), str(output_dir))
            if success:
                print(f"[GPU {gpu_id}][进程{process_idx}] 处理完成: {video_path}")
            else:
                print(f"[GPU {gpu_id}][进程{process_idx}] 处理失败: {video_path}")
                failed += 1
                
        except Exception as e:
            print(f"[GPU {gpu_id}][进程{process_idx}] 处理 {video_path} 时出错: {e}")
            failed += 1
    if failed:
        raise RuntimeError(f"Worker {process_idx} failed to process {failed} video(s)")


def build_parser():
    parser = argparse.ArgumentParser(description="多显卡并行视频预处理脚本")
    parser.add_argument("--frame_interval", type=int, default=17)
    parser.add_argument("--num_frames", type=int, default=41)
    parser.add_argument("--landmark_model_path", type=str, required=True)
    parser.add_argument("--audio_encoder_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, default=4, help="使用的GPU数量")
    parser.add_argument("--procs_per_gpu", type=int, default=1, help="每个GPU上的进程数")
    parser.add_argument(
        "--gpu_ids",
        type=str,
        help="Comma-separated physical GPU IDs; defaults to 0..num_gpus-1",
    )
    parser.add_argument("--validate_only", action="store_true")
    return parser


def validate_args(args):
    errors = []
    for label, value in (
        ("data_dir", args.data_dir),
        ("landmark_model_path", args.landmark_model_path),
        ("audio_encoder_path", args.audio_encoder_path),
        ("vae_path", args.vae_path),
    ):
        if not Path(value).expanduser().exists():
            errors.append(f"{label} does not exist: {value}")
    if args.num_gpus < 1 or args.procs_per_gpu < 1:
        errors.append("num_gpus and procs_per_gpu must be at least 1")
    gpu_ids = (
        [int(item.strip()) for item in args.gpu_ids.split(",") if item.strip()]
        if args.gpu_ids
        else list(range(args.num_gpus))
    )
    if len(gpu_ids) != args.num_gpus:
        errors.append("gpu_ids count must match num_gpus")
    return errors, gpu_ids


def main(argv=None):
    mp.set_start_method('spawn', force=True)
    args = build_parser().parse_args(argv)
    errors, gpu_ids = validate_args(args)
    if errors:
        raise ValueError("Preflight validation failed:\n- " + "\n- ".join(errors))

    video_files = sorted(Path(args.data_dir).expanduser().resolve().rglob("*.mp4"))
    if not video_files:
        raise ValueError(f"No MP4 files found under: {args.data_dir}")
    missing_audio = [video for video in video_files if not video.with_suffix(".wav").is_file()]
    if missing_audio:
        preview = ", ".join(str(path) for path in missing_audio[:5])
        raise ValueError(f"{len(missing_audio)} video(s) have no same-stem WAV file: {preview}")
    if args.validate_only:
        print(f"Validation passed: found {len(video_files)} video(s); no model was loaded.")
        return 0
    available_gpus = torch.cuda.device_count()
    invalid_gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id < 0 or gpu_id >= available_gpus]
    if invalid_gpu_ids:
        raise ValueError(
            f"GPU IDs are unavailable: {invalid_gpu_ids}; detected {available_gpus} CUDA device(s)"
        )

    random.shuffle(video_files)
    total_procs = args.num_gpus * args.procs_per_gpu
    # 均分视频文件到各个进程
    video_chunks = [[] for _ in range(total_procs)]
    for idx, video_path in enumerate(video_files):
        video_chunks[idx % total_procs].append(video_path)
    
    print(f"使用 {args.num_gpus} 个GPU，每个GPU {args.procs_per_gpu} 个进程，总共 {total_procs} 个进程")
    for i, chunk in enumerate(video_chunks):
        print(f"进程 {i}: {len(chunk)} 个视频文件")
    
    # 创建并启动所有进程
    processes = []
    for proc_idx in range(total_procs):
        gpu_id = gpu_ids[proc_idx // args.procs_per_gpu]
        
        p = mp.Process(
            target=process_videos_on_gpu, 
            args=(gpu_id, video_chunks[proc_idx], proc_idx, args)
        )
        p.start()
        processes.append(p)
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    failed = [p.exitcode for p in processes if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"{len(failed)} preprocessing worker(s) failed: {failed}")
    
    print("所有视频处理完成！")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
