import torch
import math
import numpy as np
import torch.nn as nn
from einops import rearrange
from transformers.cache_utils import DynamicCache
from src.builder import BUILDER
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
"""
export PYTHONPATH=$(pwd):$PYTHONPATH

torchrun --master-port=29511 scripts/parallel_geneval.py     --gpus 0     --checkpoint /root/cvlab/jiyeong/reconstruction-alignment/Harmon/work_dirs/ours_hint_t_target_t/iter_3000.pth    --batch_size 4     --outdir /root/cvlab/jiyeong/reconstruction-alignment/Harmon/work_dirs/ours_hint_t_target_t/iter_3000    --config configs/models/qwen2_5_0_5b_kl16_mar_b.py     --mode geneval     --seed 42
torchrun --master-port=29512 scripts/parallel_geneval.py     --gpus 0     --checkpoint /root/cvlab/jiyeong/reconstruction-alignment/Harmon_custom/work_dirs/reca_prompt30_bottom_max50000/iter_3000.pth   --batch_size 4     --outdir /root/cvlab/jiyeong/reconstruction-alignment/Harmon_custom/work_dirs/reca_prompt30_bottom_max50000/dpg_3000    --config configs/models/qwen2_5_0_5b_kl16_mar_b.py     --mode dpgbench     --seed 42
torchrun --master-port=29510 scripts/parallel_geneval.py     --gpus 0     --checkpoint /root/cvlab/jiyeong/reconstruction-alignment/Harmon/work_dirs/ablation_prompt_percent/i2i_bottom_hint_30/iter_3000.pth    --batch_size 4     --outdir /root/cvlab/jiyeong/reconstruction-alignment/Harmon/work_dirs/ablation_prompt_percent/i2i_bottom_hint_30/iter_3000    --config configs/models/qwen2_5_0_5b_kl16_mar_b.py     --mode geneval     --seed 42
torchrun --master-port=29511 scripts/parallel_geneval.py     --gpus 0     --checkpoint /root/cvlab/jiyeong/reconstruction-alignment/Harmon/work_dirs/ablation_prompt_percent/i2i_bottom_hint_30/iter_3000.pth    --batch_size 4     --outdir /root/cvlab/jiyeong/reconstruction-alignment/Harmon/work_dirs/ablation_prompt_percent/i2i_bottom_hint_30/dpg_3000    --config configs/models/qwen2_5_0_5b_kl16_mar_b.py     --mode dpgbench     --seed 42
"""

def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),)


class Harmon(nn.Module):
    def __init__(self,
                 vae,
                 vae_scale,
                 llm,
                 mar,
                 tokenizer,
                 prompt_template):
        super().__init__()
        # VAE
        self.vae = BUILDER.build(vae)
        self.vae.requires_grad_(False)
        self.vae_scale = vae_scale

        # LLM
        self.llm = BUILDER.build(llm)
        self.tokenizer = BUILDER.build(tokenizer)
        self.prompt_template = prompt_template

        # MAR
        self.mar = BUILDER.build(mar)
        
        # Projection Layers
        self.proj_in = build_mlp(hidden_size=self.mar.encoder_embed_dim,
                                 projector_dim=self.llm.config.hidden_size,
                                 z_dim=self.llm.config.hidden_size)
        self.proj_out = build_mlp(hidden_size=self.llm.config.hidden_size,
                                  projector_dim=self.llm.config.hidden_size,
                                  z_dim=self.mar.encoder_embed_dim)

    @property
    def llm_model(self):
        return self.llm.model

    @property
    def device(self):
        return self.llm.device

    @property
    def dtype(self):
        return self.llm.dtype

    @property
    def gen_seq_len(self):
        return self.mar.seq_len

    @property
    def token_embed_dim(self):
        return self.vae.embed_dim * (self.mar.patch_size ** 2)

    @torch.no_grad()
    def encode(self, x):
        posterior = self.vae.encode(x)
        z = posterior.mode().mul_(self.vae_scale)
        z = rearrange(z, 'b c (m p) (n q) -> b m n (c p q)',
                      p=self.mar.patch_size, q=self.mar.patch_size)
        return z

    @torch.no_grad()
    def decode(self, z):
        z /= self.vae_scale
        z = rearrange(z, 'b m n (c p q) -> b c (m p) (n q)',
                      p=self.mar.patch_size, q=self.mar.patch_size)
        x = self.vae.decode(z)
        return x
    
    def prepare_forward_input(self,
                              x,
                              inputs_embeds=None,
                              input_ids=None,
                              attention_mask=None,
                              past_key_values=None,
                              num_mp_tokens=0,
                              mp_start_idx=0,
                              **kwargs): # hint_original_indices 등은 이제 안 쓰므로 kwargs로 처리
        
        # x: Target Image Embeddings (Mi) -> [Batch, L_img, D]
        b, l_img, _ = x.shape
        
        # 1. inputs_embeds 준비 및 길이 계산
        if inputs_embeds is None:
            if input_ids is None:
                 raise ValueError("Both inputs_embeds and input_ids are None")
            input_ids = input_ids.to(self.device)
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            
        l_cond = inputs_embeds.shape[1] 
        total_len = l_cond + l_img

        # 2. 전체 입력 결합
        final_inputs_embeds = torch.cat([inputs_embeds, x], dim=1)

        # 3. 2D Padding Mask
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=self.device, dtype=torch.bool)
            combined_mask_2d = torch.cat([
                attention_mask, 
                torch.ones((b, l_img), device=self.device, dtype=torch.bool)
            ], dim=1)
        else:
            combined_mask_2d = torch.ones((b, total_len), device=self.device, dtype=torch.bool)

        # Position IDs
        position_ids = torch.cumsum(combined_mask_2d, dim=1) - 1
        position_ids[position_ids < 0] = 0

        # [Inference]
        if past_key_values is not None:
            final_inputs_embeds = x
            position_ids = position_ids[:, -l_img:]
            return dict(inputs_embeds=final_inputs_embeds,
                        attention_mask=combined_mask_2d,
                        position_ids=position_ids,
                        past_key_values=past_key_values)

        # ============================================================
        # ★ [Masking] Hint(Full) + Generation(Causal)
        # ============================================================
        min_dtype = torch.finfo(self.dtype).min
        mask_4d = torch.zeros((b, 1, total_len, total_len), device=self.device, dtype=self.dtype)

        # (1) Base Causal Masking (삼각형)
        #     이것만 있어도 Mp가 Mi보다 앞에 있기 때문에, Mi는 Mp를 "전부" 볼 수 있습니다.
        causal_mask = torch.triu(torch.ones((total_len, total_len), device=self.device), diagonal=1).bool()
        mask_4d.masked_fill_(causal_mask, min_dtype)

        # (2) Padding Masking
        expanded_pad_mask = combined_mask_2d[:, None, None, :].to(dtype=torch.bool)
        mask_4d.masked_fill_(~expanded_pad_mask, min_dtype)

        # --------------------------------------------------------
        # ★ [Hint Region] Full Attention (Hint끼리 서로 다 보게 함)
        # --------------------------------------------------------
        if num_mp_tokens > 0:
            hint_start = mp_start_idx
            hint_end = mp_start_idx + num_mp_tokens
            hint_end = min(hint_end, l_cond)
            
            if hint_end > hint_start:
                # 힌트 내부(Mp -> Mp)를 0.0으로 뚫어서 Full Attention으로 만듦
                # (이 코드가 없으면 힌트끼리도 Causal(삼각형)이 됨)
                mask_4d[:, :, hint_start:hint_end, hint_start:hint_end].fill_(0.0)

        # ★★★ [삭제됨] ★★★
        # 기존에 있었던 "Red Box (Identity 차단)" 로직을 완전히 제거했습니다.
        # 이제 Mi는 앞에 있는 Mp를 가리는 것 없이 100% 다 볼 수 있습니다.

        # (Safety) Padding 다시 닫기
        mask_4d.masked_fill_(~expanded_pad_mask, min_dtype)

        return dict(inputs_embeds=final_inputs_embeds,
                    attention_mask=mask_4d,
                    position_ids=position_ids,
                    past_key_values=past_key_values)
    # def prepare_forward_input(self,
    #                           x,
    #                           inputs_embeds=None,
    #                           input_ids=None,
    #                           attention_mask=None,
    #                           past_key_values=None,
    #                           num_mp_tokens=0,
    #                           mp_start_idx=0,
    #                           use_sparse_bridge=False,
    #                           hint_original_indices=None):
        
    #     # x: Target Image Embeddings (Mi) -> [Batch, L_img, D]
    #     b, l_img, _ = x.shape
        
    #     # 1. inputs_embeds 준비 및 길이 계산
    #     if inputs_embeds is None:
    #         if input_ids is None:
    #              raise ValueError("Both inputs_embeds and input_ids are None")
    #         input_ids = input_ids.to(self.device)
    #         inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            
    #     l_cond = inputs_embeds.shape[1] 
    #     total_len = l_cond + l_img

    #     # 2. 전체 입력 결합
    #     final_inputs_embeds = torch.cat([inputs_embeds, x], dim=1)

    #     # 3. 2D Padding Mask (패딩 처리용)
    #     if attention_mask is not None:
    #         attention_mask = attention_mask.to(device=self.device, dtype=torch.bool)
    #         combined_mask_2d = torch.cat([
    #             attention_mask, 
    #             torch.ones((b, l_img), device=self.device, dtype=torch.bool)
    #         ], dim=1)
    #     else:
    #         combined_mask_2d = torch.ones((b, total_len), device=self.device, dtype=torch.bool)

    #     # Position IDs
    #     position_ids = torch.cumsum(combined_mask_2d, dim=1) - 1
    #     position_ids[position_ids < 0] = 0

    #     # [Inference 시]
    #     if past_key_values is not None:
    #         final_inputs_embeds = x
    #         position_ids = position_ids[:, -l_img:]
    #         return dict(inputs_embeds=final_inputs_embeds,
    #                     attention_mask=combined_mask_2d,
    #                     position_ids=position_ids,
    #                     past_key_values=past_key_values)

    #     # ============================================================
    #     # ★ [Masking] Hint(Full) + Generation(Causal) + Sparse Bridge
    #     # ============================================================
    #     min_dtype = torch.finfo(self.dtype).min
    #     mask_4d = torch.zeros((b, 1, total_len, total_len), device=self.device, dtype=self.dtype)

    #     # (1) Causal Masking (기본 바탕: 전체를 일단 Autoregressive로 설정)
    #     #     Triangular Mask: 미래 토큰 차단
    #     causal_mask = torch.triu(torch.ones((total_len, total_len), device=self.device), diagonal=1).bool()
    #     mask_4d.masked_fill_(causal_mask, min_dtype)

    #     # (2) Padding Masking
    #     expanded_pad_mask = combined_mask_2d[:, None, None, :].to(dtype=torch.bool)
    #     mask_4d.masked_fill_(~expanded_pad_mask, min_dtype)

    #     # --------------------------------------------------------
    #     # ★ [NEW] Hint Region: Full Attention (Bidirectional)
    #     # --------------------------------------------------------
    #     if num_mp_tokens > 0:
    #         hint_start = mp_start_idx
    #         hint_end = mp_start_idx + num_mp_tokens
    #         hint_end = min(hint_end, l_cond)
    #         curr_hint_len = hint_end - hint_start
            
    #         if curr_hint_len > 0:
    #             # [이미지 좌상단 주황색 영역]
    #             # 힌트끼리는 Causal Mask를 지워서 서로 다 보게 만듦 (Full Attention)
    #             # mask_4d[:, :, hint_start:hint_end, hint_start:hint_end] 영역을 0.0으로 덮어씀
                
    #             # 주의: 단순히 fill(0.0)을 하면 Padding까지 살아날 수 있으므로, 
    #             # Causal로 인해 가려진 부분(Triu)만 열어줘야 함.
    #             # 하지만 간단하게 해당 영역 전체를 0으로 밀고 Pad Mask를 다시 씌우는 게 안전함.
                
    #             # 1. 힌트 내부 Full Open
    #             hint_area = mask_4d[:, :, hint_start:hint_end, hint_start:hint_end]
    #             hint_area.fill_(0.0) 

    #             # --------------------------------------------------------
    #             # [B] Red Box: Target -> Hint Sparse Blocking (Identity 차단)
    #             # --------------------------------------------------------
    #             # [이미지 좌하단 녹색 영역]
    #             if hint_original_indices is not None:
    #                 # 1. Target 상에서의 진짜 위치
    #                 real_target_indices = hint_original_indices + l_cond
                    
    #                 # 2. Hint 상에서의 현재 순서
    #                 current_hint_indices = torch.arange(curr_hint_len, device=self.device) + hint_start
                    
    #                 # 3. 배치별 차단
    #                 for i in range(b):
    #                     t_idx = real_target_indices[i]  # [Num_Hint]
    #                     h_idx = current_hint_indices    # [Num_Hint]
                        
    #                     valid = (t_idx < total_len)
                        
    #                     # Target(t_idx)가 Hint(h_idx)를 볼 때 Identity 차단
    #                     # [이미지의 흰색 구멍 부분]
    #                     mask_4d[i, :, t_idx[valid], h_idx[valid]] = min_dtype

    #             # (Safety) Padding이 지워졌을 수 있으니 다시 한 번 확실하게 닫음
    #             mask_4d.masked_fill_(~expanded_pad_mask, min_dtype)

    #     return dict(inputs_embeds=final_inputs_embeds,
    #                 attention_mask=mask_4d,
    #                 position_ids=position_ids,
    #                 past_key_values=past_key_values)
        
        
    # def prepare_forward_input(self,
    #                           x,
    #                           inputs_embeds=None,
    #                           input_ids=None,
    #                           attention_mask=None,
    #                           past_key_values=None,
    #                           num_mp_tokens=0,
    #                           mp_start_idx=0,
    #                           **kwargs): # hint_original_indices 등 안 쓰는 인자 에러 방지용
        
    #     # x: Target Image Embeddings (Mi) -> [Batch, L_img, D]
    #     b, l_img, _ = x.shape
        
    #     # --------------------------------------------------------
    #     # 1. inputs_embeds 준비 및 길이(total_len) 계산
    #     # --------------------------------------------------------
    #     if inputs_embeds is None:
    #         if input_ids is None:
    #              raise ValueError("Both inputs_embeds and input_ids are None")
    #         input_ids = input_ids.to(self.device)
    #         inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            
    #     l_cond = inputs_embeds.shape[1] 
    #     total_len = l_cond + l_img

    #     # 2. 전체 입력 결합 (Condition + Image)
    #     final_inputs_embeds = torch.cat([inputs_embeds, x], dim=1)

    #     # 3. 2D Padding Mask (패딩 처리용)
    #     if attention_mask is not None:
    #         attention_mask = attention_mask.to(device=self.device, dtype=torch.bool)
    #         combined_mask_2d = torch.cat([
    #             attention_mask, 
    #             torch.ones((b, l_img), device=self.device, dtype=torch.bool)
    #         ], dim=1)
    #     else:
    #         combined_mask_2d = torch.ones((b, total_len), device=self.device, dtype=torch.bool)

    #     # Position IDs
    #     position_ids = torch.cumsum(combined_mask_2d, dim=1) - 1
    #     position_ids[position_ids < 0] = 0

    #     # [Inference Optimization]
    #     if past_key_values is not None:
    #         final_inputs_embeds = x
    #         position_ids = position_ids[:, -l_img:]
    #         return dict(inputs_embeds=final_inputs_embeds,
    #                     attention_mask=combined_mask_2d,
    #                     position_ids=position_ids,
    #                     past_key_values=past_key_values)

    #     # ============================================================
    #     # ★ [Masking] Robust Checkerboard Attention
    #     # ============================================================
    #     min_dtype = torch.finfo(self.dtype).min
    #     mask_4d = torch.zeros((b, 1, total_len, total_len), device=self.device, dtype=self.dtype)

    #     # (1) Base Causal Masking (기본 바탕: 삼각형)
    #     #     - 미래 토큰 참조 방지 (Autoregressive 기본 원칙)
    #     causal_mask = torch.triu(torch.ones((total_len, total_len), device=self.device), diagonal=1).bool()
    #     mask_4d.masked_fill_(causal_mask, min_dtype)

    #     # (2) Padding Masking (패딩 토큰 참조 방지)
    #     expanded_pad_mask = combined_mask_2d[:, None, None, :].to(dtype=torch.bool)
    #     mask_4d.masked_fill_(~expanded_pad_mask, min_dtype)

    #     if num_mp_tokens > 0:
    #         hint_start = mp_start_idx
    #         hint_end = mp_start_idx + num_mp_tokens
    #         hint_end = min(hint_end, l_cond)
            
    #         # (3) Hint Region: Full Attention
    #         #     - 힌트끼리는 서로 다 볼 수 있게 Causal Mask 제거
    #         if hint_end > hint_start:
    #             mask_4d[:, :, hint_start:hint_end, hint_start:hint_end].fill_(0.0)
            
    #         # --------------------------------------------------------
    #         # ★ [Target -> Hint] Relative Checkerboard Masking
    #         # --------------------------------------------------------
    #         target_start = l_cond
    #         target_end = total_len
            
    #         if hint_end > hint_start:
    #             # A. 절대 좌표 (실제 마스크에 적용할 인덱스)
    #             abs_target_rows = torch.arange(target_start, target_end, device=self.device)
    #             abs_hint_cols = torch.arange(hint_start, hint_end, device=self.device)
                
    #             # B. 상대 좌표 (패턴 계산용 인덱스: 0, 1, 2...)
    #             #    - 텍스트 길이에 상관없이 항상 0부터 시작하도록 보정
    #             rel_target_rows = abs_target_rows - target_start
    #             rel_hint_cols = abs_hint_cols - hint_start
                
    #             # C. Meshgrid 생성 (Indexing='ij')
    #             #    - grid_abs: 마스크 위치 지정용
    #             #    - grid_rel: 홀/짝 계산용
    #             grid_rows_abs, grid_cols_abs = torch.meshgrid(abs_target_rows, abs_hint_cols, indexing='ij')
    #             grid_rows_rel, grid_cols_rel = torch.meshgrid(rel_target_rows, rel_hint_cols, indexing='ij')
                
    #             # D. 격자무늬 조건: (상대 행 + 상대 열)이 홀수면 차단
    #             #    - (0,0)은 항상 짝수 -> 보임(Allow)
    #             #    - (0,1)은 홀수 -> 가림(Block)
    #             checkerboard_mask = (grid_rows_rel + grid_cols_rel) % 2 != 0
                
    #             # E. 마스크 적용
    #             #    - 조건이 True인 곳을 min_dtype으로 덮어씀
    #             mask_4d[:, :, grid_rows_abs, grid_cols_abs] = mask_4d[:, :, grid_rows_abs, grid_cols_abs].masked_fill(
    #                 checkerboard_mask.unsqueeze(0).unsqueeze(0), min_dtype
    #             )

    #     # (Safety) Unmasking 과정에서 패딩이 뚫렸을 수 있으므로 다시 닫음
    #     mask_4d.masked_fill_(~expanded_pad_mask, min_dtype)

    #     return dict(inputs_embeds=final_inputs_embeds,
    #                 attention_mask=mask_4d,
    #                 position_ids=position_ids,
    #                 past_key_values=past_key_values)
        
    def extract_visual_feature(self, x, mask=None, detach=False):
        b, m, n, _ = x.shape
        x = x.view(b, m*n, -1)
        if mask is None:
            mask = torch.zeros_like(x[..., 0])
        null_embeds = self.mar.fake_latent.expand(x.shape[0], -1)
        
        # MAR Encoder 호출
        x_enc = self.mar.forward_mae_encoder(x, mask, null_embeds, image_shape=(m, n))

        z_enc = self.proj_in(x_enc)
        z_enc = torch.cat([
            z_enc[:, self.mar.buffer_size:],
            z_enc[:, :self.mar.buffer_size]], dim=1)

        if detach:
            x_enc = x_enc.detach()
            z_enc = z_enc.detach()

        return x_enc, z_enc

    def forward_mae_encoder(self, 
                            x, 
                            mask, 
                            detach=False, 
                            inputs_embeds=None, 
                            input_ids=None, 
                            attention_mask=None,
                            num_mp_tokens=0,    # [New]
                            mp_start_idx=0,     # [New]
                            **context):
        
        b, m, n, _ = x.shape
        x_enc, z_enc = self.extract_visual_feature(x, mask=mask, detach=detach)

        inputs = self.prepare_forward_input(
            x=z_enc, 
            inputs_embeds=inputs_embeds, 
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_mp_tokens=num_mp_tokens, 
            mp_start_idx=mp_start_idx, 
            use_sparse_bridge=False, # [Option] Random Sparsity를 쓰고 싶으면 True, 아니면 False
            **context
        )
        
        output = self.llm_model(**inputs, return_dict=True)

        z_llm = output.last_hidden_state[:, -z_enc.shape[1]:]
        z_llm = torch.cat([
            z_llm[:, -self.mar.buffer_size:],
            z_llm[:, :-self.mar.buffer_size]], dim=1)

        x_enc = x_enc + self.proj_out(z_llm)

        return x_enc

    @staticmethod
    def curtail_cache(past_key_values, cur_len):
        for past_key_values_ in past_key_values:
            keys, values = past_key_values_
            keys.data = keys.data[:, :, :cur_len]
            values.data = values.data[:, :, :cur_len]

    @torch.no_grad()
    def prepare_text_conditions(self, prompt, cfg_prompt='Generate an image.'):
        all_prompts = [self.prompt_template['INSTRUCTION'].format(input=prompt),
                       self.prompt_template['INSTRUCTION'].format(input=cfg_prompt)]

        input_ids = [self.tokenizer.encode(p, add_special_tokens=True, return_tensors='pt')[0]
                     for p in all_prompts]
        valid_lens = [len(input_ids_) for input_ids_ in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True,
                                 padding_value=self.tokenizer.eos_token_id)
        attention_mask = torch.zeros_like(input_ids).bool()
        for i in range(len(input_ids)):
            attention_mask[i, :valid_lens[i]] = True

        return dict(input_ids=input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device))
# import torch
# import math
# import numpy as np
# import torch.nn as nn
# from einops import rearrange
# from transformers.cache_utils import DynamicCache
# from src.builder import BUILDER
# from tqdm import tqdm
# from torch.nn.utils.rnn import pad_sequence


# def build_mlp(hidden_size, projector_dim, z_dim):
#     return nn.Sequential(
#         nn.Linear(hidden_size, projector_dim),
#         nn.SiLU(),
#         nn.Linear(projector_dim, z_dim),)


# def mask_by_order(mask_len, order, bsz, seq_len):
#     masking = torch.zeros(bsz, seq_len, device=order.device)
#     masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()],
#                             src=torch.ones(bsz, seq_len, device=order.device)).bool()
#     return masking


# class Harmon(nn.Module):
#     def __init__(self,
#                  vae,
#                  vae_scale,
#                  llm,
#                  mar,
#                  tokenizer,
#                  prompt_template):
#         super().__init__()
#         # VAE
#         self.vae = BUILDER.build(vae)
#         self.vae.requires_grad_(False)
#         self.vae_scale = vae_scale

#         # LLM
#         self.llm = BUILDER.build(llm)
#         self.tokenizer = BUILDER.build(tokenizer)
#         self.prompt_template = prompt_template

#         # MAR
#         self.mar = BUILDER.build(mar)
#         # projection layers
#         self.proj_in = build_mlp(hidden_size=self.mar.encoder_embed_dim,
#                                  projector_dim=self.llm.config.hidden_size,
#                                  z_dim=self.llm.config.hidden_size)
#         self.proj_out = build_mlp(hidden_size=self.llm.config.hidden_size,
#                                   projector_dim=self.llm.config.hidden_size,
#                                   z_dim=self.mar.encoder_embed_dim)

#     @property
#     def llm_model(self):
#         return self.llm.model

#     @property
#     def device(self):
#         return self.llm.device

#     @property
#     def dtype(self):
#         return self.llm.dtype

#     @property
#     def gen_seq_len(self):
#         return self.mar.seq_len

#     @property
#     def token_embed_dim(self):
#         return self.vae.embed_dim * (self.mar.patch_size ** 2)

#     @torch.no_grad()
#     def encode(self, x):
#         posterior = self.vae.encode(x)
#         z = posterior.mode().mul_(self.vae_scale)
#         z = rearrange(z, 'b c (m p) (n q) -> b m n (c p q)',
#                       p=self.mar.patch_size, q=self.mar.patch_size)

#         return z

#     @torch.no_grad()
#     def decode(self, z):
#         z /= self.vae_scale
#         z = rearrange(z, 'b m n (c p q) -> b c (m p) (n q)',
#                       p=self.mar.patch_size, q=self.mar.patch_size)

#         x = self.vae.decode(z)
#         return x

#     def prepare_forward_input(self,
#                               x,
#                               inputs_embeds=None,
#                               input_ids=None,
#                               attention_mask=None,
#                               past_key_values=None):
#         b, l, _ = x.shape
#         attention_mask = attention_mask.to(device=self.device, dtype=torch.bool)
#         attention_mask = torch.cat([
#             attention_mask, attention_mask.new_ones(b, l)
#         ], dim=1)
#         position_ids = torch.cumsum(attention_mask, dim=1) - 1
#         position_ids[position_ids < 0] = 0

#         # import pdb; pdb.set_trace()

#         # prepare context
#         if past_key_values is not None:
#             inputs_embeds = x
#             position_ids = position_ids[:, -l:]
#         else:
#             if inputs_embeds is None:
#                 input_ids = input_ids.to(self.device)
#                 inputs_embeds = self.llm.get_input_embeddings()(input_ids)
#             inputs_embeds = torch.cat([inputs_embeds, x], dim=1)

#         return dict(inputs_embeds=inputs_embeds,
#                     attention_mask=attention_mask,
#                     position_ids=position_ids,
#                     past_key_values=past_key_values)

    # def extract_visual_feature(self, x, mask=None, detach=False):
    #     b, m, n, _ = x.shape
    #     x = x.view(b, m*n, -1)
    #     # x: b mn c
    #     if mask is None:
    #         mask = torch.zeros_like(x[..., 0])
    #     null_embeds = self.mar.fake_latent.expand(x.shape[0], -1)
    #     x_enc = self.mar.forward_mae_encoder(x, mask, null_embeds, image_shape=(m, n))

    #     z_enc = self.proj_in(x_enc)
    #     # Move buffers to the end of the image sequence
    #     z_enc = torch.cat([
    #         z_enc[:, self.mar.buffer_size:],
    #         z_enc[:, :self.mar.buffer_size]], dim=1)

    #     if detach:
    #         x_enc = x_enc.detach()
    #         z_enc = z_enc.detach()

    #     return x_enc, z_enc

    # def forward_mae_encoder(self, x, mask, detach=False, inputs_embeds=None, **context):

    #     b, m, n, _ = x.shape
    #     x_enc, z_enc = self.extract_visual_feature(x, mask=mask, detach=detach)
    #     inputs = self.prepare_forward_input(x=z_enc, inputs_embeds=inputs_embeds, **context)
    #     output = self.llm_model(**inputs, return_dict=True)

    #     z_llm = output.last_hidden_state[:, -z_enc.shape[1]:]

    #     # move buffers back to the start of the image sequence
    #     z_llm = torch.cat([
    #         z_llm[:, -self.mar.buffer_size:],
    #         z_llm[:, :-self.mar.buffer_size]], dim=1)

    #     # residual learning
    #     x_enc = x_enc + self.proj_out(z_llm)

    #     return x_enc

    # @staticmethod
    # def curtail_cache(past_key_values, cur_len):
    #     for past_key_values_ in past_key_values:
    #         keys, values = past_key_values_
    #         keys.data = keys.data[:, :, :cur_len]
    #         values.data = values.data[:, :, :cur_len]

    # @torch.no_grad()
    # def prepare_text_conditions(self, prompt, cfg_prompt='Generate an image.'):
    #     all_prompts = [self.prompt_template['INSTRUCTION'].format(input=prompt),
    #                    self.prompt_template['INSTRUCTION'].format(input=cfg_prompt)]

    #     input_ids = [self.tokenizer.encode(p, add_special_tokens=True, return_tensors='pt')[0]
    #                  for p in all_prompts]
    #     valid_lens = [len(input_ids_) for input_ids_ in input_ids]
    #     input_ids = pad_sequence(input_ids, batch_first=True,
    #                              padding_value=self.tokenizer.eos_token_id)
    #     attention_mask = torch.zeros_like(input_ids).bool()
    #     for i in range(len(input_ids)):
    #         attention_mask[i, :valid_lens[i]] = True

    #     return dict(input_ids=input_ids.to(self.device),
    #                 attention_mask=attention_mask.to(self.device))

    @torch.no_grad()
    def sample(self,
               input_ids=None, inputs_embeds=None,
               attention_mask=None, num_iter=64, cfg=1.0, cfg_schedule="constant", temperature=1.0,
               progress=False, mask=None, past_key_values=None, image_shape=None, x_con=None, **kwargs):
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        bsz = attention_mask.shape[0]
        if cfg != 1.0:
            assert bsz % 2 == 0

        if image_shape is None:
            m = n = int(self.gen_seq_len ** 0.5)
        else:
            m, n = image_shape

        if mask is None:
            mask = torch.ones(bsz, m*n, device=self.device, dtype=self.dtype)
        else:
            mask = mask.view(bsz, m*n)
        tokens = torch.zeros(bsz, m*n, self.token_embed_dim,
                             device=self.device, dtype=self.dtype)
        orders = self.mar.sample_orders(bsz, seq_len=m*n)
        if cfg != 1.0:
            orders[bsz//2:] = orders[:bsz//2]

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)

        # past key values can be prepared outside (usually in multi-turn editing)
        if past_key_values is None:
            output = self.llm_model(inputs_embeds=inputs_embeds,
                                    attention_mask=None,
                                    position_ids=None,
                                    past_key_values=DynamicCache.from_legacy_cache(),
                                    return_dict=True,
                                    use_cache=True)
            past_key_values = output.past_key_values

        # generate latents
        for step in indices:
            cur_tokens = tokens.clone()
            x_enc = self.forward_mae_encoder(tokens.view(bsz, m, n, -1),
                                             mask.to(self.dtype),
                                             past_key_values=past_key_values,
                                             # inputs_embeds=inputs_embeds,
                                             attention_mask=attention_mask)
            # import pdb; pdb.set_trace()
            self.curtail_cache(past_key_values, inputs_embeds.shape[1])
            # import pdb; pdb.set_trace()

            z = self.mar.forward_mae_decoder(x_enc, mask.to(self.dtype), image_shape=(m, n), x_con=x_con)

            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(m*n * mask_ratio)]).to(self.device)

            # masks out at least one for the next iteration
            mask_len = torch.maximum(torch.Tensor([1]).to(self.device),
                                     torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))

            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, bsz, m*n).to(self.device)
            if cfg != 1.0:
                mask_next[bsz//2:] = mask_next[:bsz//2]
            if step >= num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            mask = mask_next
            # if not cfg == 1.0:
            #     mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            # sample token latents for this step
            z = z[mask_to_pred.nonzero(as_tuple=True)]
            # cfg schedule follow Muse
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (m*n - mask_len[0]) / (m*n)
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError
            sampled_token_latent = self.mar.diffloss.sample(z, temperature, cfg_iter).to(self.dtype)
            # if not cfg == 1.0:
            #     sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
            #     mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            if cfg != 1.0:
                cur_tokens[bsz//2:] = cur_tokens[:bsz//2]
            tokens = cur_tokens.clone()

        pred = self.decode(tokens.view(bsz, m, n, -1))

        if cfg != 1.0:
            pred = pred[:bsz//2]
        return pred

    @torch.no_grad()
    def sample_recon(self, 
                    image, 
                    prompt="Describe this image in details.", 
                    temperature=1.0, 
                    num_iter=32,
                    cfg=3.0,
                    progress=True):
        
        x = self.encode(image)
        b, m, n, _ = x.shape
        
        _, z_enc = self.extract_visual_feature(x, detach=True)
        
        text_prompt = self.prompt_template['INSTRUCTION'].format(input=prompt)
        input_ids = self.tokenizer.encode(text_prompt, add_special_tokens=True, return_tensors='pt')[0].to(self.device)
 
                
        input_embeds = self.llm.get_input_embeddings()(input_ids.unsqueeze(0))
        input_embeds = input_embeds.expand(b, -1, -1)
        
        input_embeds = torch.cat([
            input_embeds[:,:3,:],
            z_enc.reshape(b, -1, input_embeds.shape[-1]),
            input_embeds[:,3:,:],
        ], dim=1)
                        
        text_prompt_uncond = self.prompt_template['INSTRUCTION'].format(input="Generate an image.")
        input_ids_uncond = self.tokenizer.encode(text_prompt_uncond, add_special_tokens=True, return_tensors='pt')[0].to(self.device)
        # padding uncond to the same length
        padded_length = input_embeds.shape[1] - input_ids_uncond.shape[0]
        input_ids_uncond = torch.cat([input_ids_uncond, 
                                      torch.full((input_embeds.shape[1] - input_ids_uncond.shape[0],), 
                                                 self.tokenizer.pad_token_id, device=self.device)], dim=0)
        input_embeds_uncond = self.llm.get_input_embeddings()(input_ids_uncond.unsqueeze(0))
        input_embeds_uncond = input_embeds_uncond.expand(b, -1, -1)
        input_embeds = torch.cat([input_embeds, 
                                  input_embeds_uncond], dim=0)
        # padding attention mask
        
        attention_mask = torch.ones((b * 2, input_embeds.shape[1]), 
                                    dtype=torch.bool, 
                                    device=self.device)
        attention_mask[b:, -padded_length:] = False
                

        recon_image = self.sample(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            num_iter=num_iter,
            temperature=temperature,
            progress=progress,
            cfg=cfg,
            image_shape=(m, n)
        )
        
        return recon_image


        