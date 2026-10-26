import torch
import torch.nn.functional as F
from torch.nn.modules.module import T
from mmengine.model import BaseModel
from torch.autograd.function import Function
from mmengine.logging import print_log
from xtuner.model.utils import guess_load_checkpoint
from xtuner.utils import IMAGE_TOKEN_INDEX
from .harmon import Harmon
import math
import torch.nn as nn
import string
import nltk

class _ScaleGradient(Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None
    
def build_project_mlp(hidden_size, projector_dim, z_dim):
    mlp = nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.RMSNorm(projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
        # nn.LayerNorm(projector_dim),
    )

    # 2. 할당된 mlp에 대해 초기화를 수행합니다.
    for m in mlp.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    return mlp


# =========================================================
# [설정] 학습 시 불용어(Stopwords) 제거를 위한 전역 설정
# =========================================================
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))

# 모델 특화 불용어 추가
custom_stops = {"some", "image", "caption", "es", "ed", "describe", "details"}
STOPWORDS.update(custom_stops)

def is_exclude_token(token):
    # 토크나이저 특수문자 제거 (Qwen/RoBERTa 등)
    t_clean = token.replace("Ġ", "").replace("▁", "").strip().lower()
    
    if not t_clean: return True
    if all(char in string.punctuation for char in t_clean): return True
    if t_clean in STOPWORDS: return True
    return False

class HarmonDev(Harmon, BaseModel):
    def __init__(self,
                 grad_scale=0.1,
                 loss_weights={'image2text': 1.0, 'text2image': 1.0, 'recon': 1.0},
                 pretrained_pth=None,
                 freeze_llm=False,
                 gradient_checkpointing=True,
                 visual_hint_ratio=0.3,
                 visual_hint_selection='highest',
                 target_mask_selection='highest',
                 text_token_filtering=True,
                 text_keep_ratio=0.4,
                 text_min_tokens=3,
                 noise_scale=1.0,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        if not 0 < visual_hint_ratio <= 1:
            raise ValueError('visual_hint_ratio must be in (0, 1]')
        for strategy in (visual_hint_selection, target_mask_selection):
            if strategy not in ('highest', 'lowest', 'random'):
                raise ValueError(f'Unknown affinity selection strategy: {strategy}')
        if not 0 < text_keep_ratio <= 1:
            raise ValueError('text_keep_ratio must be in (0, 1]')
        if isinstance(text_min_tokens, bool) or not isinstance(text_min_tokens, int) or text_min_tokens < 1:
            raise ValueError('text_min_tokens must be a positive integer')
        if not 0 < noise_scale < float('inf'):
            raise ValueError('noise_scale must be positive and finite; noise is always enabled')
        self.text_keep_ratio = text_keep_ratio
        self.text_min_tokens = text_min_tokens
        self.noise_scale = noise_scale
        self.visual_hint_ratio = visual_hint_ratio
        self.visual_hint_selection = visual_hint_selection
        self.target_mask_selection = target_mask_selection
        self.text_token_filtering = text_token_filtering
        self.grad_scale = grad_scale
        self.loss_weights = loss_weights
        self.debuged = False
        self.proj_align = build_project_mlp(hidden_size=self.mar.encoder_embed_dim, 
                                            projector_dim= self.mar.encoder_embed_dim, 
                                            z_dim= self.mar.encoder_embed_dim)

        ##################################################################################### 불용어 때문에 추가
        # [Speed Optimization] Pre-compute stopword mask for GPU lookup
        print_log("[Init] Pre-computing stopword mask for fast training...")
        vocab_size = len(self.tokenizer)
        self.register_buffer("content_token_mask", torch.ones(vocab_size, dtype=torch.bool))
        
        for token_id in range(vocab_size):
            token_str = self.tokenizer.convert_ids_to_tokens(token_id)
            if token_str and is_exclude_token(token_str):
                self.content_token_mask[token_id] = False
        print_log("[Init] Stopword mask ready.")
        #####################################################################################

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            info = self.load_state_dict(pretrained_state_dict, strict=False)
            print_log(f'Load pretrained weight from {pretrained_pth}')

        if freeze_llm:
            self.llm.requires_grad_(False)

        # gradient checkpointing
        if gradient_checkpointing:
            self.gradient_checkpointing_enable()
        else:
            self.gradient_checkpointing_disable()

    def select_visual_hints(self, scores):
        """Select hints by affinity, then restore spatial token order."""
        count = max(1, int(scores.shape[-1] * self.visual_hint_ratio))
        ranked_scores = torch.rand_like(scores) if self.visual_hint_selection == 'random' else scores
        indices = torch.topk(
            ranked_scores, k=count, dim=-1,
            largest=self.visual_hint_selection != 'lowest').indices
        return indices.sort(dim=-1).values

    def make_target_mask(self, scores, return_count=False):
        """Mask high-affinity targets by default; retain the existing mask budget."""
        ranked_scores = torch.rand_like(scores) if self.target_mask_selection == 'random' else scores
        return self.mar.masking_from_affinity(
            ranked_scores, largest=self.target_mask_selection != 'lowest',
            rate=return_count)

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        self.mar.gradient_checkpointing_disable()

    def gradient_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()
        self.mar.gradient_checkpointing_enable()

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        state_dict = {k: v for k, v in state_dict.items()
                      if 'vae.' not in k}

        return state_dict

    def train(self: T, mode: bool = True) -> T:
        super().train(mode=mode)
        self.vae.train(mode=False)
        return self

    def text2image_loss(self, data_dict):
        # --------------------
        # 1) 이미지 인코딩
        # --------------------
        x = data_dict['pixel_values'].to(dtype=self.dtype, device=self.device)
        x = self.encode(x)     # [B, m, n, C]
        b, m, n, _ = x.shape
        gt_latents = x.clone().detach().view(b, m*n, -1)

        # 텍스트 (원래 캡션)
        input_ids = data_dict['input_ids'].to(self.device)          # [B, T]
        attention_mask = data_dict['attention_mask'].to(self.device)

        # --------------------
        # 2) visual feature 추출 (한 번만)
        # --------------------
        # z_enc: [B, N_img, D], 앞 m*n 이 진짜 이미지 토큰, 뒤는 buffer
        _, z_enc = self.extract_visual_feature(x)                   

        # --------------------
        # 3) 텍스트–이미지 affinity 계산 (t2i=False: 이미 z_enc 넘김)
        # --------------------
        affinity_score, wo_noise_affinity_score = self.affinity_map_t_i(
            img_emb=z_enc,                    # [B, N_img, D]
            input_ids=input_ids,
            attention_mask=attention_mask,
            t2i=False,
        )
        affinity_score = affinity_score[:, :m*n]   # buffer 토큰 제외, [B, m*n]
        # wo_noise_affinity_score = wo_noise_affinity_score[:, :m*n]   # buffer 토큰 제외, [B, m*n]

        # 필요하면 grad scale
        if self.grad_scale is not None:
            z_enc = _ScaleGradient.apply(z_enc, self.grad_scale)

        # --------------------
        # 4) 이미지 토큰 subset 뽑기 (image prompt로 쓸 부분)
        # --------------------
        z_img = z_enc[:, :m*n, :]                      # [B, m*n, D]
        num_img_tokens = z_img.shape[1]

        keep_idx = self.select_visual_hints(affinity_score)
        num_keep = keep_idx.shape[-1]

        # 배치별 인덱스로 gather
        z_enc_sub = torch.gather(
            z_img,  # [B, m*n, D]
            dim=1,
            index=keep_idx.unsqueeze(-1).expand(-1, -1, z_img.shape[-1])
        )                                             # [B, num_keep, D]

        # -----------------------------------------------------------
        # 5) input_ids / attention_mask 재구성해서
        #    이미지 토큰을 텍스트 사이에 끼워 넣기
        # --------------------
        # 전제: input_ids[:, 3] 가 원래 IMAGE_TOKEN_INDEX 하나 있는 자리.
        for i in range(input_ids.shape[0]):
            assert input_ids[i, 3] == IMAGE_TOKEN_INDEX

        # IMAGE_TOKEN_INDEX 하나를 num_keep 개로 확장
        input_ids = torch.cat(
            (
                input_ids[:, :3],
                IMAGE_TOKEN_INDEX * torch.ones(
                    b, num_keep, dtype=torch.long, device=input_ids.device
                ),
                input_ids[:, 4:],
            ),
            dim=1
        )

        attention_mask = torch.cat(
            (
                attention_mask[:, :3],
                torch.ones(
                    b, num_keep,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                ),
                attention_mask[:, 4:],
            ),
            dim=1
        )
        # --------------------
        # 6) inputs_embeds 만들기
        # --------------------
        inputs_embeds = z_enc_sub.new_zeros(
            *input_ids.shape, self.llm.config.hidden_size
        )  # [B, T_new, H]

        # 이미지 들어갈 자리
        img_mask = (input_ids == IMAGE_TOKEN_INDEX)    # [B, T_new]

        # flatten 해서 채워 넣기
        inputs_embeds[img_mask] = z_enc_sub.flatten(0, 1)

        # 나머지 텍스트 토큰은 LLM embedding 으로 채움
        inputs_embeds[~img_mask] = self.llm.get_input_embeddings()(
            input_ids[~img_mask]
        )
        
        # --------------------
        # 7) MAE 마스크 (affinity 기반)
        # --------------------
        mask, num_masked_tokens = self.make_target_mask(affinity_score, return_count=True)

        # --------------------
        # 8) MAE encoder / decoder 통과
        # --------------------
        x_enc = self.forward_mae_encoder(
            x,
            mask,
            inputs_embeds=inputs_embeds,  # ← 여기서 텍스트+이미지 프롬프트 같이 들어감
            input_ids=None,
            attention_mask=attention_mask
        )

        # --------------------
        # ★ 8) Forward 호출 (여기가 중요!)
        # --------------------
        # x_enc = self.forward_mae_encoder(
        #     x,
        #     mask,
        #     inputs_embeds=inputs_embeds,
        #     input_ids=input_ids,
        #     attention_mask=attention_mask, # 2D 마스크
            
        #     # [Bridge Mask Info]
        #     num_mp_tokens=num_keep,        # Hint 토큰 개수
        #     mp_start_idx=3,                # Hint 시작 인덱스
        #     hint_original_indices=keep_idx
        # )
        
        z = self.mar.forward_mae_decoder(x_enc, mask, image_shape=(m, n))
        loss = self.mar.forward_loss(z=z, target=gt_latents, mask=mask)

        return loss

    def affinity_map_t_i(
        self,
        img_emb,
        input_ids,
        attention_mask=None,
        t2i=False,
    ):
        IMAGE_TOKEN_INDEX = -200 

        # ------------------------------------------------
        # 1. 임베딩 및 정규화 (Embedding & Norm)
        # ------------------------------------------------
        if t2i:
            _, z_enc = self.extract_visual_feature(img_emb)
        else:
            z_enc = img_emb

        B, N_img, D = z_enc.shape

        safe_input_ids = input_ids.clone()
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        replace_id = pad_id if pad_id is not None else 0      
        safe_input_ids[safe_input_ids == IMAGE_TOKEN_INDEX] = replace_id

        text_emb = self.llm.get_input_embeddings()(safe_input_ids)

        z_norm = F.normalize(z_enc, dim=-1)
        t_norm = F.normalize(text_emb, dim=-1)

        # ------------------------------------------------
        # 2. Raw Affinity Matrix 계산
        # ------------------------------------------------
        # [B, N_img, T] : 이미지 패치와 텍스트 토큰 간의 코사인 유사도
        affinity_matrix = torch.einsum('bid,bjd->bij', z_norm, t_norm)  

        # ------------------------------------------------
        # 3. 기본 마스크 (Special Tokens & Stopwords)
        # ------------------------------------------------
        if attention_mask is not None:
            token_mask = attention_mask.bool()
        else:
            token_mask = torch.ones(input_ids.shape, dtype=torch.bool, device=input_ids.device)

        bos_id = getattr(self.tokenizer, "bos_token_id", None)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)

        if bos_id is not None: token_mask &= (input_ids != bos_id)
        if eos_id is not None: token_mask &= (input_ids != eos_id)
        if pad_id is not None: token_mask &= (input_ids != pad_id)
        token_mask &= (input_ids != IMAGE_TOKEN_INDEX)

        # [GPU Stopwords Filter]
        if self.text_token_filtering:
            # Safe indexing for look-up
            safe_ids_for_lookup = input_ids.clone()
            safe_ids_for_lookup[safe_ids_for_lookup < 0] = 0 # Handle -200
            safe_ids_for_lookup = safe_ids_for_lookup.clamp(max=len(self.tokenizer) - 1)
            
            # Lookup pre-computed mask
            is_content = self.content_token_mask[safe_ids_for_lookup]
            token_mask &= is_content

        # ------------------------------------------------
        # 4. Top-K Selection (Combined Strategy)
        # ------------------------------------------------
        if self.text_token_filtering:
            keyword_mask = torch.zeros_like(token_mask, dtype=torch.bool)

            # [Score 1] Text Self-Affinity
            sim_tt = torch.matmul(t_norm, t_norm.transpose(1, 2))
            self_sim_matrix_soft = F.softmax(sim_tt, dim=-1)
            raw_self_scores = self_sim_matrix_soft.sum(dim=1).squeeze()
            # raw_self_scores = sim_tt.mean(dim=-1) # [B, T]

            # [Score 2] Image-Text Similarity
            # raw_it_scores = affinity_matrix.mean(dim=1) # [B, T]
            raw_affinity_soft = F.softmax(affinity_matrix, dim=2) # [B, N, T]
            raw_it_scores = raw_affinity_soft.sum(dim=1).squeeze()
            
            
            for b in range(B):
                valid_b = token_mask[b]
                if not valid_b.any(): continue

                valid_idx_global = valid_b.nonzero(as_tuple=True)[0]
                
                valid_self = raw_self_scores[b, valid_b]
                valid_it = raw_it_scores[b, valid_b]

                # Min-Max Normalization (Vis 코드와 동일)
                min_s, max_s = valid_self.min(), valid_self.max()
                norm_self = (valid_self - min_s) / (max_s - min_s + 1e-8) if (max_s - min_s > 1e-8) else torch.zeros_like(valid_self)

                min_i, max_i = valid_it.min(), valid_it.max()
                norm_it = (valid_it - min_i) / (max_i - min_i + 1e-8) if (max_i - min_i > 1e-8) else torch.zeros_like(valid_it)

                # Equal weighting of normalized intra-modal and inter-modal affinities.
                combined_scores = norm_self + norm_it

                # Top-K Selection
                num_valid = int(combined_scores.shape[0])
                if num_valid == 0: continue

                k = max(self.text_min_tokens, int(num_valid * self.text_keep_ratio))
                k = min(k, num_valid)

                topk = torch.topk(combined_scores, k)
                selected_idx = valid_idx_global[topk.indices]
                keyword_mask[b, selected_idx] = True

            # ------------------------------------------------
            # ★ 5. Softmax & Aggregation (Vis 코드와 동기화된 부분)
            # ------------------------------------------------
            # (1) Softmax 적용: Raw Score가 아닌 '확률값'으로 변환
            #     dim=1 (Image Tokens) 방향으로 Softmax
            #     Visualization 코드의: attn_map_full = F.softmax(raw_affinity, dim=1) 과 동일
            affinity_prob = F.softmax(affinity_matrix, dim=1) # [B, N_img, T]

            # (2) Masking: 선택된 텍스트 토큰(keyword)만 남기기
            #     Visualization 코드의: torch.index_select(...) 과 동일한 효과
            keyword_mask_f = keyword_mask.unsqueeze(1).float() # [B, 1, T]
            affinity_masked = affinity_prob * keyword_mask_f   # [B, N_img, T]

            # (3) Summation: 선택된 맵들을 합쳐서 최종 히트맵(점수) 생성
            #     Visualization 코드의: selected_maps.sum(dim=2) 과 동일
            scores = affinity_masked.sum(dim=-1) # [B, N_img]

        else:
            # Ablation: aggregate all valid text tokens without discriminative filtering.
            
            affinity_prob = F.softmax(affinity_matrix, dim=1) # [B, N_img, T]
            token_mask_f = token_mask.unsqueeze(1).float() # [B, 1, T]
            affinity_masked = affinity_prob * token_mask_f
            scores = affinity_masked.sum(dim=-1) # [B, N_img]

        # ------------------------------------------------
        # 6. Final Normalization & Noise (MAE Masking용)
        # ------------------------------------------------
        s_min = scores.amin(dim=1, keepdim=True)
        s_max = scores.amax(dim=1, keepdim=True)
        scores_norm = (scores - s_min) / (s_max - s_min + 1e-6)
        noise = torch.rand_like(scores_norm) * self.noise_scale
        return scores_norm + noise, scores_norm

    def image2text_loss(self, data_dict):
        input_ids = data_dict['input_ids'].to(self.device)
        attention_mask = data_dict['attention_mask'].to(self.device)
        labels = data_dict['labels'].to(self.device)
        pixel_values = data_dict.get('pixel_values', None)
        if pixel_values is None:
            assert False
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            _, z_null = self.extract_visual_feature(
                torch.zeros(1, 16, 16, self.token_embed_dim,
                            dtype=self.dtype, device=self.device)
            )
            loss_null = z_null.mean() * 0.0
            print(f"No image found in this batch!", flush=True)
        else:
            x = pixel_values.to(dtype=self.dtype, device=self.device)
            x = self.encode(x)  # b m n c
            _, z_enc = self.extract_visual_feature(x)

            if self.grad_scale is not None:
                z_enc = _ScaleGradient.apply(z_enc, self.grad_scale)

            for i in range(input_ids.shape[0]):
                assert input_ids[i, 3] == IMAGE_TOKEN_INDEX
            input_ids = torch.cat(
                (
                    input_ids[:, :3],
                    IMAGE_TOKEN_INDEX * torch.ones(
                        input_ids.shape[0], 1088, dtype=torch.long, device=input_ids.device),
                    input_ids[:, 4:],
                ), dim=1
            )
            attention_mask = torch.cat(
                (
                    attention_mask[:, :3],
                    torch.ones(
                        attention_mask.shape[0], 1088, dtype=torch.long, device=attention_mask.device),
                    attention_mask[:, 4:],
                ), dim=1
            )
            labels = torch.cat(
                (
                    labels[:, :3],
                    labels[0, 3] * torch.ones(
                        labels.shape[0], 1088 , dtype=torch.long, device=labels.device),
                    labels[:, 4:],
                ), dim=1
            )
            inputs_embeds = z_enc.new_zeros(*input_ids.shape, self.llm.config.hidden_size)
            inputs_embeds[input_ids == IMAGE_TOKEN_INDEX] = z_enc.flatten(0, 1)
            inputs_embeds[input_ids != IMAGE_TOKEN_INDEX] = self.llm.get_input_embeddings()(
                input_ids[input_ids != IMAGE_TOKEN_INDEX])
            loss_null = 0.0

        output = self.llm_model(inputs_embeds=inputs_embeds,
                                attention_mask=attention_mask,
                                return_dict=True)

        last_hidden_state = output.last_hidden_state[:, :-1]
        labels = labels[:, 1:]
        last_hidden_state = last_hidden_state[labels >= 0]
        labels = labels[labels >= 0]
        logits = self.llm.get_output_embeddings()(last_hidden_state)

        loss_i2t = F.cross_entropy(input=logits, target=labels)

        return loss_i2t + loss_null

    def recon_loss(self, data_dict):

        x = data_dict['pixel_values'].to(dtype=self.dtype, device=self.device)
        x = self.encode(x)   # b m n c
        b, m, n, _ = x.shape
        gt_latents = x.clone().detach().view(b, m*n, -1)
        
        input_ids = data_dict['input_ids'].to(self.device)
        attention_mask = data_dict['attention_mask'].to(self.device)
        
        _, z_enc = self.extract_visual_feature(x)
        dumy_input_ids = data_dict['dumy_input_ids'].to(self.device)
        dumy_attention_mask = data_dict['dumy_attention_mask'].to(self.device)

        affinity_score_org, _ = self.affinity_map_t_i(
            img_emb=z_enc,                    # [B, N_img, D]
            input_ids=dumy_input_ids,
            attention_mask=dumy_attention_mask,
            t2i=False,
        )
        affinity_score = affinity_score_org[:, :m*n]   # [B, 1024]
        
        if self.grad_scale is not None:
            z_enc = _ScaleGradient.apply(z_enc, self.grad_scale)
        
        for i in range(input_ids.shape[0]):
            assert input_ids[i, 3] == IMAGE_TOKEN_INDEX
        
        # z_enc: [B, N_img, D], 여기서 앞 m*n개가 이미지 토큰
        z_img = z_enc[:, :m*n, :]                      # [B, m*n, D]

        num_img_tokens = z_img.shape[1]                # = m*n
        keep_idx = self.select_visual_hints(affinity_score)
        num_keep = keep_idx.shape[-1]

        # batch별 인덱스로 gather 해서 z_enc_sub 만들기
        z_enc_sub = torch.gather(
            z_img,                                      # [B, m*n, D]
            dim=1,
            index=keep_idx.unsqueeze(-1).expand(-1, -1, z_img.shape[-1])
        )                                               # [B, num_keep, D]

        # input_ids / attention_mask 에서도 이미지 토큰 자리 수를 num_keep 개로 맞춰줌
        input_ids = torch.cat(
            (
                input_ids[:, :3],
                IMAGE_TOKEN_INDEX * torch.ones(
                    input_ids.shape[0], num_keep,
                    dtype=torch.long, device=input_ids.device
                ),
                input_ids[:, 4:],
            ),
            dim=1
        )

        attention_mask = torch.cat(
            (
                attention_mask[:, :3],
                torch.ones(
                    attention_mask.shape[0], num_keep,
                    dtype=torch.long, device=attention_mask.device
                ),
                attention_mask[:, 4:],
            ),
            dim=1
        )

        # 이제 시퀀스 길이는: (원래 길이 - 1 placeholder + num_keep)
        # IMAGE_TOKEN_INDEX 수 = b * num_keep
        inputs_embeds = z_enc_sub.new_zeros(*input_ids.shape, self.llm.config.hidden_size)
        inputs_embeds[input_ids == IMAGE_TOKEN_INDEX] = z_enc_sub.flatten(0, 1)
        inputs_embeds[input_ids != IMAGE_TOKEN_INDEX] = self.llm.get_input_embeddings()(
            input_ids[input_ids != IMAGE_TOKEN_INDEX])
        
        mask = self.make_target_mask(affinity_score)

        x_enc = self.forward_mae_encoder(
            x, 
            mask, 
            inputs_embeds=inputs_embeds,
            input_ids=None,
            attention_mask=attention_mask
        )
        z = self.mar.forward_mae_decoder(x_enc, mask, image_shape=(m, n))
        loss = self.mar.forward_loss(z=z, target=gt_latents, mask=mask)
        return loss

    def forward(self, data, data_samples=None, mode='loss'):
        if mode == 'loss':
            return self.compute_loss(data_dict=data)
        else:
            raise NotImplementedError

    def compute_loss(self, data_dict):
        losses = {}
        for data_type, batch_data in data_dict.items():
            if 'text2image' in data_type:
                loss = self.text2image_loss(batch_data)
            elif 'image2text' in data_type:
                loss = self.image2text_loss(batch_data)
            elif 'recon' in data_type:
                loss = self.recon_loss(batch_data)
            else:
                raise NotImplementedError(f"Unknown data type: {data_type}")
            losses[f'loss_{data_type}'] = loss * self.loss_weights[data_type]
        return losses
