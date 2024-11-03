from diffusers import DDIMScheduler, StableDiffusionPipeline

import torch
import torch.nn as nn


class StableDiffusion(nn.Module):
    def __init__(self, args, t_range=[0.02, 0.98]):
        super().__init__()

        self.device = args.device
        self.dtype = args.precision
        print(f'[INFO] loading stable diffusion...')

        model_key = "stabilityai/stable-diffusion-2-1-base"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, torch_dtype=self.dtype,
        )

        pipe.to(self.device)
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype,
        )

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.t_range = t_range
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings
    
    
    def get_noise_preds(self, latents_noisy, t, text_embeddings, guidance_scale=100):
        latent_model_input = torch.cat([latents_noisy] * 2)
            
        tt = torch.cat([t] * 2)
        noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
        
        return noise_pred


    def get_sds_loss(
        self, 
        latents,
        text_embeddings, 
        guidance_scale=100, 
        grad_scale=1,
    ):
        # TODO: Implement the loss function for SDS
        t = torch.randint(1, self.num_train_timesteps, (1,), device=self.device)
        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        noise = torch.randn_like(latents)
        noisy_latents  = self.scheduler.add_noise(latents, noise, t)
        with torch.no_grad():
            noise_pred = self.get_noise_preds(noisy_latents, t, text_embeddings, guidance_scale)
        grad  = w * (noise_pred - noise)
        target = (latents - grad).detach()
        loss_sds = 0.5 * torch.nn.MSELoss()(latents, target) * grad_scale
        return loss_sds



    
    def get_pds_loss(
        self, src_latents, tgt_latents, 
        src_text_embedding, tgt_text_embedding,
        guidance_scale=7.5, 
        grad_scale=1,
    ):
        
        # TODO: Implement the loss function for PDS
        # cmd: python main.py --step 200 --loss_type pds --guidance_scale 7.5 --prompt "" --edit_prompt "" --src_img_path "" 
        t = torch.randint(self.min_step, self.max_step, (1,), device=self.device)
        z_t = torch.randn_like(src_latents)
        z_t_1 = torch.randn_like(src_latents)
        x_src_t = self.scheduler.add_noise(src_latents, z_t, t)
        x_tgt_t = self.scheduler.add_noise(tgt_latents, z_t, t)
        x_src_t_1 = self.scheduler.add_noise(src_latents, z_t_1, t-1)
        x_tgt_t_1 = self.scheduler.add_noise(tgt_latents, z_t_1, t-1)
        sigma_t = torch.sqrt((1 - self.alphas[t-1]) / (1 - self.alphas[t]) * self.scheduler.betas.to(self.device)[t])
        with torch.no_grad():
            mu_src = torch.sqrt(self.alphas[t-1]) * src_latents + torch.sqrt(1 - self.alphas[t-1] - sigma_t.square()) * self.get_noise_preds(x_src_t, t, src_text_embedding, guidance_scale)
            mu_tgt = torch.sqrt(self.alphas[t-1]) * tgt_latents + torch.sqrt(1 - self.alphas[t-1] - sigma_t.square()) * self.get_noise_preds(x_tgt_t, t, tgt_text_embedding, guidance_scale)
        z_t_src = (x_src_t_1 - mu_src)/sigma_t
        z_t_tgt = (x_tgt_t_1 - mu_tgt)/sigma_t
        grad = (z_t_src - z_t_tgt).detach()
        target = (tgt_latents - grad).detach()
        loss_pds = 0.5 * torch.nn.MSELoss()(tgt_latents, target) * grad_scale
        return loss_pds
    
    def get_vsd_loss(
        self, latents, text_embeddings,
        guidance_scale=7.5, 
        grad_scale=1,
    ):  
        def init_lora():
            from diffusers.models.attention_processor import LoRAAttnProcessor
            from diffusers.loaders import AttnProcsLayers
            pretrained_model_name_or_path_lora = "stabilityai/stable-diffusion-2-1"
            pipe_lora = StableDiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path_lora,
            ).to(self.device)
            self.unet_lora = pipe_lora.unet
            breakpoint()
            del pipe_lora
            lora_attn_procs = {}
            for name in self.unet_lora.attn_processors.keys():
                cross_attention_dim = (
                    None
                    if name.endswith("attn1.processor")
                    else self.unet_lora.config.cross_attention_dim
                )
                if name.startswith("mid_block"):
                    hidden_size = self.unet_lora.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(self.unet_lora.config.block_out_channels))[
                        block_id
                    ]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = self.unet_lora.config.block_out_channels[block_id]

                lora_attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
                )

            self.unet_lora.set_attn_processor(lora_attn_procs)

            self.lora_layers = AttnProcsLayers(self.unet_lora.attn_processors).to(
                self.device
            )
            self.lora_layers._load_state_dict_pre_hooks.clear()
            self.lora_layers._state_dict_hooks.clear()
                    
        if not hasattr(self, "unet_lora"):
            init_lora()
        
        def apply_lora(noisy_latents, t):

            
            latent_model_input = torch.cat([noisy_latents] * 2)
            
            tt = torch.cat([t] * 2)
            noise_pred = self.unet_lora(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)

            return noise_pred
        
        t = torch.randint(1, self.num_train_timesteps, (1,), device=self.device)
        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        noise = torch.randn_like(latents)
        noisy_latents  = self.scheduler.add_noise(latents, noise, t)
        with torch.no_grad():
            noise_pred = self.get_noise_preds(noisy_latents, t, text_embeddings, guidance_scale)
        noise_lora = apply_lora(noisy_latents, t)
        
        grad = w * (noise_lora - noise_pred)
        target = (latents - grad).detach()
        loss_vsd = 0.5 * torch.nn.MSELoss()(noise_lora, target) * grad_scale
        return loss_vsd

    
    @torch.no_grad()
    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    @torch.no_grad()
    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents
