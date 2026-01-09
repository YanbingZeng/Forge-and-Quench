import re
import numpy as np 

import torch
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from typing import List, Optional, Union
import inspect

from forge_and_quench.utils.constants import \
    GEN_TEMPLATE_CN, GEN_TEMPLATE_EN, \
    GENERATE_START_TOKEN, DEFAULT_GENERATE_IMAGE_TOKEN, GENERATE_END_TOKEN, \
    SYSTEM_MESSAGE,\
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN



def contains_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(pattern.search(text))


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class ForgePipeline(DiffusionPipeline):
    def __init__(
            self,
            forge_model,
            scheduler,
            tokenizer,
            latents_h = 27, # siglip model outputs: 729(27*27) tokens
            latents_w = 27,
        ):

        super().__init__()

        self.register_modules(
            forge_model=forge_model,
            scheduler=scheduler,
            tokenizer=tokenizer,
        )

        self.num_querys = self.forge_model.num_querys
        self.latents_h = latents_h
        self.latent_w = latents_w

        # special token
        self.tokenizer.add_tokens([GENERATE_START_TOKEN, GENERATE_END_TOKEN, DEFAULT_GENERATE_IMAGE_TOKEN])
        self.img_pad_id = self.tokenizer(DEFAULT_GENERATE_IMAGE_TOKEN).input_ids[0]


    def prepare_and_encode_prompt(self, prompt, template):
        # system
        system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"

        # user
        format_prompt = template.format(prompt=prompt)
        user_message = f"{DEFAULT_IM_START_TOKEN}user\n{format_prompt}{DEFAULT_IM_END_TOKEN}\n"

        # assistant
        rsp_img_tokens =  GENERATE_START_TOKEN + DEFAULT_GENERATE_IMAGE_TOKEN * self.num_querys + GENERATE_END_TOKEN
        assistant_message = f"{DEFAULT_IM_START_TOKEN}assistant\n{rsp_img_tokens}{DEFAULT_IM_END_TOKEN}\n"

        message = system_message + user_message + assistant_message

        input_ids = self.tokenizer(
            message, 
            add_special_tokens=False, 
            padding=False, 
            return_tensors='pt'
        )['input_ids']

        return input_ids[0]

    def prepare_latent_ids(self, batch_size, latent_height, latent_width, prompt_embeds_tokens, device, dtype):
        latent_image_ids = torch.zeros(latent_height, latent_width, 3)
        latent_image_ids[..., 0] = torch.Tensor([1]).repeat(latent_height, latent_width)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(latent_height)[:, None] + prompt_embeds_tokens
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(latent_width)[None, :] + prompt_embeds_tokens
        )

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = (
            latent_image_ids.shape
        )

        latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
        latent_image_ids = latent_image_ids.reshape(
            batch_size,
            latent_image_id_height * latent_image_id_width,
            latent_image_id_channels,
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    def prepare_prompt_ids(self, batch_size, prompt_embeds_tokens, device, dtype):
        text_ids = torch.zeros(prompt_embeds_tokens, 3)
        text_ids[..., 1] = torch.arange(prompt_embeds_tokens)
        text_ids[..., 2] = torch.arange(prompt_embeds_tokens)

        text_ids = text_ids[None, :].repeat(batch_size, 1, 1)

        return text_ids.to(device=device, dtype=dtype)

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        latent_height,
        latent_width,
        dtype,
        device,
        prompt_embeds_tokens
    ):
        # latents shape
        shape = (batch_size, latent_height*latent_width, num_channels_latents)
        latents = randn_tensor(shape, generator=torch.Generator().manual_seed(1234), device=device)
        latents = latents.to(dtype=dtype)

        latent_ids = self.prepare_latent_ids(
            batch_size,
            latent_height,
            latent_width,
            prompt_embeds_tokens,
            device,
            torch.float64
        )

        return latents, latent_ids


    @torch.inference_mode
    def __call__(
        self,
        prompt: str = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 4.5,
        generator=None
    ):
        device = self.forge_model.device
        if isinstance(prompt, str):
            batch_size = 1
        else:
            raise NotImplementedError

        # Prepare for prompt embs
        template = GEN_TEMPLATE_CN if contains_chinese(prompt) else GEN_TEMPLATE_EN
        prompt_input_ids = self.prepare_and_encode_prompt(prompt, template)
        null_prompt_input_ids = self.prepare_and_encode_prompt('', template)
        input_ids = [prompt_input_ids, null_prompt_input_ids]

        ids_and_mask = self.tokenizer.pad(
            {
                'input_ids': input_ids
            },
            padding=True,
            return_attention_mask=True, 
            padding_side='left', 
            return_tensors='pt'
        )
        input_ids = ids_and_mask.input_ids.to(device)
        attention_mask = ids_and_mask.attention_mask.to(device)

        inputs_embeds = self.forge_model.mllm.model.language_model.embed_tokens(input_ids)
        generate_mask = (input_ids == self.img_pad_id).unsqueeze(2).expand(-1, -1, inputs_embeds.shape[2]).to(device)
        inputs_embeds = inputs_embeds.masked_scatter(
            generate_mask, 
            self.forge_model.latent_queries.repeat(len(inputs_embeds), 1, 1)
        )

        prompt_embeds = self.forge_model.forward_prompt_embs(inputs_embeds, attention_mask)

        # Prepare for latents
        latents, latent_ids = self.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=self.forge_model.in_channels,
            latent_height=self.latents_h,
            latent_width=self.latent_w,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            prompt_embeds_tokens=prompt_embeds.shape[1]
        )

        prompt_ids = self.prepare_prompt_ids(
            batch_size=batch_size,
            prompt_embeds_tokens=prompt_embeds.shape[1],
            device=device,
            dtype=torch.float64,
        )


        # Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            None,
            sigmas,
            mu=mu,
        )

        # Denoising loop
        guidance = None
        latents = torch.cat([latents, latents], dim=0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype).to(device)
                
                noise_pred = self.forge_model.dit(
                    hidden_states=latents,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep / 1000,
                    img_ids=latent_ids,
                    txt_ids=prompt_ids,
                    guidance=guidance,
                    return_dict=False,
                )[0]

                # cfg
                pred_cond, pred_uncond = noise_pred.chunk(2, dim=0)
                noise_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

                # denoise
                noise_pred = torch.cat([noise_pred, noise_pred], 0)
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                progress_bar.update()

        adapter_feature = latents.chunk(2, dim=0)[0]

        return adapter_feature
