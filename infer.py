import os
import argparse
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import AutoProcessor

from forge_and_quench.models.modeling_forge_and_quench import ForgeModel
from forge_and_quench.pipelines.forge_pipeline import ForgePipeline


def get_pipe(t2i_model, forge_and_quench_model_path, t2i_model_path, qwen25_vl_path, dtype=torch.bfloat16, device=torch.device('cuda:0')):
    # forge pipeline
    forge_model_path = os.path.join(forge_and_quench_model_path, 'forge_model')
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(forge_model_path, subfolder="scheduler")
    forge_model = ForgeModel(qwen25_vl_path, forge_model_path)
    tokenizer = AutoProcessor.from_pretrained(qwen25_vl_path).tokenizer

    forge_pipe = ForgePipeline(
        forge_model = forge_model,
        scheduler = scheduler,
        tokenizer = tokenizer,
    ).to(device, dtype)

    # quench pipeline
    extra_kwargs = {}
    if t2i_model == 'longcat-image':
        from forge_and_quench.models.transformer_longcat_image import LongCatImageTransformer2DModel
        from forge_and_quench.pipelines.quench_pipeline_longcat_image import LongCatImagePipeline as quench_pipeline
        quench_model_path = os.path.join(forge_and_quench_model_path, 'quench_model/quench_model_longcat_image.bin')
        transformer = LongCatImageTransformer2DModel.from_pretrained(t2i_model_path, subfolder='transformer', torch_dtype=dtype)
        extra_kwargs['transformer'] = transformer
    elif t2i_model == 'flux1-dev':
        from forge_and_quench.pipelines.quench_pipeline_flux1_dev import FluxPipeline as quench_pipeline
        quench_model_path = os.path.join(forge_and_quench_model_path, 'quench_model/quench_model_flux1_dev.bin')
    else:
        raise NotImplementedError('FaQ only support FLUX.1-dev and Longcat-Image now')

    quench_pipe = quench_pipeline.from_pretrained(
        t2i_model_path,
        torch_dtype=dtype,
        **extra_kwargs,
    ).to(device)
    quench_pipe.prepare_for_quench(quench_model_path)

    return forge_pipe, quench_pipe


def infer_flux1_dev(args):
    forge_pipe, quench_pipe = get_pipe(
        t2i_model='flux1-dev',
        forge_and_quench_model_path=args.forge_and_quench_model_path,
        t2i_model_path=args.t2i_model_path,
        qwen25_vl_path=args.qwen25_vl_path,
        device=device,
        dtype=torch.bfloat16
    )
    # 1. Forge
    adapter_feature = forge_pipe(prompt)
    
    # 2. Quench
    # 2.1 faq_scale controls the influence of the adapter_feature. Setting it to 0 is equivalent to the original t2i model.
    images = quench_pipe(
        prompt,
        height=1024,
        width=1024,
        adapter_feature=adapter_feature,
        faq_scale=0.50,
        num_inference_steps=28,
        generator=generator,
    )[0]
    images[0].save('./out_faq_flux1_dev.png')  # set `faq_scale=0.0` to generate `out_t2i_flux1_dev.png`



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="inference args for FaQ")
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--qwen25_vl_path', type=str, help='Model path of Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--forge_and_quench_model_path', type=str, help='Model path of Forge-and-Quench')
    parser.add_argument('--t2i_model', type=str, choices=['flux1-dev', 'longcat-image'])
    parser.add_argument('--t2i_model_path', type=str, help='Model path of T2I Foundation Model')
    args = parser.parse_args()


    device = torch.device('cuda')
    generator = torch.Generator('cpu').manual_seed(args.seed)

    prompt = "A photograph capturing a cat gracefully jumping down from a wall, its fur sleek and shimmering in the soft afternoon light. The wall is made of rustic brick, adding texture and contrast to the scene. The background is a serene garden with lush greenery, providing a natural and tranquil setting. The composition uses a mid-air shot to freeze the cat's motion, highlighting its agility and elegance. The lighting is warm and natural, casting gentle shadows that enhance the depth and detail of the cat's form. The overall atmosphere is one of dynamic movement and serene beauty, emphasizing the cat's playful and adventurous spirit."

    # 1. init pipe
    forge_pipe, quench_pipe = get_pipe(
        t2i_model=args.t2i_model,
        forge_and_quench_model_path=args.forge_and_quench_model_path,
        t2i_model_path=args.t2i_model_path,
        qwen25_vl_path=args.qwen25_vl_path,
        device=device,
        dtype=torch.bfloat16
    )
    # 2. Forge Inference
    adapter_feature = forge_pipe(prompt)
    # 3. Quench Inference
    # 3.1 faq_scale controls the influence of the adapter_feature. Setting it to 0 is equivalent to the original t2i model.
    # 3.2 In the LongCat-Image pipeline, `enable_cfg_renorm` improves image details, and FaQ builds upon this to further boost quality. It is recommended to enable both for the best performance.
    if args.t2i_model == 'flux1-dev':
        images = quench_pipe(
            prompt,
            height=1024,
            width=1024,
            adapter_feature=adapter_feature,
            faq_scale=0.50,
            num_inference_steps=28,
            generator=generator,
        )[0]
        images[0].save('./out_faq_flux1_dev.png')  # set `faq_scale=0.0` to generate `out_t2i_flux1_dev.png`
    elif args.t2i_model == 'longcat-image':
        images = quench_pipe(
            prompt,
            height=1024,
            width=1024,
            adapter_feature=adapter_feature,
            faq_scale=0.35,
            num_inference_steps=50,
            generator=generator,
            enable_cfg_renorm=True
        )[0]
        images[0].save('./out_faq_longcat_image.png') # set `faq_scale=0.0` to generate `out_t2i_longcat_image.png`
        # set `enable_cfg_renorm=False` to generate `out_faq_longcat_image_wo_cfg_renorm.png` and `out_t2i_longcat_image_wo_cfg_renorm.png`
