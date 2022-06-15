import os
import time
import torch
import typing
from torchvision.transforms import functional as TF
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from clip_custom import clip
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from cog import BasePredictor, Path, Input, File


class Predictor(BasePredictor):

    diffusion = None
    glid_model = None
    clip_model = None
    clip_preprocessor = None
    ldm_model = None
    device = None

    def setup(self):
        def set_requires_grad(model, value):
            for param in model.parameters():
                param.requires_grad = value

        setup_start_time = time.time()
        print("Doing setup")
        print(os.popen("nvidia-smi").read())
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)
        model_params = {
            'attention_resolutions': '32,16,8',
            'class_cond': False,
            'diffusion_steps': 1000,
            'rescale_timesteps': True,
            'timestep_respacing': '27',  # Modify this value to decrease the number of
            # timesteps.
            'image_size': 32,
            'learn_sigma': True,
            'noise_schedule': 'cosine',
            'num_channels': 320,
            'num_head_channels': 64,
            'num_res_blocks': 3,
            'encoder_channels': 768,
            'resblock_updown': True,
            'use_fp16': True,
            'use_scale_shift_norm': True
        }

        model_config = model_and_diffusion_defaults()
        model_config.update(model_params)

        # Load Glid Model
        print(f'Loading glid model at {time.time() - setup_start_time}')
        self.glid_model, self.diffusion = create_model_and_diffusion(**model_config)
        self.glid_model.load_state_dict(torch.load("/ema-latest.pt", map_location='cpu'))
        self.glid_model.requires_grad_(False).eval().to(self.device)

        print(f'Converting glid model at {time.time() - setup_start_time}')
        if model_config['use_fp16']:
            self.glid_model.convert_to_fp16()
        else:
            self.glid_model.convert_to_fp32()

        # Load Latent Diffusion Model
        print(f'Loading diffusion model at {time.time() - setup_start_time}')
        config = OmegaConf.load("./vq-f8/config.yaml")
        pl_sd = torch.load("/vq-f8/model.ckpt", map_location="cpu")
        sd = pl_sd["state_dict"]
        self.ldm_model = instantiate_from_config(config.model)
        self.ldm_model.load_state_dict(sd, strict=False)
        self.ldm_model.to(self.device)
        self.ldm_model.eval()
        set_requires_grad(self.ldm_model, False)

        # Load Clip Model
        print(f'Loading clip model at {time.time() - setup_start_time}')
        self.clip_model, self.clip_preprocessor = clip.load('ViT-L/14', device=self.device, jit=False)
        self.clip_model.eval().requires_grad_(False)
        set_requires_grad(self.clip_model, False)
        print(f'Setup complete')

    def predict(self,
                prompt: str = Input(description="Image prompt"),
                negative: str = Input(description="Negative image prompt", default=None),
                batch_size: int = Input(description="Number of images to generate", default=1, ge=1, le=20)
                # model_size: str = Input(description="Size of the model", default="MINI", choices=["MINI"])
                ) -> typing.List[Path]:

        def model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = self.glid_model(combined, ts, **kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)

        def save_sample(i, sample, clip_score=False) -> str:
            for k, image in enumerate(sample['pred_xstart'][:batch_size]):
                image = 2*image
                im = image.unsqueeze(0)
                im_quant, _, _ = self.ldm_model.quantize(im)
                out = self.ldm_model.decode(im_quant)
                out = TF.to_pil_image(out.squeeze(0).add(1).div(2).clamp(0, 1))
                filename = f'output/_progress_{i * batch_size + k:05}.png'
                out.save(filename)
                print(f'Saved image {filename} at {time.time() - predict_start_time}')

                if clip_score:
                    image_emb = self.clip_model.encode_image(self.clip_preprocessor(out).unsqueeze(0).to(self.device))
                    image_emb_norm = image_emb / image_emb.norm(dim=-1, keepdim=True)
                    similarity = torch.nn.functional.cosine_similarity(image_emb_norm, text_emb_norm, dim=-1)
                    final_filename = f'output/_{similarity.item():0.3f}_{i * batch_size + k:05}.png'
                    os.rename(filename, final_filename)
                return filename

        predict_start_time = time.time()
        print("Doing prediction")
        print(os.popen("nvidia-smi").read())
        all_images = []

        guidance_scale = 5.0
        width = 256
        height = 256
        num_batches = 1

        print(f'Tokenizing prompt at {time.time() - predict_start_time}')
        text = clip.tokenize([prompt]*batch_size, truncate=True).to(self.device)
        print(f'Test 1 {time.time() - predict_start_time}')
        text_emb, text_out = self.clip_model.encode_text(text, out=True)
        print(f'Test 2 {time.time() - predict_start_time}')
        text_emb_norm = text_emb[0] / text_emb[0].norm(dim=-1, keepdim=True)
        print(f'Test 3 {time.time() - predict_start_time}')
        text_out = text_out.permute(0, 2, 1)
        print(f'Test 4 {time.time() - predict_start_time}')
        text_blank = clip.tokenize([negative]*batch_size).to(self.device)
        print(f'Test 5 {time.time() - predict_start_time}')
        text_emb_blank, text_out_blank = self.clip_model.encode_text(text_blank, out=True)
        print(f'Test 6 {time.time() - predict_start_time}')
        text_out_blank = text_out_blank.permute(0, 2, 1)
        print(f'Test 7 {time.time() - predict_start_time}')
        kwargs = { "xf_proj": torch.cat([text_emb, text_emb_blank], dim=0), "xf_out": torch.cat([text_out, text_out_blank], dim=0) }
        print(f'Test 8 {time.time() - predict_start_time}')

        sample_fn = self.diffusion.plms_sample_loop_progressive

        print(f'Generating images at {time.time() - predict_start_time}')
        for i in range(num_batches):
            samples = sample_fn(
                model_fn,
                (batch_size*2, 4, int(height/8), int(width/8)),
                clip_denoised=False,
                model_kwargs=kwargs,
                cond_fn=None,
                device=self.device,
                progress=True,
            )
            for j, sample in enumerate(samples):
                if j > 0 and j % 50 == 0:
                    all_images.append(Path(save_sample(i, sample)))
                    yield Path(save_sample(i, sample))
            all_images.append(Path(save_sample(i, sample)))
        print(f'Prediction done at {time.time() - predict_start_time}')
        return all_images


