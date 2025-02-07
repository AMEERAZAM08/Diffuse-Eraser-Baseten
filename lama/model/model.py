"""
The `Model` class is an interface between the ML model that you're packaging and the model
server that you're running it on.

The main methods to implement here are:
* `load`: runs exactly once when the model server is spun up or patched and loads the
   model onto the model server. Include any logic for initializing your model, such
   as downloading model weights and loading the model into memory.
* `predict`: runs every time the model server is called. Include any logic for model
  inference and return the model output.

See https://truss.baseten.co/quickstart for more.


Code done By AMEER AZAM
for more contact at https://linktr.ee/ameerazam22

"""
import base64
from PIL import Image
from io import BytesIO
from PIL import Image
import torch
from diffusers import DDIMScheduler, DiffusionPipeline
from diffusers.utils import load_image
import torch.nn.functional as F
from model.utils.pipeline import *

class Model:
    def __init__(self, **kwargs):
        #Basten Simple file 
        # Uncomment the following to get access
        # to various parts of the Truss config.
        # self._data_dir = kwargs["data_dir"]
        # self._config = kwargs["config"]
        # self._secrets = kwargs["secrets"]
        self.pipeline = None
        self.dtype = torch.float16
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
        self.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        self.model_path = "stabilityai/stable-diffusion-xl-base-1.0"

        # pipeline.enable_attention_slicing()
        # pipeline.enable_model_cpu_offload()

    def load(self):
        # Load model here and assign to self._model.
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.model_path,
            custom_pipeline="pipeline_stable_diffusion_xl_attentive_eraser",  #"./pipelines/pipeline_stable_diffusion_xl_attentive_eraser.py",
            scheduler=self.scheduler,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=self.dtype,
        ).to(self.device)


    def preprocess(self, request):
        encoded_image = request.get('input_img', None)
        encoded_mask = request.get('input_mask', None)
        print("Encoded image received:", encoded_image is not None)
        if encoded_image is not None or encoded_mask is None:
            try:
                image_data = base64.b64decode(encoded_image)
                input_img = Image.open(BytesIO(image_data))
                print("Input image loaded successfully:")

                mask_data = base64.b64decode(encoded_mask)
                mask_image = Image.open(BytesIO(mask_data))
                print("Input Mask loaded successfully:")
            except Exception as e:
                print("Error decoding or loading the image:", e)
                mask_image = None
        else:
            print("No image data found in the request.")
            input_img = None
            mask_image = None

        generate_args = {
            "input_img": input_img,
            "mask_img": mask_image,
        }
        print("Generate arguments:", generate_args)
        request["generate_args"] = generate_args
        return request

    def predict(self, request):
        # Run model inference here
        model_input = request.pop("generate_args")
        print("Model input received in predict method:", model_input)

        input_img = model_input['input_img']
        mask_img = model_input['mask_img']
        try:

            mask_img = mask_img.convert('L')
            result = remove_objects(pipeline=self.pipeline,
                            device=self.device,
                            edit_images=[input_img,mask_img])

        except Exception as e:
            print("Error during model inference:", e)
            return {'error': str(e)}

        # Encode the images
        encoded_output_images = None
    
        try:
            buffered = BytesIO()
            result.save(buffered, format='PNG')
            encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            encoded_output_images = encoded_image

        except Exception as e:
            print(f"Error encoding image with label '{label}':", e)

        return {'output_images': encoded_output_images}
