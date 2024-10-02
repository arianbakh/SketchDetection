import argparse
import gradio as gr
import numpy as np
from sketchdetection import ViTB16_SHAPE
from sketchdetection.datasets import SKETCHY_INPUT_SHAPE, get_class_index_to_name_map
from sketchdetection.models import get_model
from time import time
import torch


def get_pretrained_inference_model(checkpoint_path, model_architecture):
    model = get_model(model_architecture)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=str, help='Port of the service', required=True)
    parser.add_argument('--checkpoint', type=str, help='Path of the checkpoint', required=True)
    parser.add_argument('--architecture', type=str, help='Model architecture', required=True)
    args = parser.parse_args()
    inference_model = get_pretrained_inference_model(args.checkpoint, args.architecture)
    class_index_to_name = get_class_index_to_name_map()
    def run_inference(inputs):
        image = inputs["composite"]
        model_input = np.swapaxes(image, 0, -1)
        model_input = 255 - model_input
        model_input = np.expand_dims(model_input, axis=0)
        model_input = model_input.astype(np.float32)
        model_input = torch.from_numpy(model_input)
        start_time = time()
        with torch.no_grad():
            predictions = inference_model(model_input)
        print(f"Took {time() - start_time:.2f}s")
        predictions = torch.max(predictions, dim=1)[1]
        prediction = str(torch.squeeze(predictions).item())
        class_name = class_index_to_name[prediction]
        return class_name
    demo = gr.Interface(
        fn=run_inference,
        inputs=[
            gr.Sketchpad(
                crop_size=ViTB16_SHAPE if args.architecture == "ViTB16" else SKETCHY_INPUT_SHAPE,
                type="numpy",
                image_mode="RGB",
                height=600,
                brush=gr.Brush(colors=["#ffffff"], color_mode="fixed")
            )
        ],
        outputs=["text"]
    )
    demo.launch(server_port=int(args.port), server_name="0.0.0.0")
