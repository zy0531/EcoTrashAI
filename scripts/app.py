
import gradio as gr
import os
import torch

from model import customized_efficinet_b0
from timeit import default_timer as timer

"""
with open('class_names.txt', "w") as f:
    f.write("\n".join(['BIODEGRADABLE', 'CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']))
"""
with open("class_names.txt", "r") as f:
    class_names = [food_name.strip() for food_name in  f.readlines()]
    
efficient_b0, transform = customized_efficinet_b0(num_classes=len(class_names))

efficient_b0.load_state_dict(
    torch.load( f="checkpoint_b0_long_best_model.pt",
               map_location=torch.device("cpu")) )

def predict(img):
    
    start_time = timer()
    
    img = transform(img).unsqueeze(0)
    
    efficient_b0.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(efficient_b0(img), dim=1)
    
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    pred_time = round(timer() - start_time, 5)
    
    return pred_labels_and_probs, pred_time

title = "EcoTrashAI"
description = "An EfficientNetB2 model to classify images of trash"
article = ""

example_list = [["examples/" + example] for example in os.listdir("examples")]

demo = gr.Interface(fn=predict, inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=4, label="Predictions"),
                             gr.Number(label="Prediction time (s)")],
                    examples=example_list, title=title, description=description, article=article)

demo.launch(debug=True, share=True)