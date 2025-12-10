# app.py
import gradio as gr
import os
from main import process_video  # uses the function you added

def run_and_return_file(uploaded_file):
    """
    uploaded_file will be a local filepath (type='filepath').
    We pass that to process_video() and return the output path.
    """
    if uploaded_file is None:
        return None
    # uploaded_file is a path to the uploaded file inside the runtime
    out_path = process_video(uploaded_file)
    # Make sure file exists before returning
    if os.path.exists(out_path):
        return out_path
    else:
        raise FileNotFoundError(f"Expected output not found: {out_path}")

# Create the interface
demo = gr.Interface(
    fn=run_and_return_file,
    inputs=gr.File(label="Upload a video", file_count="single", type="filepath"),
    outputs=gr.File(label="Processed video"),
    title="My Video Processor",
    description="Upload a video; it will be processed and returned."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)