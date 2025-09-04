# app.py
import os, shlex, tempfile, subprocess, time, zipfile, glob, shutil, sys
from pathlib import Path
import gradio as gr

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "inference" / "new_map_inference.py"     # adjust if needed
EXAMPLES_DIR = ROOT / "examples"           # put a few small demo images here

# Model presets: map dropdown names -> weight file paths (relative or absolute)
MODEL_PRESETS = {
    "UNet (default)": ROOT / "models" / "unet_best_weight.pth",   # change if different
    "Upload…": None,
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

def _list_examples():
    if EXAMPLES_DIR.exists():
        ex = []
        for p in sorted(EXAMPLES_DIR.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                ex.append([str(p)])
        return ex
    return []

def run_inference(input_img_path, model_choice, uploaded_model, model_type, unseen, vectorization, extra_args):
    # Validate inputs
    if not input_img_path:
        return [], None, "No input image provided."
    # Resolve model path
    model_path = None
    if model_choice != "Upload…":
        mp = MODEL_PRESETS.get(model_choice)
        if mp is None:
            return [], None, f"Preset model '{model_choice}' is not configured."
        model_path = Path(mp)
        if not model_path.exists():
            return [], None, f"Preset model not found: {model_path}"
    else:
        if not uploaded_model:
            return [], None, "Please upload a .pth model."
        model_path = Path(uploaded_model.name)

    # Prepare temp working dir so all outputs land in one place
    work = Path(tempfile.mkdtemp(prefix="histmapseg_"))
    logs = []
    def log(line):
        logs.append(line)
        return "\n".join(logs[-500:])

    # Copy inputs into workdir
    in_img = work / Path(input_img_path).name
    shutil.copy2(input_img_path, in_img)
    if model_choice == "Upload…":
        up = work / Path(model_path).name
        shutil.copy2(model_path, up)
        model_path = up
    # Build CLI
    cmd = [
        sys.executable, "-u", str(SCRIPT),
        "--model_type", model_type,
        "--model", str(model_path),
        "--input_map_path", str(in_img),
    ]
    if unseen: cmd.append("--unseen")
    if vectorization: cmd.append("--vectorization")
    # extra args (freeform)
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    # Run
    log(f"cwd: {work}")
    log(f"cmd: {' '.join(shlex.quote(c) for c in cmd)}")
    try:
        proc = subprocess.Popen(
            cmd, cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        # stream logs
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                yield gr.update(value=None), gr.update(value=None), log(line.rstrip())
        code = proc.wait()
    except FileNotFoundError as e:
        yield [], None, log(f"ERROR: {e}")
        return
    except Exception as e:
        yield [], None, log(f"EXCEPTION: {e}")
        return

    # Collect outputs from workdir
    imgs = []
    for ext in IMAGE_EXTS:
        imgs.extend(sorted(str(p) for p in work.glob(f"*{ext}")))
    # Pack everything as a zip for download
    zip_path = work / "outputs.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(work.iterdir()):
            # Don’t re-zip the zip
            if p == zip_path: 
                continue
            zf.write(p, arcname=p.name)

    # Final return
    final_log = "\n".join(logs[-2000:])
    yield imgs, str(zip_path), final_log

with gr.Blocks(title="Historical Map Vectorization") as demo:
    gr.Markdown("### Map segmentation & vectorization")
    with gr.Row():
        input_img = gr.Image(type="filepath", label="Input map image", sources=["upload", "clipboard"])
        with gr.Column():
            model_choice = gr.Dropdown(list(MODEL_PRESETS.keys()), value="UNet (default)", label="Model")
            uploaded_model = gr.File(label="Upload .pth model", file_count="single", visible=False)
            model_type = gr.Dropdown(choices=["unet"], value="unet", label="--model_type")
            unseen = gr.Checkbox(value=True, label="--unseen")
            vectorization = gr.Checkbox(value=True, label="--vectorization")
            extra_args = gr.Textbox(label="Extra CLI args", placeholder="e.g. --foo 123 --bar")
            run_btn = gr.Button("Run", variant="primary")

    # Show/Hide upload control
    def _toggle_upload(choice):
        return gr.update(visible=(choice == "Upload…"))
    model_choice.change(_toggle_upload, inputs=model_choice, outputs=uploaded_model)

    gallery = gr.Gallery(label="Outputs", columns=2, height=400)
    zip_out = gr.File(label="Download all outputs (.zip)")
    logs = gr.Textbox(label="Log", lines=16)

    run_btn.click(
        fn=run_inference,
        inputs=[input_img, model_choice, uploaded_model, model_type, unseen, vectorization, extra_args],
        outputs=[gallery, zip_out, logs]
    )

    # Examples (optional)
    ex = _list_examples()
    if ex:
        gr.Examples(
            examples=ex,
            inputs=[input_img],
            label="Examples",
            examples_per_page=10
        )

if __name__ == "__main__":
    demo.queue(api_open=False).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
