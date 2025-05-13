from flask import Flask, request, jsonify
import io
from PIL import Image
import moondream as md
from moondream.types import EncodedImage, OnnxEncodedImage
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-01-09",
    trust_remote_code=True,
    device_map={"": "cuda"}
)

app = Flask(__name__)

# model = md.vl(model="./moondream-2b-int8.mf")

def encode_request_img(input_req):
    if 'input' not in input_req.files:
        return jsonify({"error": "No file part in the request"}), 400
    form_file = input_req.files['input']
    if form_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    try:
        # Read the uploaded file and convert it to a PIL Image
        image = Image.open(io.BytesIO(form_file.read()))
        image = model.encode_image(image)
        return image
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {e}"}), 400

@app.route("/")
def index():
    return {"ok":"we good"}


@app.route("/caption", methods=["POST"])
def caption_shit():
    get_img = encode_request_img(request)
    if type(get_img) is not OnnxEncodedImage:
        return get_img
    else:
        caption = model.caption(get_img)
        return jsonify({"success":caption}), 200 


@app.route("/detect", methods=["POST"])
def detect_shit():
    get_img = encode_request_img(request)
    try: 
        obj_name = request.form["object_name"]
    except Exception as _:
        return jsonify({"status": "PLEASE PROVIDE INDEX NAME!"}), 400

    if type(get_img) is not OnnxEncodedImage:
        return get_img 
    else:
        bbox = model.detect(get_img, obj_name)
        return jsonify({"success":bbox}), 200 

app.run(port=9000, host="0.0.0.0", debug=True)



