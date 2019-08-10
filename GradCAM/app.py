from flask import Flask, render_template, request, make_response
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg
import torch
from . import gradcam, model, transforms
from .dataset import ImageDataset

from PIL import Image
from torch.utils.data import DataLoader


app = Flask(__name__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/send', methods=['post'])
def posttest():
    img_file = request.files['img_file']
    image = Image.open(img_file)
    image.save('img.jpg')
    dataset = ImageDataset(image_dir='.', file_paths=['img.jpg'], transform=transforms.get_transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    fig = gradcam.plotGradCAM(model.model, model.final_conv, model.fc_params, loader, device=device)
    canvas = FigureCanvasAgg(fig)
    buf = io.BytesIO()
    canvas.print_png(buf)
    data = buf.getvalue()
    """
    response = make_response(data)
    response.headers['Content-Type'] = 'image/jpg'
    response.headers['Content-Length'] = len(data)
    return response
    """
    qr_b64str = base64.b64encode(data).decode("utf-8")
    qr_b64data = "data:image/png;base64,{}".format(qr_b64str)
    return render_template('result.html', img=qr_b64data)


if __name__ == '__main__':
    app.run()
