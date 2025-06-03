🧠 Depth Estimation with RGB Images using PyTorch

This project trains and evaluates various deep learning models to perform monocular depth estimation from RGB images using datasets like NYU Depth V2 and DIML RGB-D.

📦 Project Structure

.
├── datasets.py          # Custom PyTorch Datasets: NYUDepthDataset and DIMLRGBDDataset
├── models.py            # Model definitions: ResNet18/34, DenseNet, MobileNetV2, EfficientNetB0
├── train.py             # Training pipeline (train_model function)
├── test.py              # Evaluation and visualization (test_model function)
├── main.py              # CLI entry point to train/test models
├── losses/              # CSV files with training and validation loss logs
├── weigths/             # Pretrained model weights for NYU and DIML
├── README.txt           # Project description and usage

🚀 Features

- Plug-and-play model architecture (--model)
- Automatic dataset download (for NYU)
- DIML RGB-D fixed-path loading from /datasets/
- Augmentation (for training)
- Evaluation with MSE loss
- Visualization of predicted depth maps
- Easily extensible for other RGB-D datasets

🧠 Supported Models

Model Name     | Description
---------------|---------------------------
resnet18       | Lightweight baseline
resnet34       | Deeper ResNet
densenet       | DenseNet-121
mobilenet      | MobileNetV2
efficientnet   | EfficientNet-B0

📚 Supported Datasets

Dataset     | Path / Source
------------|-------------------------------------------------------------
nyudepth    | KaggleHub: soumikrakshit/nyu-depth-v2
diml        | RGB-D Zip Files: https://drive.google.com/drive/folders/1lexd7hia3oFVbZW3im_sEQpGUss1WOcE?usp=sharing

⚙️ Setup

run thşs line before requirements if you intent to use Cuda pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
cuda driver 12.8 or higher must be installed.

If you're using NYU, install KaggleHub:

pip install kagglehub

🏁 Usage

📌 Train
python main.py --mode train --model efficientnet --dataset nyudepth

📌 Test + Visualize
python main.py --mode test --model efficientnet --dataset nyudepth --model_path weigths/efficientnet_best_nyudepth.pth --visualize_index 25

📈 Output

- Best model is saved in the \texttt{weigths/} folder as: MODELNAME\_best\_DATASET.pth
  - Example: \texttt{resnet34\_best\_dataset2.pth} refers to ResNet34 trained on DIML dataset.
- CSV training logs are saved in the \texttt{losses/} folder as: MODELNAME\_losses\_DATASET.csv
  - Each file contains per-epoch training and validation loss.
  - CSV file contains non normalized losses. Values must be multiplyed with 255*255

📌 Notes

- nyudepth downloads automatically using KaggleHub.
- diml RGB-D dataset zip files (rgb224.zip and depth224.zip) must be downloaded from the link above and extracted into:
  /datasets/rgb224/   ← extracted contents of rgb224.zip  
  /datasets/depth224/ ← extracted contents of depth224.zip
- \texttt{dataset2} in file names refers to DIML dataset.
- if you intend to use pretrained weigths they also need to be dowbloaded from google drive to a directory named /weigths


🤖 Author

Developed by [Ufuk Ulas Tokat, İbrahim Mahir Akbaş, Hüseyin Berke Fırat]  
Feel free to use and modify for research and educational purposes.