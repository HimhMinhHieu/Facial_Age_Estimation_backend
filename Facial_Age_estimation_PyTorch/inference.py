import torch
import os
import glob
import cv2
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from .model import AgeEstimationModel

import torchvision.transforms as transforms

config = {
    'img_width': 128,
    'img_height': 128,
    'img_size': 128,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'model_name': 'resnet',
    'root_dir': '',
    'csv_path': '',
    'device': 'cpu',
    'leaky_relu': False,
    'epochs': 2,
    'batch_size': 128,
    'eval_batch_size': 256,
    'seed': 42,
    'lr': 0.0001,
    'wd': 0.001,
    'save_interval': 1,
    'reload_checkpoint': None,
    'finetune': 'weights/FA_DOCS/crnn-fa-base.pt',
    # 'finetune': None,
    'weights_dir': 'weights',
    'log_dir': 'logs',
    'cpu_workers': 4,
}

def faceBox(faceNet,frame):
    # print(frame)
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bbox=[]
    for i in range(detection.shape[2]):
        confidence = detection[0,0,i,2]
        if confidence > 0.7:
            x1 = int(detection[0,0,i,3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            bbox.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)
    return frame, bbox

faceProto = "D:/AI/final/2/web/FAE-Res50/backend/Facial_Age_estimation_PyTorch/opencv_face_detector.pbtxt"
faceModel = "D:/AI/final/2/web/FAE-Res50/backend/Facial_Age_estimation_PyTorch/opencv_face_detector_uint8.pb"

faceNet = cv2.dnn.readNet(faceModel, faceProto)

def inference(image_path, output_path= './what/out.png', **kwargs):
    model = AgeEstimationModel(input_dim=3, output_nodes=1, model_name=config['model_name'], pretrain_weights='IMAGENET1K_V2').to(config['device'])
    # Load the model using the latest checkpoint file
    # model.load_state_dict(torch.load("C:/Users/admin/Desktop/smile/backend/Facial_Age_estimation_PyTorch/trained/256.pt", weights_only=True))

    # model.load_state_dict(torch.load("D:/AI/final/2/web/FAE-Res50/backend/Facial_Age_estimation_PyTorch/trained/256.pt", weights_only=True))
    model.load_state_dict(torch.load("D:/AI/final/2/web/FAE-Res50/backend/Facial_Age_estimation_PyTorch/checkpoints/epoch-16-loss_valid-4.73.pt", weights_only=False, map_location=torch.device('cpu')))

    model.eval()
    with torch.no_grad():
        # image=Image.open(f'C:/Users/admin/Desktop/smile/backend/{image_path}').convert('RGB')

        face_cascade = cv2.CascadeClassifier('D:/AI/final/2/web/FAE-Res50/backend/Facial_Age_estimation_PyTorch/haarcascade_frontalface_default.xml')
        # directory_test = '/kaggle/working/test-img-visualization/img_test/'
        # if not os.path.exists(directory_test):
        #     os.makedirs(directory_test)

        img = cv2.imread(f'D:/AI/final/2/web/FAE-Res50/backend/{image_path}')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        directory = f'/kaggle/working/test-img-visualization/img_test/test_idx_infe'
        if not os.path.exists(directory):
            os.makedirs(directory)
        print(faces)
        if (len(faces) != 0):
            # print('abc')
            image_path_tmp = f"D:/AI/final/2/web/FAE-Res50/backend/upload/img_test_tmp/{image_path}"
            for (x, y, w, h) in faces:
                #             print("1")
                #             cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                face = img[y:y + h, x:x + w]
                cv2.imwrite(image_path_tmp, face)
        else:
            frame, bbox = faceBox(faceNet, img)
            image_path_tmp = f"D:/AI/final/2/web/FAE-Res50/backend/upload/img_test_tmp/{image_path}"
            print(f'bbox:{bbox}')
            for bb in bbox:
                face = frame[bb[1]:bb[3], bb[0]:bb[2]]
                print(f'face:{face}')
                cv2.imwrite(image_path_tmp, face)
            if (bbox == []):
                image_path_tmp = image_path

        image = Image.open(image_path_tmp).convert('RGB')
        transform = T.Compose([T.Resize(((config['img_width'], config['img_height']))),
                               T.ToTensor(),
                               T.Normalize(mean=config['mean'],
                                           std=config['std'])
                              ])
        input_data = transform(image).unsqueeze(0).to(config['device'])
        outputs = model(input_data)  # Forward pass through the model

        # Extract the age estimation value from the output tensor
        age_estimation = outputs.item()
        text = f"{age_estimation:.2f}"

    return output_path, text, image_path

# path = "./checkpoints/"
# checkpoint_files = glob.glob(os.path.join(path, 'epoch-*-loss_valid-*.pt'))
# latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
