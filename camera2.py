import os
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Scale, Compose

import torch
import matplotlib.pyplot as plt

from torchvision import transforms, models
from torch.autograd import Variable

from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import PIL

import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm

import cv2
import numpy as np



# helper functions
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images = batch[0].to(device)
        labels = batch[1].to(device)
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images = batch[0].to(device)
        labels = batch[1].to(device)
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}],{} train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, "last_lr: {:.5f},".format(result['lrs'][-1]) if 'lrs' in result else '',
            result['train_loss'], result['val_loss'], result['val_acc']))

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def predict(self, image):
        image = image.to(device)
        out = self(image).cpu().data.numpy()[0]
        softmax_out = self.softmax(out)
        res = np.argmax(softmax_out)
        return res, softmax_out

    def save_model(self, path):
        torch.save(self, path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))



class MasksModel(ImageClassificationBase):

    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet34(pretrained=pretrained) #downloading weights from this model when it was trained on ImageNet dataset
        # Replace last layer
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, xb):
        return self.network(xb)




image_size = (32, 32)

loader = Compose([Scale(image_size), ToTensor()])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
res_mask = {
    0: 'Mask weared incorrectly :/',
    1: 'Has Mask! :)',
    2: 'No mask :('
}

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = PIL.Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.to(device)  #assumes that you're using GPU

def prepare_image(image):
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.to(device)  #assumes that you're using GPU


def detect(image):
	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
	# 检测图像中的所有人脸
	faces = face_cascade.detectMultiScale(image_gray, 1.2, 6)
	print(f"{len(faces)} faces detected in the image.")
	return faces

font = cv2.FONT_HERSHEY_SIMPLEX
# fontScale
fontScale = 0.8
# Line thickness of 2 px
thickness = 1

model = torch.load("saved_model/resnet_model/model.pth", map_location='cpu')


if __name__ == "__main__":
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:

        faces = detect(frame)
        for x, y, width, height in faces:
            # get the face
            face = frame[y:y + height, x:x + width]
            # prepare image
            im = PIL.Image.fromarray(face)
            # predict
            result, output = model.predict(prepare_image(im))
            print(result)
            if result == 1:
                cv2.rectangle(frame, (x, y), (x + width, y + height), color=(0, 255, 0), thickness=2)
                cv2.putText(frame, "Positive %f" % output[0], (x + int(width * 0.05), y + int(height * 0.1)), font, 0.5,
                            (0, 255, 0), thickness, cv2.LINE_AA)
            elif result == 2:
                # print("No mask worn")
                cv2.rectangle(frame, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)
                cv2.putText(frame, "Negative %f" % output[1], (x + int(width * 0.05), y + int(height * 0.1)), font, 0.5,
                            (255, 0, 0), thickness, cv2.LINE_AA)
            else:
                cv2.rectangle(frame, (x, y), (x + width, y + height), color=(0, 0, 255), thickness=2)
                cv2.putText(frame, "Positive %f" % output[0], (x + int(width * 0.05), y + int(height * 0.1)), font, 0.5,
                            (0, 0, 255), thickness, cv2.LINE_AA)

        cv2.imshow("preview", frame)

        # Read next frame
        rval, frame = vc.read()

        # Exit
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break

    vc.release()
    cv2.destroyWindow("preview")







