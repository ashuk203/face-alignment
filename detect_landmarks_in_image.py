import torch
import face_alignment
from face_alignment.utils import plot_face, landmark_error
from mpl_toolkits.mplot3d import Axes3D
import torch.nn as nn
import numpy as np
from skimage import io
import matplotlib.pyplot as plt


# Run the 3D face alignment on a test image, without CUDA.
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', flip_input=False)


input_img = io.imread('test/assets/new_dataset/5012.jpg')
rand_mask = np.random.rand(*input_img.shape)

# rand_mask = np.stack([rand_mask] * 3, axis=-1)
comp_img = (input_img * rand_mask).astype(int)
# comp_img = (0.9 * input_img).astype(int)

# Sets pixels to red...
# input_img[:22, :20] = [255, 0, 0]

preds, faces, inp, out = fa.get_landmarks(input_img)
preds = preds[-1]

ret_vals = fa.get_landmarks(comp_img)
if ret_vals is None:
    quit()
else:
    preds2, faces2, inp2, out2 = ret_vals
    preds2 = preds2[-1]


criterion = nn.MSELoss(reduction='sum')
loss = criterion(out, out2)

bounding_h = np.abs(faces[0][0] - faces[0][2])
bounding_w = np.abs(faces[0][2] - faces[0][3])

# err = landmark_error(preds, preds2, bounding_h * bounding_w)
print("Loss: ", loss.item())

# loss = Variable(loss, requires_grad = True)
loss.backward()
print(inp2.grad.shape)
plt.imshow(torch.sum(inp2.grad[0], dim=0))


plot_face(preds, faces[0], input_img)
plot_face(preds2, faces2[0], comp_img)
plt.show()
