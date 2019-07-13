from __future__ import division, print_function
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
np.random.seed(98765)

#from tf_unet import image_gen
#from tf_unet import unet
#from tf_unet import util
import image_gen
import unet
import util

height = 160
width = 160
batch_size = 10
train_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/from_senior/0_cv_train.csv'
val_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/from_senior/0_cv_val.csv'

dataset, iters = image_gen.GetDataset(train_path, batch_size)
generator = image_gen.BladderDataProvider(height, width, dataset)
"""
x_test, y_test = generator(1)
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(8, 4))
ax[0].imshow(x_test[0, ..., 0], aspect="auto")
ax[1].imshow(y_test[0, ..., 1], aspect="auto")
#plt.show()
"""
net = unet.Unet(channels=generator.channels, n_class=generator.n_class, layers=4, features_root=64)
trainer = unet.Trainer(net, batch_size=batch_size, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
path = trainer.train(generator, '/DB/rhome/qyzheng/Desktop/qyzheng/PROGRAM/bladder/my_unet/output', training_iters=iters, epochs=100, display_step=100, 
	                 prediction_path='/DB/rhome/qyzheng/Desktop/qyzheng/PROGRAM/bladder/my_unet/result')

'''
x_test, y_test = generator(1)
print(x_test.shape)
print(y_test.shape)
prediction = net.predict("../unet_trained/model.ckpt", x_test)
print(prediction.shape)
'''
"""
fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 5))
ax[0].imshow(x_test[0,...,0], aspect="auto")
ax[1].imshow(y_test[0,...,1], aspect="auto")
mask = prediction[0,...,1] > 0.3
ax[2].imshow(mask, aspect="auto")
fig.tight_layout()
fig.savefig("../unet_trained/toy_problem.png")
"""
'''
plt.imshow(x_test[0,...,0])
plt.savefig("../unet_trained/toy_problem_image.png")
plt.imshow(y_test[0,...,1])
plt.savefig("../unet_trained/toy_problem_label.png")
mask = prediction[0,...,1] > 0.4
plt.imshow(mask)
plt.savefig("../unet_trained/toy_problem_pre.png")
'''