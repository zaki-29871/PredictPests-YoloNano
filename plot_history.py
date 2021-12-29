import matplotlib.pyplot as plt
import tools
import utils
import os
import utils.profile
import numpy as np

version = None
trend_kernel = 1
start = 100
profile = utils.profile.YoloNanoPaper()

version, loss_history = profile.load_history(version)
print('Number of epochs:', len(loss_history['test']))

loss_history['train'] = loss_history['train'][start:]
loss_history['test'] = loss_history['test'][start:]

if len(loss_history['test']) == 1:
    marker = 'o'
else:
    marker = 'o'

plt.figure()
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

p_train = plt.plot(loss_history['train'][(trend_kernel - 1):], label='Train', marker=marker)
p_test = plt.plot(loss_history['test'][(trend_kernel - 1):], label='Test', marker=marker)

test_loss_trend = np.array(loss_history['test'])
test_loss_trend = np.convolve(test_loss_trend, np.ones(trend_kernel,)) / trend_kernel
test_loss_trend = test_loss_trend[(trend_kernel - 1):-trend_kernel + 1]
# plt.plot(test_loss_trend, label='Test Trend', marker=marker, color=p_train[0].get_color(), linestyle='--')
plt.plot(test_loss_trend, label='Test Trend', marker=marker)

train_loss_trend = np.array(loss_history['train'])
train_loss_trend = np.convolve(train_loss_trend, np.ones(trend_kernel,)) / trend_kernel
train_loss_trend = train_loss_trend[(trend_kernel - 1):-trend_kernel + 1]
# plt.plot(train_loss_trend, label='Train Trend', marker=marker, color=p_test[0].get_color(), linestyle='--')
plt.plot(train_loss_trend, label='Train Trend', marker=marker)

plt.legend()
plt.show()