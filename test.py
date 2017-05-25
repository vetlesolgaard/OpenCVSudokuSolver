import numpy as np
import cv2

data = np.load('validation.npy')
train, labels = data
for i in range(0, len(labels)):
    print(train[i])
    cv2.imshow('img'+str(i), train[i])
    print('Label: ', labels[i])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit(0)
    inp = raw_input('')
