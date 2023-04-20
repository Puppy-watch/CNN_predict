import numpy as np
import tflite_runtime.interpreter as tflite
interpreter = tflite.Interpreter(model_path='test_model.tflite')
interpreter.allocate_tensors() # tensors init

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']

input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32) # generate random input_data
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke() # call

output_data = interpreter.get_tensor(output_details[0]['index'])
print(np.argmax(output_data), np.max(output_data)) # 분류한 label, 추론 최댓값


"""
import pandas as pd
import numpy as np
from keras.utils import to_categorical
import tensorflow as tf

input_data = pd.read_csv('./data/test_data.csv')
test_x = input_data.drop('label', axis=1)
test_y = input_data['label']

test_shape = test_x.shape
test_x = np.array_split(test_x,4,axis=1)
test_ax, test_ay, test_az, test_str = test_x[0], test_x[1], test_x[2], test_x[3]


test_x = np.zeros((test_shape[0], 64, 4))
test_x[..., 0] = test_ax
test_x[..., 1] = test_ay
test_x[..., 2] = test_az
test_x[..., 3] = test_str
del test_ax, test_ay, test_az, test_str

test_y = to_categorical(test_y, num_classes=8)

model = tf.keras.models.load_model('odd_even_model.h5')


(loss, accuracy) = model.evaluate(test_x,test_y, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
"""
