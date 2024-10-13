import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from matplotlib.widgets import Button
import tensorflow as tf

model=tf.keras.models.load_model('Models/model_4_layer.h5')

canvas=np.zeros((28,28))

def draw(event):
    if event.xdata is not None and event.ydata is not None:
        x=int(event.xdata)
        y=int(event.ydata)
        if 0 <= x < 28 and 0 <= y < 28:
            canvas[y, x] = 1  # Draw on the canvas
            ax.clear()
            ax.imshow(canvas, cmap='gray')
            plt.draw()
def clear_canvas(event):
    global canvas
    canvas=np.zeros((28,28))
    ax.clear()
    ax.imshow(canvas,cmap='gray')
    plt.draw()

def predict(event):
    global canvas
    input_image=canvas.reshape(1,28,28)
    input_image=input_image/1.0
    prediction=model.predict(input_image)
    predicted_digit=np.argmax(prediction)
    print(f"Predicted digit is {predicted_digit}")

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
ax.imshow(canvas,cmap='gray')
ax.set_xticklabels([])
ax.set_yticklabels([])

cid = fig.canvas.mpl_connect('motion_notify_event', draw)

clear_button_ax=plt.axes([0.5, 0.05, 0.3, 0.075])
clear_btn=Button(clear_button_ax,'Clear')
clear_btn.on_clicked(clear_canvas)

predict_button_ax=plt.axes([0.1, 0.05, 0.3, 0.075])
predict_button_ax=Button(predict_button_ax,'Predict')
predict_button_ax.on_clicked(predict)

plt.show()