import tensorflow as tf

#using this we can directly load data into minst variable
minst=tf.keras.datasets.mnist

(X, Y), (x,y) = minst.load_data()

X=X/255.0
x=x/255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(256, activation='relu'),
                                    tf.keras.layers.Dense(128, activation='sigmoid'),
                                    tf.keras.layers.Dense(10, activation='softmax')
                                    ])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x,y,epochs=15)
validation_data=(x,y)

test_loss, test_acc = model.evaluate(x, y)
print(f'Test accuracy: {test_acc}')
model.save('Models/model_3_layer.h5'
