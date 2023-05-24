#importuri
import csv #pentru a crea csv cerut pentru submisii
import seaborn as sns #pentru a plota
import tensorflow as tf #pentru a folosi tool-uri pentru machine learning
import numpy as np
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D #pentru layere
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report, confusion_matrix #pentru a determina matricea de comfuzie
import matplotlib.pyplot as plt #pentru plot

#folosim aceast conditie pentru a lista device-urile GPU de pe dispozitiv si a aloca memorie, deoarece programul ruleaza
#mult mai rapid pe GPU(graphic processing unit) (fiind imagini datele noastre)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')#setăm ca acestea să fie vizibile, totodată,alocând memorie pentru primul GPU din lista de dispozitive
        tf.config.experimental.set_memory_growth(gpus[0], True)#alocarea dinamică de memorie, ceea ce înseamnăcă TensorFlow va aloca memorie pentru model pe măsură ce este necesar întimpul antrenării
    except RuntimeError as e:
        print(e)

path_img_file = "unibuc-brain-ad/data/data/" #path-ul pentru imagini
train_labels_file = "unibuc-brain-ad/data/train_labels.txt"#path-ul pentru fisierul de train pentru model


train_labels_data = np.loadtxt(train_labels_file, delimiter=",", skiprows=1, dtype=str)#facem skip la prima linie ce continte id/clasa, apoi delimitam dupa ",", obtinand astfel numele imaginii si label ul, apoi convertim datele la str
train_data = []#vom retine imaginile
train_labels = []# vom retine label urile

for img_path, label in train_labels_data:
    img = tf.keras.preprocessing.image.load_img(path_img_file + img_path + ".png", color_mode="grayscale", target_size=(150, 150))# formam path-ul pentru fiecare imagine cu color modu-l gray si marimea 150*150
    img_arr = tf.keras.preprocessing.image.img_to_array(img)
    train_data.append(img_arr)
    train_labels.append(int(label))
#convertim datele la np array
train_data = np.array(train_data)
train_labels = np.array(train_labels)

#utilizat pentru a genera loturi de date noi
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,#specifică intervalul de rotații aleatoare de aplicatimaginilor, în grade
    width_shift_range=0.1,#specifică intervalul de deplasări orizontalealeatoare de aplicat imaginilor, ca fracțiune din lățimea totală
    height_shift_range=0.1,#specifică intervalul de deplasări verticalealeatoare de aplicat imaginilor, ca fracțiune din înălțimea totală
    zoom_range=0.1,#specifică intervalul de zoom-uri aleatoare deaplicat imaginilor, ca fracțiune din dimensiunea originală
    horizontal_flip=True,#specifică dacă să se inverseze aleatoriuimaginile orizontal în timpul antrenamentului.
    vertical_flip=True,#specifică dacă să se inverseze aleatoriu imaginilevertical în timpul antrenamentului
    validation_split=0.2)#specifică fracțiunea de date de antrenamentde utilizat pentru validare


# Calculate class weights
total_samples = len(train_labels)
class_1_samples = np.sum(train_labels)
class_0_samples = total_samples - class_1_samples
print(class_0_samples, class_1_samples)
class_weights = {0: 1, 1: round(class_0_samples/class_1_samples, 1)}#calculăm prioritățile de clasă și aplicăm ponderi ale clasei fiecărui eșantion din datele de antrenament pe baza etichetei sale
print(total_samples, class_1_samples, class_0_samples)#numar total de imagini, numar total de imagini cu label ul 1 si numar total de imagini cu label-ul 0


model = Sequential() #defininim un model gol, pe care  vom adăuga treptat layere:
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(150, 150, 1)))#strat Conv2D cu 32 de filtre de dimensiune 3x3, activare ReLU și "same" padding (acesta va menține dimensiunea imaginilor la ieșirea)
model.add(BatchNormalization())#ajută la normalizarea valorilor de ieșire pentru fiecare filtru înainte de a fi trecută prin stratul de activare
model.add(MaxPooling2D((2, 2)))# reducem dimensiunea imaginilor cu un factor de 2
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())#După aceste straturi convoluționale, trebuie să aplatizăm imaginea pentru a o transforma intr un tensor unidimensional
model.add(Dense(128, activation='relu'))#adaugam un strat cu 128 neuroni si functia de activare relu complet conectat la reteaua noastra
model.add(BatchNormalization())
model.add(Dropout(0.5))# adauga un strat de regularizare numit dropout in reteaua neuronala
model.add(Dense(1, activation='sigmoid')) # adaugam un strat complet conectat cu 1 neuraon si cu functia sigmoid care poate fi interpretata ca probabilitatea clasificarii obiectelor in cele doua clase posibile

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(
     learning_rate=0.001 ),#rată de învățare
              loss='binary_crossentropy',#funcția de pierdere
              metrics=['accuracy'])#metricile care vor fi utilizate pentru aevalua performanța modelului


train_generator = datagen.flow(train_data, train_labels, batch_size=32, subset='training',shuffle=False)#generam loturi de date augmentate prin aplicarea parametrilor de augmentare la datele originale
val_generator = datagen.flow(train_data, train_labels, batch_size=32, subset='validation',shuffle=False)#genereazam loturi de date augmentate prin aplicarea parametrilor de augmentare la datele originale.

history=model.fit(train_generator, #antrenam modelul
          epochs=15, #numarul de epoci pe care invata modelul.O epoca reprezinta un ciclu complet de antrenare al modelului pe intregul set de date de antrenare
          batch_size=32,# dimensiunea loturilor pentru antrenarea modelului. Numarul de exemple de antrenare care sunt procesate deodata.
          steps_per_epoch=len(train_generator),#numarul de pasi pentru fiecare epoca de antrenare. Acesta determina cati loturi de date sunt procesate intr-o epoca. In cazul nostru, aceasta este setata la numarul de loturi din generatorul de date pentru antrenare
          validation_data=val_generator,#generatorul de date pentru validarea modelului. Acesta furnizeaza loturi de imagini si etichete pentru a evalua performanta modelulu
          validation_steps=len(val_generator),# numarul de pasi pentru validarea modelului. Acesta determina cati loturi de date sunt procesate in timpul validarii. In cazul nostru, aceasta este setata la numarul de loturi din generatorul de date pentru validare
          class_weight=class_weights)#ponderile claselor pentru a echilibra distributia claselor in setul de date de antrenare. Aceasta este utila in cazul in care avem un dezechilibru in distributia claselor in setul de date de antrenare

model.save("model.h5")#salvam modelul nostru

train_loss = history.history['loss']#retinem loss si val loss dupa antrenament ca sa plotam raportul
val_loss = history.history['val_loss']

epochs = range(1, len(train_loss) + 1)#plotam un grafic cu aceste valori per epoch
plt.plot(epochs, train_loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'k', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()




model = tf.keras.models.load_model('model.h5')#incarcam modelul salvat anterior
 

validation_labels_file = "unibuc-brain-ad/data/validation_labels.txt"#path catre validation
validation_labels = np.loadtxt(validation_labels_file, delimiter=",", skiprows=1, dtype=str) #analog
predictions = []
for img_path, label in validation_labels:
    img = tf.keras.preprocessing.image.load_img(path_img_file + img_path + ".png", color_mode="grayscale", target_size=(150, 150))# formam path-ul pentru fiecare imagine cu color modu-l gray si marimea 150*150
    img_arr = tf.keras.preprocessing.image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    prediction = model.predict(img_arr)
    predictions.append(int(round(prediction[0][0])))



with open('validation_predictions.txt', 'w') as f:# am salvat intr un fisier text ca sa vedem daca nimereste doar 0 sau 1
    f.write("id\tclass\tprediction\n")#pentru ca pt 86% accurate nimerea doar 0
    for i in range(len(validation_labels)):
        img_name = validation_labels[i][0]
        true_class = validation_labels[i][1]
        pred_class = predictions[i]
        f.write(f"{img_name}\t{true_class}\t{pred_class}\n")

accuracy = np.mean(predictions == validation_labels[:, 1].astype(int))#calculam accuratetea
print(accuracy)


y_true = [int(label[1]) for label in validation_labels]#calculam cate sunt match intre clasa prezisa si cea adevarata
y_pred = predictions


cm = confusion_matrix(y_true, y_pred)#generam matricea de confuzie

sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')#plotam
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()


#analog ca la validation, doar ca nu avem cum sa calculam accuratetea
predictions = []

sample_submission_file = "unibuc-brain-ad/data/sample_submission.txt"
sample_submission_labels = np.loadtxt(sample_submission_file, delimiter=",", skiprows=1, dtype=str)



predictions = []
for img_path, label in sample_submission_labels:
    img = tf.keras.preprocessing.image.load_img(path_img_file + img_path + ".png", color_mode="grayscale", target_size=(150, 150))# formam path-ul pentru fiecare imagine cu color modu-l gray si marimea 150*150
    img_arr = tf.keras.preprocessing.image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    prediction = model.predict(img_arr)
    predictions.append(int(round(prediction[0][0])))

 
with open('validation_predictions.csv', 'w', newline='') as f:# salvam csv ul cerut
    writer = csv.writer(f)
    writer.writerow(['id', 'class'])
    for i in range(len(sample_submission_labels)):
        img_name = sample_submission_labels[i][0]
        true_class = sample_submission_labels[i][1]
        pred_class = predictions[i]
        writer.writerow([img_name, pred_class])



y_true = [int(label[1]) for label in sample_submission_labels]
y_pred = predictions

cm = confusion_matrix(y_true, y_pred)


sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()




