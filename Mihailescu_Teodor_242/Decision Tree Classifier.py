#importuri
import numpy as np
from sklearn.tree import DecisionTreeClassifier#pentru a folosi modelul acesta
from sklearn.metrics import accuracy_score, classification_report#pentru raport
from PIL import Image#pentru a citi
from sklearn.metrics import confusion_matrix # pentru matricea de confuzie
import matplotlib.pyplot as plt#pentru a plota


train_data = []
train_labels = []

path_img_file = "unibuc-brain-ad/data/train_labels.txt"#path-ul pentru imagini
train_data_dir = "unibuc-brain-ad/data/data/"#path-ul pentru fisierul de train pentru model

train_labels_data = np.loadtxt(path_img_file, delimiter=",", skiprows=1, dtype=str)#facem skip la prima linie ce continte id/clasa, apoi delimitam dupa ",", obtinand astfel numele imaginii si label ul, apoi convertim datele la str

for img_path, label in train_labels_data:
    img = Image.open(train_data_dir + img_path + ".png")#deschidem imaginea
    img_arr = np.array(img)
    mean_intensity = np.mean(img_arr)#calculez media a intensității pixelilor
    std_intensity = np.std(img_arr)#calculez deviația standard a intensității pixelilor
    train_data.append([mean_intensity, std_intensity])
    train_labels.append(int(label))

#convertim la np array
train_data = np.array(train_data)
train_labels = np.array(train_labels)


# Calculam class weights
total_samples = len(train_labels)
class_1_samples = np.sum(train_labels)
class_0_samples = total_samples - class_1_samples
print(class_0_samples, class_1_samples)
class_weights = {0: 1, 1: round(class_0_samples/class_1_samples, 1)}#calculăm prioritățile de clasă și aplicăm ponderi ale clasei fiecărui eșantion din datele de antrenament pe baza etichetei sale
print(total_samples, class_1_samples, class_0_samples)#numar total de imagini, numar total de imagini cu label ul 1 si numar total de imagini cu label-ul 0

#Calculam probabilitatile claselor pentru setul de antrenare
class_priors = np.array([np.mean(train_labels == i) for i in np.unique(train_labels)])#calculam media valorilor din array, in acest caz media de True  din array-ul returnat( toate clasele unice din train_labels cu True daca elementul i este in train_labels si False altfel)
class_weights = np.array([class_weights[i] for i in np.unique(train_labels)])
sample_weights = np.array([class_weights[i] / class_priors[i] for i in train_labels])#Calculam ponderile mostrelor pe baza greutatilor si a probabilitatilor claselor din setul de antrenare

model = DecisionTreeClassifier()#antrena,
model.fit(train_data, train_labels, sample_weight=sample_weights)


validation_data_dir = "unibuc-brain-ad/data/data/"
validation_labels_file = "unibuc-brain-ad/data/validation_labels.txt"#path catre validation
validation_labels_data = np.loadtxt(validation_labels_file, delimiter=",", skiprows=1, dtype=str)#analog

validation_data = []
validation_labels = []

for img_path, label in validation_labels_data:#analog
    img = Image.open(validation_data_dir + img_path + ".png")
    img_arr = np.array(img)
    mean_intensity = np.mean(img_arr)
    std_intensity = np.std(img_arr)
    validation_data.append([mean_intensity, std_intensity])
    validation_labels.append(int(label))

validation_data = np.array(validation_data)
validation_labels = np.array(validation_labels)

#facem prezicerile
predictions = model.predict(validation_data)

#calculam acuratetea si o afisam
accuracy = accuracy_score(validation_labels, predictions)
print("Validation accuracy: {:.2f}%".format(accuracy * 100))


print(classification_report(validation_labels, predictions))

cm = confusion_matrix(validation_labels, predictions)

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=["Class 0", "Class 1"],
       yticklabels=["Class 0", "Class 1"],
       xlabel='Predicted label',
       ylabel='True label',
       title='Confusion matrix')
plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
         rotation_mode="anchor")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2. else "black")

fig.tight_layout()
plt.show()


sample_submission_file = "unibuc-brain-ad/data/sample_submission.txt"#citim subbmision
sample_submission_data = np.loadtxt(sample_submission_file, delimiter=",", skiprows=1, dtype=str)#analog

test_data = []
test_labels = []

for img_path, label in sample_submission_data:#analog
    img = Image.open(train_data_dir + img_path + ".png")
    img_arr = np.array(img)
    mean_intensity = np.mean(img_arr)
    std_intensity = np.std(img_arr)
    test_data.append([mean_intensity, std_intensity])
    test_labels.append(int(label))

test_data = np.array(test_data)
test_labels = np.array(test_labels)

test_predictions = model.predict(test_data)#facem preziceri


with open("test_predictions.txt", "w") as f:#salvam
    f.write("id,class\n")
    for i, prediction in enumerate(test_predictions):
        f.write("{},{}\n".format(sample_submission_data[i][0], prediction))

