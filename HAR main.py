import tkinter as tk
from tkinter.font import Font
import tkinter.messagebox

#Importing basic libraries
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.cm as cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import accuracy_score,confusion_matrix

# %matplotlib inline

accuracy_scores=[0]*3
total=[0]

root=tk.Tk()
#For tkinter window setting
root.geometry("400x300+250+200")          
root.title("Human Activity Recognization")
myfont=Font(family="FreeMono",size=14)
root.resizable(width=False, height=False)
root.configure(bg='black')


# Support Vector Classifier
def  SVM(x_train,y_train,x_test,y_test):
 from sklearn.svm import SVC
 clf = SVC().fit(x_train, y_train)
 prediction = clf.predict(x_test)
 accuracy_scores[2] = accuracy_score(y_test, prediction)*100
 print("Training using Support Vector Classifier done")



# K Nearest Neighbors
def KNN(x_train,y_train,x_test,y_test):
 from sklearn.neighbors import KNeighborsClassifier
 clf = KNeighborsClassifier().fit(x_train, y_train)
 prediction = clf.predict(x_test)
 accuracy_scores[0] = accuracy_score(y_test, prediction)*100
 print("Training using KNeighborsClassifier done")



# Random Forest
def RF(x_train,y_train,x_test,y_test):
 from sklearn.ensemble import RandomForestClassifier
 clf = RandomForestClassifier().fit(x_train, y_train)
 prediction = clf.predict(x_test)
 accuracy_scores[1] = accuracy_score(y_test, prediction)*100
 print("Training using RandomForestClassifier done")
 


#Dividing the dataset into test-train datasets and train machine with different algorithms
def split():
 print("\n TRAINING INFORMATION\n")
 #Importing dataset
 train_data= pd.read_csv('train.csv')
 test_data= pd.read_csv('test.csv')

 #Spliting data into training and testing
 y_train=train_data['Activity']
 x_train= train_data.drop(columns = ['Activity', 'subject'])
 y_test=test_data['Activity']
 x_test= test_data.drop(columns = ['Activity', 'subject'])

 #Traning Models
 KNN(x_train,y_train,x_test,y_test)
 RF(x_train,y_train,x_test,y_test)
 SVM(x_train,y_train,x_test,y_test)
 
 activities = sorted(y_train.unique())
 total[0]=len(list(activities))

 tk.messagebox.showinfo("Train","Training Successfully Completed")

 
#Print accuracy score of all the different algorithms and plot graph of the same
def result():
    print("\n\n\n ACCURACY RESULT")

    new=tk.Tk()
    new.geometry("700x500+250+200") 

    print('\nK Nearest Neighbors Classifier accuracy: {}%'.format(accuracy_scores[0]))
    print('\nRandom Forest Classifier accuracy: {}%'.format(accuracy_scores[1]))
    print('\nSupport Vector Classifier accuracy: {}%'.format(accuracy_scores[2]))
    

    colors = cm.rainbow(np.linspace(0, 1, 3))
    labels = ['K Nearest Neighbors', 'Random Forest', 'Support Vector']

    
    fig = Figure(figsize=(6,6))
    plt = fig.add_subplot(111)
    plt.bar(labels,
        accuracy_scores,
        color = colors)
    plt.set_xlabel('Classifiers')
    plt.set_ylabel('Accuracy')
    plt.set_title('Accuracy of various algorithms')
 
    canvas = FigureCanvasTkAgg(fig, master=new)
    canvas.get_tk_widget().pack()
    canvas.draw()
    new.mainloop()

    
    
    
    
# Identify all the unqiue activities and in sorted order
def count():

    tk.messagebox.showinfo("Activities","\n1. Lying \n2. Sitting \n3. Standing \n4. Walking \n5. Walking Upstair \n6. Walking Downstair")
    



#Main window of GUI interface
tk.Button(root,text="Spliting Train-Test dataset And Train Machine",font="times 12",activeforeground="white",bg="lightblue",command=split).place(x=70,y=0)
tk.Button(root,text="Activities",font="times 12",activeforeground="white",bg="lightgreen",command=count).place(x=170,y=100)
tk.Button(root,text="Result And Visualization",font="times 12",activeforeground="white",bg="violet",command=result).place(x=120,y=200)




root.mainloop()#tkinter window ends
