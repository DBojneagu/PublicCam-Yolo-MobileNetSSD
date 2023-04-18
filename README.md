# PublicCam-Yolo-MobileNetSSD

# Description
This is a python program that uses 2 models separately on a public webcam, to identify cars moving by. It uses OpenCV to open up and create the connection to the Public WebCam, and MobileNETSSD Model for Car recognitioning for the first model, and YoloV3 as the second model.

# How to install and run the application

In order to run the application, you need to download the zipfile from this github repo, install the necessary packages ( numpy and opencv).

So, step by step, it would be : 

1. Download the zipfile
2. Download yolov3.weights ( I could not push it into this repo, even with github LFS. So if you want to run the program you need to go and install yolov3.weights)
You can download it from here : https://drive.google.com/file/d/1tQ5wa_kBLDst9OT4_3EENjRLvHFfkeHH/view?usp=sharing 
3. Open it using a code editor ( i'm going to show it using PyCharm ) 
4. Install the needed packages by going into the terminal, and typing : 

![image](https://user-images.githubusercontent.com/93039914/232247811-0fdbc6a7-b015-4dc6-992a-9fbdad86116c.png)
![image](https://user-images.githubusercontent.com/93039914/232247838-bbdd8d25-7957-4614-ad72-8d4523986d72.png)

4. After successfully installing the needed packages, now you can press the run button in the top right corner  : ![image](https://user-images.githubusercontent.com/93039914/232247867-0e6ddba6-ea5f-4e6b-956b-bb9c79d6f3db.png)

5. This is how it should look : 
![Test12312313](https://user-images.githubusercontent.com/93039914/232247981-5278af24-af0c-4268-8031-d3574908239c.jpg)

6. If you want to stop the program, you can either press the "q" button , or press the red square inside the program : ![image](https://user-images.githubusercontent.com/93039914/232248011-99baf9c4-dad2-49aa-84b8-d35b56ddf8e4.png)
7. Make sure that you put the yolov3.weight file inside the main program. It should look like this : ![image](https://user-images.githubusercontent.com/93039914/232814939-949bf497-f3fe-49af-86f4-855cdb666386.png)


# How does the algorithm work 
This program uses the YoloV3 model and MobileNetSSD. They are using convulation neural networks ( CNN's ) , but they have a different processing process so to say. 
We have the config and weights needed for the models, and after that we run a for loop going through the detections and if they're score is good enough we are drawing a box around the object and text, indicating the object detected. I've set the conf threshold 0.5 by default. 

# MobileNetSSD 
MobileNetSSD Algorithm : ![image](https://user-images.githubusercontent.com/93039914/232248149-973d3329-ec67-4f99-beda-bf4f339deff6.png)

Here, after we read each frame, in the while true loop, using the blob ( that is an array of image data ), we resize it and normalize it, ( for example the 300 by 300 is the resize value, the 0.007843 is equivalent to dividing each pixel by 255 , that resulting in every pixel being in the range of 0 to 1, so we pretty much take care of the lighting of the image so the model can process it without problems. After that, we iterate through each detection, and if the confidence score of it ( the confidence score is a score based on how good the image is, how accurate it is). If it is bigger then the setted threshold, we get the class id and draw a box around the objet detected. 


# YoloV3

![image](https://user-images.githubusercontent.com/93039914/232248628-07cbae35-b0c1-4097-aa57-8add5ec6abf2.png)

The same idea is to YoloV3 aswell. The YoloV3 works in a different way when it comes to the idea behind the model. The model takes an image, then sets a grid , either 4by4 or 3by3 or whatever, and after that, in the second layer of the convulational neural networks, it has really simple and abstract filters, for example, a straightline etc. Through each layer of the CNN the filters become more specific. The way the algorithm draws the box around the object, is by detecting the center of it, by going through each grid and checking if the center of the object is inside. After all of this we are going to still have a problem, and that would be that we are going to have multiple boxes around the same object, The way we deal with this problem is called non-maximum suppression (NMS). The way this works is if we have multiple detections for the same object, we compare the confidence score between them and the one with bigger score is the only one shown.

# Synchronizing the two models. 

The way i deal with the synchronization problem that is inevitable ( because 1 model is going to process faster then the other ) i simply put a wait(1) after each model, so it waits 1 second before getting the next frame. In this way we are always going to have them synched.
![image](https://user-images.githubusercontent.com/93039914/232248908-da3d3e75-3686-47b2-b112-b5555bce8254.png)

# Time Sheet
- 4/7/2023 12:00:00 - 13:00:00  PM - > First Look on the project and the required features
- 4/7/2023 13:00:00 - 14:00:00  PM - > Figured out how to connect the program to a live public webcam and searched for a public webcam.
- 4/7/2023 14:00:00 - 16:00:00  PM - > Found the first model ( MobileNetSSD ) and found how to apply in into the live public webcam. 
- 4/8/2023 12:00:00 - 14:00:00  PM - > Found the second model, ( YoloV3 ) and applied it into the public webcam, the same time as MobileNetSSD, by creating 2 separate windows
- 4/8/2023 14:00:00 - 14:45:00  PM - > Synched the two windows.
- 4/14/2023 13:00:00 - 15:00:00 PM - > Learned how the YoloV3 works
- 4/15/2023 20:00:00 - 21:00:00 PM - > Learned how the MobileNetSSD works
- 4/15/2023 21:00:00 - 22:00:00 -> Learned about CNN's 


# Bugs during development
1. One of the big walls i hitted during the development was about finding the config/cfg/weights needed to apply the model. I needed time to figure out for example, even if 1 of the objects needed are not present in the class names list, the program is going to drop and not work. 






