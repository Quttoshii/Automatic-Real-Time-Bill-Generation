# Automatic-Real-Time-Bill-Generation
This project is an Automatic Bill Generation web application that utilizes YOLOv8 and a custom augmented dataset. The application detects objects, streamlining the billing process. Built with a focus on efficiency and accuracy, this web app automates the generation of bills, providing a user-friendly interface for seamless data input and output.

# To run
- Train the model on the augmented dataset
- Run the yolov8_training notebook
- Update the WEIGHTS variable in app.py to your yolov8 model location
- Run the flask app.py
