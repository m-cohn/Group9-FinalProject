## Run instructions:
1. Make sure you have all the proper libraries installed (you may use requirements.txt). The recommended version of Python, and the one that was used during development, is Python 3.8. You may download necessary libraries as necessary or you can run the following command when in the project directory:
    ```
    pip install -r requirements.txt
    ```
2. Make sure that you have all the augmented audio data features on your local machine. You may do this in one of two ways:
    - Download the prepared data at https://drive.google.com/drive/folders/1-vZL23B2bJwauUFE90_Vm-cO8_Y48xx4?usp=sharing and put it into the project folder root folder
    - Run the "Create Zipped Data" section of run.ipynb (this should take roughly 30 minutes)
        - Before this step, please make sure that you have the original data downloaded and unzipped in the project folder root folder. This data can be downloaded at https://www.kaggle.com/datasets/rtatman/speech-accent-archive. You'll only need the audio recordings for this, so move the main recordings folder into the project's root folder, such that the path to any audio file in the recordings foler is:
        
        project_root_folder/recordings/recordings/audio.mp3 

3. Run run.ipynb. If you downloaded the data in step two, you may skip the first cell, the "Create Zipped Data" section
