# SOCIS_2019
This project lies on the complexity of detecting whales in satelite images. In particular, only the Mediterranean Sea is considered. 

## Steps
* Image gathering from the first 3km margin from the coast into the ocean coming from Google Maps and Bing Maps. The former uses the GCP framework and the static maps API. Then with the unique key assigned for the use of the API many requests can be made with the exact bounding box coordinates (latitud and longitude of the top left corner and bottom right corner) to extract the image. The latter uses the code in [this repository](https://github.com/manurare/Satellite-Aerial-Image-Retrieval.git) to read a .csv file where each line indicates the coordinates of a particular bounding box to get the current image.
* Due to the lack of spatial resolution in satellite images a superresolution step is required to increased high frequencies and restore fine details. SRGAN and Pixel Recursive Super Resolution were tested. This step will ease the performance of different object detection algorithms.
* Apply object detection algorithms with multilabelling to set a threshold of confidence to reduce as much as possible false positives. 
* Use chlorophyll data from Sentinel 3 to nail down the most probable areas for the whales to be. Thus, the whole pipeline will not need to be executed all over the Mediterranean Sea images but in the regions where whale location probability is higher.

## Results
In this [link](https://www-iuem.univ-brest.fr/datacube/sample-apps/rshiny_app/) you can check an interactive map where high probability areas of whale presence are highlighted. This areas are computed taking on account the chl density (chlorophyll).
