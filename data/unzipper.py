import os, zipfile

for path, subdirs, files in os.walk("./data/matterportdata/v1/scans"):
    for name in files:
        if name.endswith(".zip"):
            print(name)
            file_name = os.path.join(path, name)
            print(file_name)
            zip_ref = zipfile.ZipFile(file_name) # create zipfile object
            zip_ref.extractall("./data/matterport_unzip/") # extract file to dir
            zip_ref.close() # close file
            os.remove(file_name) # delete zipped file