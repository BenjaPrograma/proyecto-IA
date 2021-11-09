import os, zipfile

for item in os.listdir("matterportdata/v1/scans"):
  if item.endswith(".zip"):
      print(item)
    #file_name = os.path.abspath(item)
    #zip_ref = zipfile.ZipFile(file_name) # create zipfile object
    #zip_ref.extractall(dir_name) # extract file to dir
    #zip_ref.close() # close file
    #os.remove(file_name) # delete zipped file