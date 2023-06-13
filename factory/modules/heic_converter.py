   from wand.image import Image
   import os

   SourceFolder="K:/HeicFolder"
   TargetFolder="K:/JpgFolder"

   for file in os.listdir(SourceFolder):
      SourceFile=SourceFolder + "/" + file
      TargetFile=TargetFolder + "/" + file.replace(".HEIC",".JPG")
    
      img=Image(filename=SourceFile)
      img.format='jpg'
      img.save(filename=TargetFile)
      img.close()