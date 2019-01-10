import os, gdal
from glob import glob

in_path = os.getcwd()+os.sep #r'/home/filfy/github_forks/Semantic-Segmentation-Suite/Elwha'+os.sep

out_path = os.getcwd()+os.sep+'R_G_B_tiles'+os.sep  #r'/home/filfy/github_forks/Semantic-Segmentation-Suite/Elwha/label_tiles'+os.sep


tile_size_x = 512 
tile_size_y = 512 

input_filenames = glob('images/*.tif')

counter=0
for input_filename in input_filenames:
    output_filename = 'tile'+str(counter)+'_'
    ds = gdal.Open(in_path + input_filename)
    band = ds.GetRasterBand(1)
    xsize = band.XSize
    ysize = band.YSize

    for i in range(0, xsize, tile_size_x):
        for j in range(0, ysize, tile_size_y):
            #com_string = "gdal_translate -of GTIFF -srcwin " + str(i)+ ", " + str(j) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + str(in_path) + str(input_filename) + " " + str(out_path) + str(output_filename) + str(i) + "_" + str(j) + ".tif"
            com_string = "gdal_translate -of GTIFF -srcwin " + str(i)+ " " + str(j) + " " + str(tile_size_x) + " " + str(tile_size_y) + " " + str(in_path) + str(input_filename) + " " + str(out_path) + str(output_filename) + str(i) + "_" + str(j) + ".tif"
            os.system(com_string)        
    counter +=1 

