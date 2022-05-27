# SOFTWARE-pRACT-CAL
  This repository creates Voronoi cells with preferred image size, number of cells, and line width. There are few parameters in the main function that should be decided by the user.
  
Parameters:
  * rgb_img should be chosen as True if one wants to construct colored Voronoi images.
  * img_height, img_width are used to create the desired image size, they can be chosen any integer.
  * total_img is the total number of images one wants to construct.
  * line_width is the width of the line which separates two Voronoi cells, it can be chosen as 0 as well if one wants to construct Voronoi cells without any separation lines.
  * line_color must be a string, can be any color.
  * voronoi_cell_no is the total number of random points and also total number of Voronoi cells one wants in the output image.
  
  <ins>vor_rgb_img_creator</ins> function should be called to construct a colored Voronoi image with the decided parameters. This function does not show any plots but saves them as .png files to the current directory. Image is saved as Image_rgb_#.png after the function execution.
 
  To get a tensor output, one needs to use <ins>tensor_out_colored</ins> function. This function returns a tensor with size (total_img, 3, img_width, img_height).
  
  <ins>vor_bw_img_creator</ins> is similar as the rgb one, the only difference is it saves a black and white images.
  
  <ins>tensor_our_bw</ins> is also similar as before but the output is (total_img, 1, img_width, img_height).

  Sample results can be seen below.

<p align="center">
  <img src="https://github.com/cemdaloglu/SOFTWARE-pRACT-CAL/blob/main/Results/Image_rgb%201.png" width="350" title="hover text">
  <img src="https://github.com/cemdaloglu/SOFTWARE-pRACT-CAL/blob/main/Results/Image_rgb%202_lines.png" width="350" title="hover text">
  <img src="https://github.com/cemdaloglu/SOFTWARE-pRACT-CAL/blob/main/Results/Image%201.png" width="350" title="hover text">
  <img src="https://github.com/cemdaloglu/SOFTWARE-pRACT-CAL/blob/main/Results/Image%202_lines.png" width="350" title="hover text">
</p>
