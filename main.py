import matplotlib
import torch
import pandas as pd
import Software_Practical

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    rgb_img = True
    img_height = 100
    img_width = 100
    total_img = 5
    line_width = 2
    line_color = 'white'
    voronoi_cell_no = 8
    base_colors = list(matplotlib.colors.BASE_COLORS)
    css4_colors = list(matplotlib.colors.CSS4_COLORS)
    color_list = [base_colors[0], base_colors[1], base_colors[2], base_colors[3], base_colors[4], base_colors[5],
                  css4_colors[11], css4_colors[21]]

    '''
    tensor_1: Tensor of the all images, return shape is (total_img, 3, img_width, img_height)
    tensor_2: Tensor of the labels of the all images. If there are repetitive colors tensor will return the same value.
    Return shape is (total_img, img_width, img_height)
    tensor_3: Tensor of the labels of the all images. If there are repetitive colors tensor will return different
    values. In other words, in each image there will be as many labels as random points.
    Return shape is (total_img, img_width, img_height)
    '''

    if rgb_img:
        tensor_2, tensor_3 = Software_Practical.vor_rgb_img_creator("dataset", total_img=total_img, rand_point_no=voronoi_cell_no,
                                                                    img_height=img_height,
                                                                    img_width=img_width, line_width=line_width,
                                                                    line_color=line_color, color_list=color_list)
        tensor_1 = Software_Practical.tensor_out_colored('dataset/Image_rgb', total_img=total_img, img_height=img_height,
                                                         img_width=img_width)
        x_np = tensor_2[0].numpy()
        x_df = pd.DataFrame(x_np)
        x_df.to_csv('tensor2.csv')

        x_np = tensor_3[0].numpy()
        x_df = pd.DataFrame(x_np)
        x_df.to_csv('tensor3.csv')
    else:
        Software_Practical.vor_bw_img_creator(total_img=total_img, rand_point_no=voronoi_cell_no, img_height=img_height,
                                              img_width=img_width, line_width=line_width, line_color=line_color)
        tensor = Software_Practical.tensor_out_bw(total_img=total_img, img_height=img_height, img_width=img_width)
