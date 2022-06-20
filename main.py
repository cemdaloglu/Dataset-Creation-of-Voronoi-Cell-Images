import Software_Practical

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    rgb_img = True
    img_height = 100
    img_width = 100
    total_img = 5
    line_width = 1
    line_color = 'blue'
    voronoi_cell_no = 8

    '''
    tensor_1: Tensor of the all images, return shape is (total_img, 3, img_width, img_height)
    tensor_2: Tensor of the labels of the all images. If there are repetitive colors tensor will return the same value.
    Return shape is (total_img, img_width, img_height)
    tensor_3: Tensor of the labels of the all images. If there are repetitive colors tensor will return different
    values. In other words, in each image there will be as many labels as random points.
    Return shape is (total_img, 3, img_width, img_height)
    '''

    if rgb_img:
        tensor_2, tensor_3 = Software_Practical.vor_rgb_img_creator(total_img=total_img, rand_point_no=voronoi_cell_no,
                                                                    img_height=img_height,
                                                                    img_width=img_width, line_width=line_width,
                                                                    line_color=line_color)
        tensor_1 = Software_Practical.tensor_out_colored(total_img=total_img, img_height=img_height,
                                                         img_width=img_width)
    else:
        Software_Practical.vor_bw_img_creator(total_img=total_img, rand_point_no=voronoi_cell_no, img_height=img_height,
                                              img_width=img_width, line_width=line_width, line_color=line_color)
        tensor = Software_Practical.tensor_out_bw(total_img=total_img, img_height=img_height, img_width=img_width)
